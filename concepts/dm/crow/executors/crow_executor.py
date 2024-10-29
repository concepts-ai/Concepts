#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : crow_executor.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/17/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""This file contains the executor classes for the Crow framework.

All the executors are based on :class:`~concepts.dsl.tensor_value.TensorValue` classes.
It supports all expressions defined in :mod:`~concepts.dsl.expressions`, including the basic
function application expressions and a few first-order logic expression types. The executors
are designed to be "differentiable", which means we can directly do backpropagation on the
computed values.

The main entry for the executor is the :class:`CrowExecutor` class.
Internally it contains two executor implementations: the basic one, and an "optimistic" one,
which handles the case where the value of a variable can be unknown and "optimistic".
"""

import itertools
import contextlib
from typing import Any, Optional, Union, Iterator, Sequence, Tuple, List, Mapping, Dict, Callable

import torch
import jactorch
from jacinle.logging import get_logger

import concepts.dsl.expression as E
from concepts.dsl.dsl_types import BOOL, INT64, FLOAT32, STRING, AutoType, TensorValueTypeBase, ScalarValueType, VectorValueType, NamedTensorValueType, PyObjValueType, BatchedListType, QINDEX, Variable
from concepts.dsl.expression import Expression, BoolOpType, QuantificationOpType
from concepts.dsl.expression_visitor import ExpressionVisitor
from concepts.dsl.value import ListValue
from concepts.dsl.tensor_value import TensorizedPyObjValues, TensorValue, MaskedTensorStorage
from concepts.dsl.tensor_state import StateObjectReference, StateObjectList, TensorState, NamedObjectTensorState
from concepts.dsl.tensor_value_utils import expand_argument_values
from concepts.dsl.constraint import OPTIM_MAGIC_NUMBER_MAGIC, is_optimistic_value, OptimisticValue, cvt_opt_value, Constraint, EqualityConstraint, ConstraintSatisfactionProblem, SimulationFluentConstraintFunction
from concepts.dsl.parsers.parser_base import ParserBase
from concepts.dsl.executors.tensor_value_executor import BoundedVariablesDictCompatible, TensorValueExecutorBase, TensorValueExecutorReturnType

from concepts.dm.crow.crow_function import CrowFeature, CrowFunction, CrowFunctionEvaluationMode
from concepts.dm.crow.crow_domain import CrowDomain, CrowState
from concepts.dm.crow.executors.python_function import CrowPythonFunctionRef, CrowPythonFunctionCrossRef, CrowSGC


logger = get_logger(__file__)

__all__ = [
    'CrowExecutor', 'CrowExecutionDefaultVisitor', 'CrowExecutionCSPVisitor',
]


class CrowExecutor(TensorValueExecutorBase):
    def __init__(self, domain: CrowDomain, parser: Optional[ParserBase] = None, load_external_function_implementations: bool = True):
        """Initialize a Crow expression executor.

        Args:
            domain: the domain of this executor.
            parser: the parser to be used. This argument is optional. If provided, the execute function can take strings as input.
            load_external_function_implementations: whether to load the external function implementations defined in the domain file.
        """
        super().__init__(domain, parser)
        self._csp = None
        self._optimistic_execution = False
        self._sgc = None

        self._default_visitor = CrowExecutionDefaultVisitor(self)
        self._csp_visitor = CrowExecutionCSPVisitor(self)
        self._register_default_function_implementations()
        if load_external_function_implementations:
            self._register_external_function_implementations_from_domain()
        self._effect_update_from_simulation = False
        self._effect_update_from_execution = False
        self._effect_action_index = None

    @property
    def csp(self) -> Optional[ConstraintSatisfactionProblem]:
        """The CSP that describes the constraints in past executions."""
        return self._csp

    @property
    def sgc(self) -> Optional[CrowSGC]:
        """The SGC (state-goal-constraints) context."""
        return self._sgc

    @property
    def optimistic_execution(self) -> bool:
        """Whether to execute the expression optimistically (i.e., treat all CSP constraints True)."""
        return self._optimistic_execution

    @contextlib.contextmanager
    def with_csp(self, csp: Optional[ConstraintSatisfactionProblem]):
        """A context manager to temporarily set the CSP of the executor."""
        old_csp = self._csp
        self._csp = csp
        yield
        self._csp = old_csp

    @contextlib.contextmanager
    def with_sgc(self, sgc: Optional[CrowSGC]):
        """A context manager to temporarily set the SGC of the executor."""
        old_sgc = self._sgc
        self._sgc = sgc
        yield
        self._sgc = old_sgc

    def _register_default_function_implementations(self):
        for t in itertools.chain(self.domain.types.values(), [BOOL, INT64, FLOAT32, STRING]):
            if isinstance(t, NamedTensorValueType) and isinstance(t.parent_type, ScalarValueType) or isinstance(t, ScalarValueType):
                # NB(Jiayuan Mao @ 2024/07/11): Can't use the lambda function here, because the value of `t` is not captured.
                self.register_function_implementation(f'type::{t.typename}::add', CrowPythonFunctionRef(_CrowArithFunctionImpl(t, lambda x, y: x + y), support_batch=True, auto_broadcast=True, unwrap_values=False))
                self.register_function_implementation(f'type::{t.typename}::sub', CrowPythonFunctionRef(_CrowArithFunctionImpl(t, lambda x, y: x - y), support_batch=True, auto_broadcast=True, unwrap_values=False))
                self.register_function_implementation(f'type::{t.typename}::mul', CrowPythonFunctionRef(_CrowArithFunctionImpl(t, lambda x, y: x * y), support_batch=True, auto_broadcast=True, unwrap_values=False))
                self.register_function_implementation(f'type::{t.typename}::div', CrowPythonFunctionRef(_CrowArithFunctionImpl(t, lambda x, y: x / y), support_batch=True, auto_broadcast=True, unwrap_values=False))
                self.register_function_implementation(f'type::{t.typename}::neg', CrowPythonFunctionRef(_CrowUnaryFunctionImpl(t, lambda x: -x), support_batch=True, auto_broadcast=True, unwrap_values=False))
                self.register_function_implementation(f'type::{t.typename}::equal', CrowPythonFunctionRef(_CrowComparisonFunctionImpl(BOOL, lambda x, y: torch.eq(x, y)), support_batch=True, auto_broadcast=True, unwrap_values=False))
                self.register_function_implementation(f'type::{t.typename}::not_equal', CrowPythonFunctionRef(_CrowComparisonFunctionImpl(BOOL, lambda x, y: torch.ne(x, y)), support_batch=True, auto_broadcast=True, unwrap_values=False))
                self.register_function_implementation(f'type::{t.typename}::less', CrowPythonFunctionRef(_CrowComparisonFunctionImpl(BOOL, lambda x, y: torch.lt(x, y)), support_batch=True, auto_broadcast=True, unwrap_values=False))
                self.register_function_implementation(f'type::{t.typename}::less_equal', CrowPythonFunctionRef(_CrowComparisonFunctionImpl(BOOL, lambda x, y: torch.le(x, y)), support_batch=True, auto_broadcast=True, unwrap_values=False))
                self.register_function_implementation(f'type::{t.typename}::greater', CrowPythonFunctionRef(_CrowComparisonFunctionImpl(BOOL, lambda x, y: torch.gt(x, y)), support_batch=True, auto_broadcast=True, unwrap_values=False))
                self.register_function_implementation(f'type::{t.typename}::greater_equal', CrowPythonFunctionRef(_CrowComparisonFunctionImpl(BOOL, lambda x, y: torch.ge(x, y)), support_batch=True, auto_broadcast=True, unwrap_values=False))
                self.register_function_implementation(
                    f'type::{t.typename}::hash',
                    CrowPythonFunctionRef(lambda x: x.tensor.item(), support_batch=False, unwrap_values=False)
                )
            elif isinstance(t, NamedTensorValueType) and isinstance(t.parent_type, VectorValueType):
                self.register_function_implementation(f'type::{t.typename}::add', CrowPythonFunctionRef(_CrowArithFunctionImpl(t, lambda x, y: x + y), support_batch=True, auto_broadcast=True, unwrap_values=False))
                self.register_function_implementation(f'type::{t.typename}::sub', CrowPythonFunctionRef(_CrowArithFunctionImpl(t, lambda x, y: x - y), support_batch=True, auto_broadcast=True, unwrap_values=False))
                self.register_function_implementation(f'type::{t.typename}::mul', CrowPythonFunctionRef(_CrowArithFunctionImpl(t, lambda x, y: x * y), support_batch=True, auto_broadcast=True, unwrap_values=False))
                self.register_function_implementation(f'type::{t.typename}::div', CrowPythonFunctionRef(_CrowArithFunctionImpl(t, lambda x, y: x / y), support_batch=True, auto_broadcast=True, unwrap_values=False))
                self.register_function_implementation(f'type::{t.typename}::neg', CrowPythonFunctionRef(_CrowUnaryFunctionImpl(t, lambda x: -x), support_batch=True, auto_broadcast=True, unwrap_values=False))
                self.register_function_implementation(f'type::{t.typename}::equal', CrowPythonFunctionRef(_CrowComparisonFunctionImpl(BOOL, lambda x, y: torch.eq(x, y).all(dim=-1)), support_batch=True, auto_broadcast=True, unwrap_values=False))
                self.register_function_implementation(f'type::{t.typename}::not_equal', CrowPythonFunctionRef(_CrowComparisonFunctionImpl(BOOL, lambda x, y: torch.ne(x, y).all(dim=-1)), support_batch=True, auto_broadcast=True, unwrap_values=False))
                self.register_function_implementation(f'type::{t.typename}::less', CrowPythonFunctionRef(_CrowComparisonFunctionImpl(BOOL, lambda x, y: torch.lt(x, y).all(dim=-1)), support_batch=True, auto_broadcast=True, unwrap_values=False))
                self.register_function_implementation(f'type::{t.typename}::less_equal', CrowPythonFunctionRef(_CrowComparisonFunctionImpl(BOOL, lambda x, y: torch.le(x, y).all(dim=-1)), support_batch=True, auto_broadcast=True, unwrap_values=False))
                self.register_function_implementation(f'type::{t.typename}::greater', CrowPythonFunctionRef(_CrowComparisonFunctionImpl(BOOL, lambda x, y: torch.gt(x, y).all(dim=-1)), support_batch=True, auto_broadcast=True, unwrap_values=False))
                self.register_function_implementation(f'type::{t.typename}::greater_equal', CrowPythonFunctionRef(_CrowComparisonFunctionImpl(BOOL, lambda x, y: torch.ge(x, y).all(dim=-1)), support_batch=True, auto_broadcast=True, unwrap_values=False))
            elif isinstance(t, PyObjValueType):
                if t.base_typename == 'string':
                    def compare(x, y):
                        return TensorValue(BOOL, x.batch_variables, torch.tensor(x.tensor.values == y.tensor.values, dtype=torch.bool), x.batch_dims)
                    self.register_function_implementation(f'type::{t.typename}::equal', CrowPythonFunctionRef(compare, support_batch=True, auto_broadcast=True, unwrap_values=False))
                    def compare_neg(x, y):
                        return TensorValue(BOOL, x.batch_variables, torch.tensor(x.tensor.values != y.tensor.values, dtype=torch.bool), x.batch_dims)
                    self.register_function_implementation(f'type::{t.typename}::not_equal', CrowPythonFunctionRef(compare_neg, support_batch=True, auto_broadcast=True, unwrap_values=False))

        for fname, cross_ref_name in self.domain.external_function_crossrefs.items():
            self.register_function_implementation(fname, CrowPythonFunctionCrossRef(cross_ref_name))

    def _register_external_function_implementations_from_domain(self):
        for filepath in self.domain.external_function_implementation_files:
            self.load_external_function_implementations_from_file(filepath)

    def load_external_function_implementations_from_file(self, filepath: str):
        from jacinle.utils.imp import load_module_filename
        module = load_module_filename(filepath)

        for name, func in module.__dict__.items():
            if isinstance(func, CrowPythonFunctionRef):
                self.register_function_implementation(name, func)

    _domain: CrowDomain

    @property
    def domain(self) -> CrowDomain:
        return self._domain

    @property
    def state(self) -> CrowState:
        """The state of the executor."""
        return self._state

    @property
    def effect_update_from_simulation(self) -> bool:
        """A context variable indicating whether the current effect should be updated from simulation, instead of the evaluation of expressions."""
        return self._effect_update_from_simulation

    @property
    def effect_update_from_execution(self) -> bool:
        """A context variable indicating whether the current effect should be updated from the execution of the operator."""
        return self._effect_update_from_execution

    @property
    def effect_action_index(self) -> Optional[int]:
        return self._effect_action_index

    def parse(self, string: Union[str, Expression], *, state: Optional[CrowState] = None, variables: Optional[Sequence[Variable]] = None) -> Expression:
        if isinstance(string, Expression):
            return string
        return self._domain.parse(string, state=state, variables=variables)

    _function_implementations: Dict[str, Union[CrowPythonFunctionRef, CrowPythonFunctionCrossRef]]

    @property
    def function_implementations(self) -> Dict[str, Union[CrowPythonFunctionRef, CrowPythonFunctionCrossRef]]:
        return self._function_implementations

    def register_function_implementation(self, name: str, func: Union[Callable, CrowPythonFunctionRef, CrowPythonFunctionCrossRef]):
        if isinstance(func, CrowPythonFunctionRef):
            self._function_implementations[name] = func.set_executor(self)
        elif isinstance(func, CrowPythonFunctionCrossRef):
            self._function_implementations[name] = func
        else:
            self._function_implementations[name] = CrowPythonFunctionRef(func)

    def get_function_implementation(self, name: str) -> CrowPythonFunctionRef:
        while name in self._function_implementations:
            func = self._function_implementations[name]
            if isinstance(func, CrowPythonFunctionCrossRef):
                name = func.cross_ref_name
            else:
                return func
        raise KeyError(f'Function {name} not found.')

    def execute(
        self, expression: Union[Expression, str],
        state: Optional[TensorState] = None,
        bounded_variables: Optional[BoundedVariablesDictCompatible] = None,
        csp: Optional[ConstraintSatisfactionProblem] = None,
        sgc: Optional[CrowSGC] = None,
        bypass_bounded_variable_check: bool = False,
        optimistic_execution: bool = False
    ) -> TensorValueExecutorReturnType:
        """Execute an expression.

        Args:
            expression: the expression to execute.
            state: the state to use. If None, the current state of the executor will be used.
            bounded_variables: the bounded variables to use. If None, the current bounded variables of the executor will be used.
            csp: the constraint satisfaction problem to use. If None, the current CSP of the executor will be used.
            sgc: the SGC (state-goal-constraints) context to use. If None, the current SGC context of the executor will be used.
            bypass_bounded_variable_check: whether to bypass the check of the bounded variables.
            optimistic_execution: whether to execute the expression optimistically (i.e., treat all CSP constraints True).

        Returns:
            the TensorValue object.
        """

        if isinstance(expression, str):
            all_variables = list()
            if bounded_variables is not None:
                for k, v in bounded_variables.items():
                    if isinstance(k, str):
                        all_variables.append(Variable(k, AutoType))
                    elif isinstance(k, Variable):
                        all_variables.append(k)
            expression = self.parse(expression, state=state, variables=all_variables)

        state = state if state is not None else self._state
        csp = csp if csp is not None else self._csp
        sgc = sgc if sgc is not None else self._sgc
        bounded_variables = bounded_variables if bounded_variables is not None else self._bounded_variables
        with self.with_state(state), self.with_csp(csp), self.with_sgc(sgc), self.with_bounded_variables(bounded_variables, bypass_bounded_variable_check=bypass_bounded_variable_check):
            self._optimistic_execution, backup = optimistic_execution, self._optimistic_execution
            try:
                return self._execute(expression)
            finally:
                self._optimistic_execution = backup

    def _execute(self, expression: Expression) -> TensorValueExecutorReturnType:
        if self.csp is not None:
            return self._csp_visitor.visit(expression)
        return self._default_visitor.visit(expression)

    @contextlib.contextmanager
    def update_effect_mode(self, evaluation_mode: CrowFunctionEvaluationMode, action_index: Optional[int] = None):
        old_from_simulation = self._effect_update_from_simulation
        old_from_execution = self._effect_update_from_execution
        old_action_index = self._effect_action_index
        self._effect_update_from_simulation = evaluation_mode is CrowFunctionEvaluationMode.SIMULATION
        self._effect_update_from_execution = evaluation_mode is CrowFunctionEvaluationMode.EXECUTION
        self._effect_action_index = action_index
        yield
        self._effect_update_from_simulation = old_from_simulation
        self._effect_update_from_execution = old_from_execution
        self._effect_action_index = old_action_index


def _fast_index(value, ind):
    if isinstance(value, TensorValue):
        return value.fast_index(ind)
    if len(ind) == 0:
        return value
    raise ValueError(f'Unsupported value type: {value}.')


class _CrowUnaryFunctionImpl(object):
    def __init__(self, dtype, op):
        self.dtype = dtype
        self.op = op

    def __call__(self, x):
        return TensorValue(self.dtype, x.batch_variables, self.op(x.tensor), x.batch_dims)


class _CrowArithFunctionImpl(object):
    def __init__(self, dtype, op):
        self.dtype = dtype
        self.op = op

    def __call__(self, x, y):
        return TensorValue(self.dtype, x.batch_variables, self.op(x.tensor, y.tensor), x.batch_dims)


class _CrowComparisonFunctionImpl(object):
    def __init__(self, dtype, op):
        self.dtype = dtype
        self.op = op

    def __call__(self, x, y):
        return TensorValue(BOOL, x.batch_variables, self.op(x.tensor, y.tensor), x.batch_dims)


class CrowExecutionDefaultVisitor(ExpressionVisitor):
    """The underlying default implementation for :class:`CrowExecutor`. This function does not handle CSPs (a.k.a. optimistic execution)."""

    def __init__(self, executor: CrowExecutor):
        """Initialize a PDExpressionExecutionDefaultVisitor.

        Args:
            executor: the executor that uses this visitor.
        """
        self.executor = executor

    @property
    def csp(self) -> ConstraintSatisfactionProblem:
        return self.executor.csp

    def visit_null_expression(self, expr: E.NullExpression) -> Any:
        return None

    def visit_variable_expression(self, expr: E.VariableExpression) -> TensorValueExecutorReturnType:
        variable = expr.variable
        if variable.dtype is AutoType:
            return self.executor.retrieve_bounded_variable_by_name(variable.name)
        return self.executor.bounded_variables[variable.dtype.typename][variable.name]

    def visit_object_constant_expression(self, expr: E.ObjectConstantExpression) -> Union[StateObjectReference, ListValue]:
        const = expr.constant
        if isinstance(const.name, (StateObjectReference, StateObjectList)):
            return const.name

        state = self.executor.state
        assert isinstance(state, NamedObjectTensorState)

        if isinstance(const, ListValue):
            return StateObjectList(const.dtype, [StateObjectReference(c.name, state.get_typed_index(c.name, c.dtype.typename), c.dtype) for c in const.values])

        return StateObjectReference(
            const.name,
            state.get_typed_index(const.name, const.dtype.typename),
            const.dtype
        )

    def visit_constant_expression(self, expr: E.ConstantExpression) -> TensorValueExecutorReturnType:
        value = expr.constant
        assert isinstance(value, (TensorValue, ListValue))
        return value

    def visit_list_creation_expression(self, expr: E.ListCreationExpression) -> Any:
        argument_values = self.forward_args(*expr.arguments, force_tuple=True)
        return ListValue(expr.return_type, argument_values)

    def visit_list_expansion_expression(self, expr: E.ListExpansionExpression) -> Any:
        raise RuntimeError('List expansion is not supported in the expression evaluation.')

    def visit_function_application_expression(
        self, expr: E.FunctionApplicationExpression,
        argument_values: Optional[Tuple[TensorValueExecutorReturnType, ...]] = None
    ) -> TensorValueExecutorReturnType:
        function = expr.function
        return_type = function.return_type
        state = self.executor.state
        assert isinstance(function, (CrowFeature, CrowFunction))

        if argument_values is None:
            argument_values = self.forward_args(*expr.arguments, force_tuple=True)

        has_list_values = any(isinstance(v, ListValue) for v in argument_values)

        if isinstance(function, CrowFunction) and function.is_generator_placeholder:  # always true branch
            if has_list_values:
                raise NotImplementedError('List values are not supported in the generator placeholder function.')
            argument_values = expand_argument_values(argument_values)

            batched_value = None
            for argv in argument_values:
                if isinstance(argv, TensorValue):
                    batched_value = argv
                    break
            assert batched_value is not None

            rv = torch.ones(
                batched_value.tensor.shape[:batched_value.total_batch_dims],
                dtype=torch.bool, device=batched_value.tensor.device if isinstance(batched_value.tensor, torch.Tensor) else None
            )
            assert return_type == BOOL

            # Add "true" asserts to the csp.
            if self.csp is not None and not self.executor.optimistic_execution:
                expr_string = expr.cached_string(-1)
                for ind in _iter_tensor_indices(rv):
                    self.csp.add_constraint(Constraint.from_function(
                        function,
                        # I think there is some bug here... for "StateObjectReference"
                        [argv.fast_index(tuple(ind)) if isinstance(argv, TensorValue) else argv for argv in argument_values],
                        True
                    ), note=f'{expr_string}::{ind}' if len(ind) > 0 else expr_string)

            return TensorValue(
                BOOL, batched_value.batch_variables,
                rv, batch_dims=state.batch_dims
            )
        elif function.is_cacheable and function.name in state.features:
            argument_values = [v.index if isinstance(v, StateObjectReference) else v for v in argument_values]

            batch_variables = list()
            anonymous_index = 0
            accessors = list()
            index_types = list()
            for i, (arg, value) in enumerate(zip(expr.arguments, argument_values)):
                if value == QINDEX:
                    batch_variables.append(arg.variable.name)
                    accessors.append(value)
                elif isinstance(value, StateObjectList):
                    batch_variables.append(f'@{anonymous_index}')
                    index_types.append(value.element_type)
                    anonymous_index += 1
                    accessors.append(value.array_accessor)
                else:
                    accessors.append(value)
            value = state.features[function.name][tuple(accessors)]

            if 'dirty_features' in state.internals and function.name in state.internals['dirty_features']:
                if has_list_values:
                    raise NotImplementedError('List values are not supported for dirty features.')
                value_opt = state.features[function.name].tensor_optimistic_values[tuple(argument_values)]
                if (value_opt < 0).any().item():
                    assert function.is_derived
                    with self.executor.with_bounded_variables({k: v for k, v in zip(function.arguments, argument_values)}):
                        return self._rename_derived_function_application(self.visit(function.derived_expression), function.arguments, expr.arguments, argument_values)

            if len(index_types) == 0:
                return value.rename_batch_variables(batch_variables, force=True)
            else:
                rv_dtype = BatchedListType(value.dtype, index_types)
                return value.rename_batch_variables(batch_variables, dtype=rv_dtype, force=True)
        elif function.is_derived:
            with self.executor.with_bounded_variables({k: v for k, v in zip(function.arguments, argument_values)}):
                return self._rename_derived_function_application(self.visit(function.derived_expression), function.arguments, expr.arguments, argument_values)
        else:
            # dynamic predicate is exactly the same thing as a pre-defined external function.
            # only supports external functions with a single return value.
            return self.forward_external_function(function.name, argument_values, return_type, expression=expr)

    def _rename_derived_function_application(self, rv: TensorValue, function_argument_variables, outer_arguments, argument_values):
        if not isinstance(rv, TensorValue):
            return rv

        output_batch_variables = list(rv.batch_variables)
        for function_argument_variable, outer_argument_expr, argument_value in zip(function_argument_variables, outer_arguments, argument_values):
            if argument_value is QINDEX:
                assert function_argument_variable.name in output_batch_variables, f'Variable {function_argument_variable.name} not found in the output batch variables {output_batch_variables}. Report this as a bug.'
                assert isinstance(outer_argument_expr, E.VariableExpression), 'Only variable arguments can be QINDEX. Report this as a bug.'
                index = output_batch_variables.index(function_argument_variable.name)
                output_batch_variables[index] = outer_argument_expr.variable.name

        return rv.clone(clone_tensor=False).rename_batch_variables(output_batch_variables, clone=False)

    def visit_list_function_application_expression(self, expr: E.ListFunctionApplicationExpression) -> Any:
        raise DeprecationWarning('List function application is deprecated.')
        argument_values = self.forward_args(*expr.arguments, force_tuple=True)
        return self.visit_function_application_expression(expr, argument_values)

        # if nr_values is None:
        #     return self.visit_function_application_expression(expr, argument_values)
        # else:
        #     rvs = list()
        #     for i in range(nr_values):
        #         this_argv = tuple(argv.values[i] if isinstance(argv, ListValue) else argv for argv in argument_values)
        #         rv = self.visit_function_application_expression(expr, this_argv)
        #         rvs.append(rv)
        #     return ListValue(expr.return_type, rvs)

    def visit_bool_expression(self, expr: E.BoolExpression, argument_values: Optional[Tuple[TensorValueExecutorReturnType, ...]] = None) -> TensorValueExecutorReturnType:
        if argument_values is None:
            argument_values = self.forward_args(*expr.arguments, force_tuple=True, expand_list_arguments=True)
            argument_values = expand_argument_values(argument_values)

        assert len(argument_values) > 0
        assert all(isinstance(v, TensorValue) for v in argument_values)

        dtype = argument_values[0].dtype
        batch_variables = argument_values[0].batch_variables

        if expr.bool_op is BoolOpType.NOT:
            assert len(argument_values) == 1
            return TensorValue(
                dtype, batch_variables,
                torch.logical_not(argument_values[0].tensor) if argument_values[0].tensor.dtype == torch.bool else 1 - argument_values[0].tensor,
                batch_dims=self.executor.state.batch_dims
            )
        elif expr.bool_op is BoolOpType.AND:
            if len(argument_values) == 1:
                return argument_values[0]
            return TensorValue(
                dtype, batch_variables,
                MaskedTensorStorage(torch.stack([argv.tensor for argv in argument_values], dim=-1).amin(dim=-1), None, argument_values[0].tensor_mask),
                batch_dims=self.executor.state.batch_dims
            )
        elif expr.bool_op is BoolOpType.OR:
            if len(argument_values) == 1:
                return argument_values[0]
            return TensorValue(
                dtype, batch_variables,
                MaskedTensorStorage(torch.stack([argv.tensor for argv in argument_values], dim=-1).amax(dim=-1), None, argument_values[0].tensor_mask),
                batch_dims=self.executor.state.batch_dims
            )
        elif expr.bool_op is BoolOpType.XOR:
            if len(argument_values) == 1:
                return argument_values[0]
            for argv in argument_values:
                if argv.tensor.requires_grad:
                    raise RuntimeError('XOR does not support gradients.')
            return TensorValue(
                dtype, batch_variables,
                MaskedTensorStorage(torch.stack([argv.tensor for argv in argument_values], dim=-1).sum(dim=-1) % 2, None, argument_values[0].tensor_mask),
                batch_dims=self.executor.state.batch_dims
            )
        elif expr.bool_op is BoolOpType.IMPLIES:
            assert len(argument_values) == 2
            return TensorValue(
                dtype, batch_variables,
                MaskedTensorStorage(torch.max(1 - argument_values[0].tensor, argument_values[1].tensor), None, argument_values[0].tensor_mask),
                batch_dims=self.executor.state.batch_dims
            )
        else:
            raise ValueError(f'Unknown bool op type: {expr.bool_op}.')

    def visit_quantification_expression(self, expr: E.QuantificationExpression, value: Optional[TensorValue] = None) -> TensorValueExecutorReturnType:
        if value is None:
            with self.executor.new_bounded_variables({expr.variable: QINDEX}):
                value = self.forward_args(expr.expression)
            assert isinstance(value, TensorValue)

        dtype = value.dtype
        batch_variables = value.batch_variables
        variable_index = batch_variables.index(expr.variable.name)
        batch_variables = batch_variables[:variable_index] + batch_variables[variable_index + 1:]

        if value.tensor_mask is None:
            tensor = value.tensor
            mask = None
        else:
            tensor, mask = value.tensor, value.tensor_mask
            if expr.quantification_op is QuantificationOpType.FORALL:
                tensor = (tensor * mask + (1 - mask)).to(tensor.dtype)
            elif expr.quantification_op is QuantificationOpType.EXISTS:
                tensor = (tensor * mask).to(tensor.dtype)
            else:
                raise ValueError(f'Unknown quantification op type: {expr.quantification_op}.')

        if expr.quantification_op is QuantificationOpType.FORALL:
            return TensorValue(
                dtype, batch_variables,
                MaskedTensorStorage(tensor.amin(dim=variable_index + value.batch_dims), None, mask),
                batch_dims=self.executor.state.batch_dims
            )
        elif expr.quantification_op is QuantificationOpType.EXISTS:
            return TensorValue(
                dtype, batch_variables,
                MaskedTensorStorage(tensor.amax(dim=variable_index + value.batch_dims), None, mask),
                batch_dims=self.executor.state.batch_dims
            )
        else:
            raise ValueError(f'Unknown quantifier type: {expr.quantification_op}.')

    def visit_object_compare_expression(self, expr: E.ObjectCompareExpression) -> Any:
        v1, v2 = self.forward_args(expr.arguments[0], expr.arguments[1])

        values = list()
        batched_variables = list()
        for i, v in enumerate([v1, v2]):
            if v is QINDEX:
                arg = expr.arguments[i]
                assert isinstance(arg, E.VariableExpression), 'Quantified object comparison only supports variable arguments.'
                batched_variables.append(arg.variable.name)
                values.append(torch.arange(0, len(self.executor.state.object_type2name[arg.variable.dtype.typename]), dtype=torch.int64))
            elif isinstance(v, StateObjectReference):
                values.append(torch.tensor(v.index, dtype=torch.int64))
            else:
                raise ValueError(f'Unsupported value type: {v}.')

        if v1 is QINDEX:
            if v2 is QINDEX:
                value = values[0].unsqueeze(1).eq(values[1].unsqueeze(0))
            else:
                value = values[0].eq(values[1])
        else:
            value = values[0].eq(values[1])

        if expr.compare_op is E.CompareOpType.EQ:
            pass
        elif expr.compare_op is E.CompareOpType.NEQ:
            value = torch.logical_not(value)
        else:
            raise ValueError(f'Unknown compare op type for object types: {expr.compare_op}.')

        return TensorValue(BOOL, batched_variables, value, batch_dims=self.executor.state.batch_dims)

    def visit_value_compare_expression(self, expr: E.ValueCompareExpression) -> Any:
        v1, v2 = self.forward_args(expr.arguments[0], expr.arguments[1])

        mapping = {
            E.CompareOpType.EQ: 'equal',
            E.CompareOpType.NEQ: 'not_equal',
            E.CompareOpType.LT: 'less',
            E.CompareOpType.LEQ: 'less_equal',
            E.CompareOpType.GT: 'greater',
            E.CompareOpType.GEQ: 'greater_equal',
        }
        target_op = mapping[expr.compare_op]
        if isinstance(v1.dtype, PyObjValueType):
            rv = self.forward_external_function(f'type::{v1.dtype.typename}::{target_op}', [v1, v2], BOOL, expression=expr)
        elif isinstance(v1.dtype, NamedTensorValueType) or isinstance(v1.dtype, ScalarValueType):
            rv = self.forward_external_function(f'type::{v1.dtype.typename}::{target_op}', [v1, v2], BOOL, expression=expr)
        else:
            raise NotImplementedError(f'Unsupported FeatureEqual computation for dtype {v1.dtype} and {v2.dtype}.')
        return rv

    def visit_condition_expression(self, expr: E.ConditionExpression) -> Any:
        raise NotImplementedError('Condition expression is not supported in the expression evaluation.')

    def visit_find_one_expression(self, expr: E.FindOneExpression) -> Any:
        with self.executor.new_bounded_variables({expr.variable: QINDEX}):
            values = self.visit(expr.expression)

        assert values.batch_dims == 0
        assert len(values.batch_variables) == 1

        x = (values.tensor > 0.5)
        objects = x.nonzero().squeeze(-1).detach().cpu().tolist()
        if len(objects) == 0:
            raise RuntimeError('No object found. Currently the executor does not support this case.')
        names = self.executor.state.object_type2name[expr.variable.dtype.typename]
        return StateObjectReference(names[objects[0]], objects[0], expr.variable.dtype)

    def visit_find_all_expression(self, expr: E.FindAllExpression) -> Any:
        with self.executor.new_bounded_variables({expr.variable: QINDEX}):
            values = self.visit(expr.expression)

        assert values.batch_dims == 0
        assert len(values.batch_variables) == 1

        x = (values.tensor > 0.5)
        objects = x.nonzero().squeeze(-1).detach().cpu().tolist()
        names = self.executor.state.object_type2name[expr.variable.dtype.typename]
        return StateObjectList(expr.return_type, [StateObjectReference(names[i], i, expr.variable.dtype) for i in objects])

    def visit_predicate_equal_expression(self, expr: E.PredicateEqualExpression, feature: Optional[TensorValue] = None, value: Optional[TensorValue] = None) -> TensorValueExecutorReturnType:
        if feature is None or value is None:
            feature, value = self.forward_args(expr.predicate, expr.value)
            feature, value = expand_argument_values([feature, value])

        if isinstance(feature.dtype, PyObjValueType):
            rv = self.forward_external_function(f'type::{feature.dtype.typename}::equal', [feature, value], BOOL, expression=expr)
        elif isinstance(feature.dtype, NamedTensorValueType):
            rv = self.forward_external_function(f'type::{feature.dtype.typename}::equal', [feature, value], BOOL, expression=expr)
        else:
            raise NotImplementedError(f'Unsupported FeatureEqual computation for dtype {feature.dtype} and {value.dtype}.')

        return rv

    def visit_assign_expression(self, expr: E.AssignExpression):
        # TODO(Jiayuan Mao @ 2024/01/22): is this really the right thing to do?
        if self.executor.effect_update_from_simulation or self.executor.effect_update_from_execution:
            return

        state: CrowState = self.executor.state
        argument_values = list(self.forward_args(*expr.predicate.arguments, force_tuple=True))
        target_value = self.forward_args(expr.value)

        for i, (arg, value) in enumerate(zip(expr.predicate.arguments, argument_values)):
            if value == QINDEX:
                assert isinstance(arg, E.VariableExpression), 'Quantified object comparison only supports variable arguments.'
                # TODO(Jiayuan Mao @ 2024/08/3): I think we need to think about how to align the batch variable dimensions...
                pass
            elif isinstance(value, StateObjectList):
                argument_values[i] = value.array_accessor
            elif isinstance(value, StateObjectReference):
                argument_values[i] = value.index

        # if state.features[expr.predicate.function.name].quantized:
        #     if not value.quantized:
        #         value = self.executor.value_quantizer.quantize_value(value)
        # else:
        #     if value.quantized:
        #         value = self.executor.value_quantizer.unquantize_value(value)

        function_name = expr.predicate.function.name
        if function_name not in state.features:
            raise NotImplementedError('Assignments to dirty features are not supported.')
        else:
            if isinstance(target_value, ListValue):
                if len(target_value.values) == 0:
                    return
                target_value = TensorValue(target_value.values[0].dtype, 1 + len(target_value.values[0].batch_variables), torch.stack([x.tensor for x in target_value.values]))
            state.features[expr.predicate.function.name][tuple(argument_values)] = target_value

    def visit_conditional_select_expression(self, expr: E.ConditionalSelectExpression) -> TensorValueExecutorReturnType:
        value, condition = self.forward_args(expr.predicate, expr.condition)
        value = value.clone()
        if value.tensor_mask is None:
            value.tensor_mask = condition.tensor
        else:
            value.tensor_mask = torch.min(value.tensor_mask, condition.tensor)
        return value

    def visit_deictic_select_expression(self, expr: E.DeicticSelectExpression) -> TensorValueExecutorReturnType:
        with self.executor.new_bounded_variables({expr.variable: QINDEX}):
            return self.visit(expr.expression)

    CONDITIONAL_ASSIGN_QUANTIZE = False

    def visit_conditional_assign_expression(self, expr: E.ConditionalAssignExpression):
        state = self.executor.state
        argument_values = self.forward_args(*expr.predicate.arguments, force_tuple=True)
        argument_values = [v.index if isinstance(v, StateObjectReference) else v for v in argument_values]
        value = self.forward_args(expr.value)
        condition = self.forward_args(expr.condition)

        condition_tensor = jactorch.quantize(condition.tensor) if self.CONDITIONAL_ASSIGN_QUANTIZE else condition.tensor

        feature = state.features[expr.predicate.function.name]
        origin_tensor = feature[argument_values].tensor  # I am not using feature.tensor[argument_values] because the current code will handle TensorizedPyObjValues too.
        # assert value.tensor.dim() == condition_tensor.dim() or value.tensor.dim() == 0

        if value.is_tensorized_pyobj:
            raise NotImplementedError('Cannot make conditional assignments for tensorized pyobj.')
        else:
            if condition_tensor.dim() < value.tensor.dim():
                condition_tensor = condition_tensor.unsqueeze(-1)
            state.features[expr.predicate.function.name].tensor[argument_values] = (
                condition_tensor.to(origin_tensor.dtype) * value.tensor + (1 - condition_tensor).to(origin_tensor.dtype) * origin_tensor
            )

    def visit_deictic_assign_expression(self, expr: E.DeicticAssignExpression):
        with self.executor.new_bounded_variables({expr.variable: QINDEX}):
            self.visit(expr.expression)

    def forward_args(self, *args, force_tuple: bool = False, expand_list_arguments: bool = False) -> Union[TensorValueExecutorReturnType, Tuple[TensorValueExecutorReturnType, ...]]:
        if len(args) == 1 and not force_tuple:
            rvs = self.visit(args[0])
        else:
            rvs = tuple(self.visit(arg) for arg in args)

        if expand_list_arguments:
            expanded_rvs = list()
            for rv in rvs:
                if isinstance(rv, ListValue):
                    expanded_rvs.extend(rv.values)
                else:
                    expanded_rvs.append(rv)
            if not force_tuple and len(expanded_rvs) == 1:
                return expanded_rvs[0]
            else:
                return tuple(expanded_rvs)
        else:
            return rvs

    def forward_external_function(
        self, function_name: str, argument_values: Sequence[TensorValueExecutorReturnType],
        return_type: Union[TensorValueTypeBase, PyObjValueType], auto_broadcast: bool = True, expression: Optional[Expression] = None
    ) -> TensorValue:
        external_function = self.executor.get_function_implementation(function_name)
        assert isinstance(external_function, CrowPythonFunctionRef)
        function_def = expression.function if isinstance(expression, E.FunctionApplicationExpression) else None
        return external_function.forward(argument_values, return_type=return_type, auto_broadcast=auto_broadcast, function_def=function_def)


class CrowExecutionCSPVisitor(CrowExecutionDefaultVisitor):
    def __init__(self, executor: CrowExecutor):
        super().__init__(executor)

    def forward_external_function(
        self, function_name: str, argument_values: Sequence[TensorValueExecutorReturnType],
        return_type: Union[TensorValueTypeBase, PyObjValueType], auto_broadcast: bool = True, expression: Optional[E.FunctionApplicationExpression] = None
    ) -> TensorValue:
        argument_values = expand_argument_values(argument_values)
        optimistic_masks = [is_optimistic_value(argv.tensor_optimistic_values) for argv in argument_values if isinstance(argv, TensorValue) and argv.tensor_optimistic_values is not None]
        if len(optimistic_masks) > 0:
            optimistic_mask = torch.stack(optimistic_masks, dim=-1).any(dim=-1)

            rv = super().forward_external_function(function_name, argument_values, return_type=return_type, auto_broadcast=auto_broadcast, expression=expression)
            if optimistic_mask.sum().item() == 0:
                return rv

            rv.init_tensor_optimistic_values()
            if self.executor.optimistic_execution:
                rv.tensor_optimistic_values[optimistic_mask.nonzero(as_tuple=True)] = OPTIM_MAGIC_NUMBER_MAGIC
            else:
                expr_string = expression.cached_string(-1)
                if isinstance(expression, E.ValueCompareExpression):
                    if expression.compare_op is E.CompareOpType.EQ:
                        constraint_function = Constraint.EQUAL
                    else:
                        raise NotImplementedError(f'Unsupported compare op type: {expression.compare_op} for optimistic execution.')
                elif isinstance(expression, E.PredicateEqualExpression):
                    constraint_function = Constraint.EQUAL
                elif isinstance(expression, E.FunctionApplicationExpression):
                    constraint_function = expression.function
                else:
                    raise NotImplementedError(f'Unsupported expression type: {expression} for optimistic execution.')
                for ind in optimistic_mask.nonzero().tolist():
                    ind = tuple(ind)
                    new_identifier = self.executor.csp.new_var(return_type, wrap=True)
                    rv.tensor_optimistic_values[ind] = new_identifier.identifier
                    self.csp.add_constraint(Constraint.from_function(
                        constraint_function,
                        [_fast_index(argv, ind) for argv in argument_values],
                        new_identifier
                    ), note=f'{expr_string}::{ind}' if len(ind) > 0 else expr_string)
            return rv

        return super().forward_external_function(function_name, argument_values, return_type=return_type, auto_broadcast=auto_broadcast, expression=expression)

    def visit_function_application_expression(self, expr: E.FunctionApplicationExpression, argument_values: Optional[Tuple[TensorValueExecutorReturnType, ...]] = None) -> TensorValueExecutorReturnType:
        return super().visit_function_application_expression(expr, argument_values)

    def visit_bool_expression(self, expr: E.BoolExpression, argument_values: Optional[Tuple[TensorValueExecutorReturnType, ...]] = None) -> TensorValueExecutorReturnType:
        if argument_values is None:
            argument_values = self.forward_args(*expr.arguments, force_tuple=True, expand_list_arguments=True)
            argument_values = list(expand_argument_values(argument_values))

        for argv in argument_values:
            assert argv.dtype == BOOL, 'Boolean expression only supports boolean values in CSP mode.'

        optimistic_masks = [is_optimistic_value(argv.tensor_optimistic_values) for argv in argument_values if isinstance(argv, TensorValue) and argv.tensor_optimistic_values is not None]
        if len(optimistic_masks) > 0:
            optimistic_mask = torch.stack(optimistic_masks, dim=-1).any(dim=-1)
            if optimistic_mask.sum().item() > 0:
                rv = super().visit_bool_expression(expr, argument_values)
                rv.init_tensor_optimistic_values()

                if self.executor.optimistic_execution:
                    rv.tensor_optimistic_values[optimistic_mask.nonzero(as_tuple=True)] = OPTIM_MAGIC_NUMBER_MAGIC
                else:
                    expr_string = expr.cached_string(-1)
                    for ind in optimistic_mask.nonzero().tolist():
                        ind = tuple(ind)

                        this_argv = [argv.fast_index(ind, wrap=False) for argv in argument_values]
                        determined = None
                        if expr.return_type == BOOL:
                            if expr.bool_op is BoolOpType.NOT:
                                pass  # nothing we can do.
                            elif expr.bool_op is BoolOpType.AND:
                                if 0 in this_argv or False in this_argv:
                                    determined = False
                            elif expr.bool_op is BoolOpType.OR:
                                if 1 in this_argv or True in this_argv:
                                    determined = True
                            this_argv = [v for v in this_argv if isinstance(v, OptimisticValue)]
                        else:  # generalized boolean operations.
                            pass

                        if determined is None:
                            new_identifier = self.csp.new_var(BOOL)
                            rv.tensor_optimistic_values[ind] = new_identifier
                            self.csp.add_constraint(Constraint(
                                expr.bool_op,
                                this_argv,
                                cvt_opt_value(new_identifier, BOOL),
                            ), note=f'{expr_string}::{ind}' if len(ind) > 0 else expr_string)
                        else:
                            rv[ind] = determined
                return rv
            else:
                return super().visit_bool_expression(expr, argument_values)
        else:  # if len(optimistic_masks) == 0
            return super().visit_bool_expression(expr, argument_values)

    def visit_quantification_expression(self, expr: E.QuantificationExpression, value: Optional[TensorValue] = None) -> Any:
        if value is None:
            with self.executor.new_bounded_variables({expr.variable: QINDEX}):
                value = self.forward_args(expr.expression)
            assert isinstance(value, TensorValue)

        assert value.dtype == BOOL, 'Quantification expression only supports boolean values in CSP mode.'

        value.init_tensor_optimistic_values()
        rv = super().visit_quantification_expression(expr, value)

        dim = value.batch_variables.index(expr.variable.name) + value.batch_dims
        value_transposed = value.tensor
        optimistic_values_transposed = value.tensor_optimistic_values
        if dim != value.tensor.ndim - 1:
            value_transposed = value_transposed.transpose(dim, -1)  # put the target dimension last.
            optimistic_values_transposed = optimistic_values_transposed.transpose(dim, -1)
        optimistic_mask_transposed = is_optimistic_value(optimistic_values_transposed)

        value_transposed = torch.where(
            optimistic_mask_transposed,
            optimistic_values_transposed,
            value_transposed.to(optimistic_values_transposed.dtype)
        )
        optimistic_mask = optimistic_mask_transposed.any(dim=-1)

        if optimistic_mask.sum().item() == 0:
            return rv

        rv.init_tensor_optimistic_values()
        if self.executor.optimistic_execution:
            rv.tensor_optimistic_values[optimistic_mask.nonzero(as_tuple=True)] = OPTIM_MAGIC_NUMBER_MAGIC
        else:
            expr_string = expr.cached_string(-1)
            for ind in optimistic_mask.nonzero().tolist():
                ind = tuple(ind)

                this_argv = value_transposed[ind].tolist()
                determined = None
                if expr.quantification_op is QuantificationOpType.FORALL:
                    if 0 in this_argv or False in this_argv:
                        determined = False
                else:
                    if 1 in this_argv or True in this_argv:
                        determined = True
                this_argv = list(filter(is_optimistic_value, this_argv))

                if determined is None:
                    new_identifier = self.csp.new_var(BOOL)
                    rv.tensor_optimistic_values[ind] = new_identifier
                    self.csp.add_constraint(Constraint(
                        expr.quantification_op,
                        [OptimisticValue(value.dtype, int(v)) for v in this_argv],
                        OptimisticValue(value.dtype, new_identifier),
                    ), note=f'{expr_string}::{ind}' if len(ind) > 0 else expr_string)
                else:
                    rv.tensor[ind] = determined
        return rv

    def visit_predicate_equal_expression(self, expr: E.PredicateEqualExpression, feature: Optional[TensorValue] = None, value: Optional[TensorValue] = None) -> Any:
        if feature is None or value is None:
            feature, value = self.forward_args(expr.predicate, expr.value)
            feature, value = expand_argument_values([feature, value])

        feature.init_tensor_optimistic_values()
        value.init_tensor_optimistic_values()
        optimistic_mask = torch.logical_or(is_optimistic_value(feature.tensor_optimistic_values), is_optimistic_value(value.tensor_optimistic_values))

        if optimistic_mask.sum().item() > 0:
            raise NotImplementedError('Optimistic execution is not supported for predicate equal expression.')

        rv = super().visit_predicate_equal_expression(expr, feature, value)
        return rv

        # feature.init_tensor_optimistic_values()
        # value.init_tensor_optimistic_values()
        # optimistic_mask = torch.logical_or(is_optimistic_value(feature.tensor_optimistic_values), is_optimistic_value(value.tensor_optimistic_values))

        # if optimistic_mask.sum().item() == 0:
        #     return rv

        # rv.init_tensor_optimistic_values()
        # if self.executor.optimistic_execution:
        #     rv.tensor_optimistic_values[optimistic_mask.nonzero(as_tuple=True)] = OPTIM_MAGIC_NUMBER_MAGIC
        # else:
        #     expr_string = expr.cached_string(-1)
        #     for ind in optimistic_mask.nonzero().tolist():
        #         ind = tuple(ind)
        #         this_argv = feature.fast_index(ind), value.fast_index(ind)
        #         new_identifier = self.csp.new_var(BOOL)
        #         rv.tensor_optimistic_values[ind] = new_identifier
        #         self.csp.add_constraint(EqualityConstraint(
        #             *[cvt_opt_value(v, feature.dtype) for v in this_argv],
        #             OptimisticValue(BOOL, new_identifier)
        #         ), note=f'{expr_string}::{ind}' if len(ind) > 0 else expr_string)
        # return rv

    def visit_assign_expression(self, expr: E.AssignExpression) -> Any:
        if self.executor.effect_update_from_simulation or self.executor.effect_update_from_execution:
            feature = self.executor.state.features[expr.predicate.function.name]
            feature.init_tensor_optimistic_values()
            argument_values = self.forward_args(*expr.predicate.arguments, force_tuple=True)

            assert self.executor.effect_action_index is not None, 'Effect action index must be set if the target predicate will be updated from simulation.'
            expr_string = expr.cached_string(-1)
            for entry_values in _expand_tensor_indices(feature, argument_values):
                if self.executor.optimistic_execution:
                    raise RuntimeError('Optimistic execution is not supported for effect update from simulation.')
                else:
                    opt_identifier = self.csp.new_var(feature.dtype, wrap=True)
                    feature.tensor_optimistic_values[entry_values] = opt_identifier.identifier
                    self.csp.add_constraint(Constraint(
                        SimulationFluentConstraintFunction(self.executor.effect_action_index, expr.predicate.function, entry_values, is_execution_constraint=self.executor.effect_update_from_execution),
                        [],
                        opt_identifier,
                        note=f'{expr_string}::{entry_values}' if len(entry_values) > 0 else expr_string
                    ))
        else:
            return super().visit_assign_expression(expr)

    def visit_conditional_select_expression(self, expr: E.ConditionalSelectExpression) -> TensorValueExecutorReturnType:
        return super().visit_conditional_select_expression(expr)

    def visit_deictic_select_expression(self, expr: E.DeicticSelectExpression) -> Any:
        return super().visit_deictic_select_expression(expr)

    def visit_conditional_assign_expression(self, expr: E.ConditionalAssignExpression) -> Any:
        if self.executor.effect_update_from_simulation or self.executor.effect_update_from_execution:
            raise NotImplementedError('Conditional assign is not supported in simulation mode.')
        if self.executor.optimistic_execution:
            raise RuntimeError('Optimistic execution is not supported for conditional assign.')

        state = self.executor.state
        argument_values = self.forward_args(*expr.predicate.arguments, force_tuple=True)
        value = self.forward_args(expr.value)
        condition = self.forward_args(expr.condition)

        condition_tensor = jactorch.quantize(condition.tensor) if self.CONDITIONAL_ASSIGN_QUANTIZE else condition.tensor
        condition_tensor = (condition_tensor > 0.5).to(torch.bool)

        feature = state.features[expr.predicate.function.name]
        origin_tensor = feature[argument_values].tensor  # I am not using feature.tensor[argument_values] because the current code will handle TensorizedPyObjValues too.
        # assert value.tensor.dim() == condition_tensor.dim() or value.tensor.dim() == 0

        # NB(Jiayuan Mao @ 2023/08/15): conditional assignment does not support "soft" assignment.
        if value.is_tensorized_pyobj:
            raise NotImplementedError('Cannot make conditional assignments for tensorized pyobj.')
        else:
            if condition_tensor.dim() < value.tensor.dim():
                condition_tensor_expanded = condition_tensor.unsqueeze(-1)
            else:
                condition_tensor_expanded = condition_tensor

            feature.tensor[argument_values] = (
                condition_tensor_expanded.to(origin_tensor.dtype) * value.tensor + (1 - condition_tensor_expanded).to(origin_tensor.dtype) * origin_tensor
            )
            feature.init_tensor_optimistic_values()
            if value.tensor_optimistic_values is not None:
                feature.tensor_optimistic_values[argument_values] = (
                    condition_tensor.to(torch.int64) * value.tensor_optimistic_values +
                    (1 - condition_tensor).to(torch.int64) * state.features[expr.predicate.function.name].tensor_optimistic_values[argument_values]
                )

            if condition.tensor_optimistic_values is None:
                pass
            else:
                optimistic_mask = is_optimistic_value(condition.tensor_optimistic_values)
                if optimistic_mask.sum().item() == 0:
                    pass
                else:
                    expr_string = expr.cached_string(-1)
                    dtype = expr.predicate.function.return_type
                    for ind in optimistic_mask.nonzero().tolist():
                        ind = tuple(ind)

                        new_identifier = self.csp.new_var(dtype, wrap=True)
                        neg_condition_identifier = self.csp.new_var(BOOL, wrap=True)
                        eq_1_identifier = self.csp.new_var(BOOL, wrap=True)
                        eq_2_identifier = self.csp.new_var(BOOL, wrap=True)
                        condition_identifier = condition.tensor_optimistic_values[ind].item()

                        self.csp.add_constraint(EqualityConstraint(
                            new_identifier,
                            cvt_opt_value(value.fast_index(ind), dtype),
                            eq_1_identifier,
                        ), note=f'{expr_string}::{ind}::eq-1')
                        self.csp.add_constraint(EqualityConstraint(
                            new_identifier,
                            cvt_opt_value(origin_tensor.fast_index(ind) if isinstance(origin_tensor, TensorizedPyObjValues) else origin_tensor[ind], dtype),
                            eq_2_identifier
                        ), note=f'{expr_string}::{ind}::eq-2')

                        self.csp.add_constraint(Constraint(
                            BoolOpType.NOT,
                            [OptimisticValue(BOOL, condition_identifier)],
                            neg_condition_identifier
                        ), note=f'{expr_string}::{ind}::neg-cond')

                        self.csp.add_constraint(Constraint(
                            BoolOpType.OR,
                            [neg_condition_identifier, eq_1_identifier],
                            cvt_opt_value(True, BOOL)
                        ), note=f'{expr_string}::{ind}::implies-new')

                        self.csp.add_constraint(Constraint(
                            BoolOpType.OR,
                            [OptimisticValue(BOOL, condition_identifier), eq_2_identifier],
                            cvt_opt_value(True, BOOL)
                        ), note=f'{expr_string}::{ind}::implies-old')

                        feature.tensor_optimistic_values[ind] = new_identifier.identifier

    def visit_deictic_assign_expression(self, expr: E.DeicticAssignExpression) -> Any:
        return super().visit_deictic_assign_expression(expr)


def _iter_tensor_indices(target_tensor: torch.Tensor) -> Iterator[Tuple[int, ...]]:
    """Iterate from the indices of a tensor. """
    for ind in torch.nonzero(torch.ones_like(target_tensor)):
        yield tuple(ind.tolist())


def _expand_tensor_indices(target_value: TensorValue, input_indices: Tuple[Union[int, slice, StateObjectReference], ...]) -> Iterator[Tuple[int, ...]]:
    """Iterate over entry indices based on the input indices. Supported indices are int and QINDEX (:).

    Args:
        target_value: the target value, used to determine the size of ``QINDEX``.
        input_indices: the indices to iterate over.

    Yields:
        the entry indices.
    """
    indices = list()
    for i, ind in enumerate(input_indices):
        if isinstance(ind, int):
            indices.append(torch.tensor([ind], dtype=torch.int64))
        elif isinstance(ind, slice):
            assert ind.step is None and ind.start is None and ind.stop is None  # == ':'
            indices.append(torch.arange(target_value.tensor.shape[i], dtype=torch.int64))
        elif isinstance(ind, StateObjectReference):
            indices.append(torch.tensor([ind.index], dtype=torch.int64))
        else:
            raise ValueError(f'Invalid index type: {ind}')

    if len(indices) == 0:
        yield tuple()
        return

    if len(indices) == 1:
        for x in indices[0]:
            yield tuple([x.item()], )
        return

    indices = torch.meshgrid(*indices, indexing='ij')
    indices = [i.flatten() for i in indices]
    for i in range(len(indices[0])):
        yield tuple(indices[j][i].item() for j in range(len(indices)))
