#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : executor.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 11/18/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""This file contains the executor classes for the PDSketch framework.

All the executors are based on :class:`~concepts.dsl.tensor_value.TensorValue` classes.
It supports all expressions defined in :mod:`~concepts.dsl.expressions`, including the basic
function application expressions and a few first-order logic expression types. The executors
are designed to be "differentiable", which means we can directly do backpropagation on the
computed values.

The main entry for the executor is the :class:`PDSketchExecutor` class.
Internally it contains two executor implementations: the basic one, and an "optimistic" one,
which handles the case where the value of a variable can be unknown and "optimistic".
"""

import itertools
import functools
from typing import Any, Optional, Union, Iterator, Sequence, Tuple, List, Mapping, Dict, Callable
from collections import defaultdict
from tabulate import tabulate

import torch
import jactorch
from jacinle.logging import get_logger

import concepts.dsl.expression as E
from concepts.dsl.dsl_types import BOOL, TensorValueTypeBase, ScalarValueType, VectorValueType, NamedTensorValueType, PyObjValueType, QINDEX, Variable
from concepts.dsl.expression import Expression, BoolOpType, QuantificationOpType
from concepts.dsl.expression_visitor import ExpressionVisitor
from concepts.dsl.tensor_value import TensorizedPyObjValues, TensorValue, MaskedTensorStorage
from concepts.dsl.tensor_state import StateObjectReference, TensorState, NamedObjectTensorState, ObjectNameArgument, ObjectTypeArgument
from concepts.dsl.constraint import OPTIM_MAGIC_NUMBER_MAGIC, is_optimistic_value, OptimisticValue, cvt_opt_value, Constraint, EqualityConstraint, ConstraintSatisfactionProblem, AssignmentDict, SimulationFluentConstraintFunction
from concepts.dsl.parsers.parser_base import ParserBase
from concepts.dsl.executors.tensor_value_executor import TensorValueExecutorBase, TensorValueExecutorReturnType, compose_bvdict_args
from concepts.pdsketch.predicate import Predicate
from concepts.pdsketch.operator import Operator, MacroOperator, OperatorApplier
from concepts.pdsketch.generator import Generator
from concepts.pdsketch.domain import Domain, State

logger = get_logger(__file__)

__all__ = [
    'PythonFunctionRef', 'PythonFunctionCrossRef', 'config_function_implementation',
    'PDSketchExecutor', 'PDSketchExecutionDefaultVisitor', 'PDSketchExecutionCSPVisitor',
    'StateDefinitionHelper',
    'GeneratorManager', 'wrap_singletime_function_to_iterator'
]


class PythonFunctionRef(object):
    """A reference to a Python function.

    This class is used to wrap external function implementations in domains.
    """

    def __init__(
        self, function: Callable, function_quantized: Optional[Callable] = None,
        *,
        return_type: Optional[Union[TensorValueTypeBase, PyObjValueType, Tuple[Union[TensorValueTypeBase, PyObjValueType], ...]]] = None,
        support_batch: bool = False, auto_broadcast: bool = False,
        use_object_names: bool = True, unwrap_values: Optional[bool] = None,
        include_executor_args: bool = False, is_iterator: bool = False,
        executor: Optional['PDSketchExecutor'] = None
    ):
        """Initialize a Python function reference.

        Args:
            function: the function to be wrapped.
            function_quantized: the quantized version of the function (can be None).
            support_batch: whether the function supports batched inputs.
            auto_broadcast: whether the executor should automatically broadcast the arguments before calling the function.
            use_object_names: whether the executor should use the object names in the state (instead of the index).
            unwrap_values: whether the executor should unwrap the tensor values before calling the function.
            include_executor_args: whether the caller should include the executor as the first argument.
            is_iterator: whether the function is an iterator.
            executor: the executor that is using this function reference.
        """

        self.function = function
        self.function_quantized = function_quantized
        self.return_type = return_type
        self.support_batch = support_batch
        self.auto_broadcast = auto_broadcast
        self.use_object_names = use_object_names
        if unwrap_values is None:
            unwrap_values = not support_batch or auto_broadcast
        self.unwrap_values = unwrap_values
        self.include_executor_args = include_executor_args
        self.is_iterator = is_iterator
        self._executor = executor

    function: Callable
    """The internal implementation of the function."""

    function_quantized: Optional[Callable]
    """The quantized version of the function (can be None)."""

    return_type: Optional[Union[TensorValueTypeBase, PyObjValueType, Tuple[Union[TensorValueTypeBase, PyObjValueType], ...]]]
    """The return type of the function."""

    support_batch: bool
    """Whether the function supports batched inputs."""

    auto_broadcast: bool
    """Whether the executor should automatically broadcast the arguments before calling the function."""

    use_object_names: bool
    """Whether the executor should use the object names in the state (instead of the index)."""

    unwrap_values: bool
    """Whether the executor should unwrap the tensor values before calling the function."""

    include_executor_args: bool
    """Whether the caller should include the executor as the first argument."""

    is_iterator = False
    """Whether the function is an iterator."""

    def set_executor(self, executor: 'PDSketchExecutor') -> 'PythonFunctionRef':
        """Set the executor that is using this function reference.

        Args:
            executor: the executor that is using this function reference.

        Returns:
            the function reference itself.
        """
        self._executor = executor
        return self

    def __str__(self) -> str:
        return (
            'PythonFunctionRef('
            f'{self.function}, support_batch={self.support_batch}, auto_broadcast={self.auto_broadcast}, '
            f'use_object_names={self.use_object_names}, unwrap_values={self.unwrap_values}, include_executor_args={self.include_executor_args}, '
            f'is_iterator={self.is_iterator})'
        )

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def flags(self) -> Dict[str, bool]:
        return {
            'support_batch': self.support_batch,
            'auto_broadcast': self.auto_broadcast,
            'use_object_names': self.use_object_names,
            'unwrap_values': self.unwrap_values,
            'include_executor_args': self.include_executor_args,
            'is_iterator': self.is_iterator,
        }

    def forward(
        self, argument_values: Sequence[TensorValueExecutorReturnType], return_type: Optional[Union[TensorValueTypeBase, PyObjValueType]] = None,
        auto_broadcast: bool = True, wrap_rv: bool = True
    ) -> Union[TensorValue, Tuple[TensorValue, ...]]:
        """Call the function.

        Args:
            argument_values: the arguments to the function.
            return_type: the type of the return value.
            auto_broadcast: whether the executor should automatically broadcast the arguments before calling the function.
            wrap_rv: whether the executor should wrap the return value.

        Returns:
            the result of the function.
        """
        function = self.function

        if self.use_object_names:
            argument_values = [v.name if isinstance(v, StateObjectReference) else v for v in argument_values]
        else:
            argument_values = [v.index if isinstance(v, StateObjectReference) else v for v in argument_values]

        if self.support_batch:
            if self.auto_broadcast and auto_broadcast:
                argument_values = expand_argument_values(argument_values)

        argument_values_flat = argument_values
        if self.unwrap_values:
            argument_values_flat = [v.tensor if isinstance(v, TensorValue) else v for v in argument_values_flat]
            if not self.support_batch:
                argument_values_flat = [v.item() if isinstance(v, TensorizedPyObjValues) else v for v in argument_values_flat]

        if self.include_executor_args:
            assert self._executor is not None, 'Executor is None.'
            rv = function(self._executor, *argument_values_flat)
        else:
            rv = function(*argument_values_flat)

        if not wrap_rv:
            return rv
        return self._wrap_rv(rv, return_type, argument_values, auto_broadcast)

    def forward_generator(
        self, argument_values: Sequence[TensorValueExecutorReturnType], return_type: Optional[Union[TensorValueTypeBase, PyObjValueType]] = None,
        auto_broadcast: bool = True, wrap_rv: bool = True
    ) -> Union[Iterator[TensorValue], Iterator[Tuple[TensorValue, ...]]]:
        """Call the function and return a generator.

        Args:
            argument_values: the arguments to the function.
            return_type: the type of the return value.
            auto_broadcast: whether the executor should automatically broadcast the arguments before calling the function.
            wrap_rv: whether the executor should wrap the
        """

        generator = self.forward(argument_values, return_type=return_type, auto_broadcast=auto_broadcast, wrap_rv=False)
        if not wrap_rv:
            yield from generator
        else:
            for v in generator:
                yield self._wrap_rv(v, return_type, argument_values, auto_broadcast)

    def __call__(self, *args, return_type: Optional[Union[TensorValueTypeBase, PyObjValueType]] = None, auto_broadcast: bool = True, wrap_rv: bool = True) -> Union[TensorValue, Tuple[TensorValue, ...]]:
        assert not self.is_iterator, 'Use iter_from to call an iterator function.'
        return self.forward(args, return_type=return_type, auto_broadcast=auto_broadcast, wrap_rv=wrap_rv)

    def iter_from(self, *args, return_type: Optional[Union[TensorValueTypeBase, PyObjValueType]] = None, auto_broadcast: bool = True, wrap_rv: bool = True) -> Union[Iterator[TensorValue], Iterator[Tuple[TensorValue, ...]]]:
        assert self.is_iterator, 'Use __call__ to call a non-iterator function.'
        return self.forward_generator(args, return_type=return_type, auto_broadcast=auto_broadcast, wrap_rv=wrap_rv)

    def _wrap_rv(self, rv, return_type, argument_values, auto_broadcast):
        if isinstance(rv, TensorValue):
            return rv
        elif isinstance(rv, tuple) and all(isinstance(v, TensorValue) for v in rv):
            return rv

        if return_type is None:
            return_type = self.return_type
        if return_type is None:
            raise RuntimeError('Return type can not be None if the function return is not a TensorValue.')

        if isinstance(return_type, tuple):
            if not isinstance(rv, tuple) and len(return_type) == 1:
                rv = (rv, )
            return tuple(self._wrap_single_rv(v, t, argument_values, auto_broadcast) for v, t in zip(rv, return_type))
        else:
            return self._wrap_single_rv(rv, return_type, argument_values, auto_broadcast)

    def _wrap_single_rv(self, rv, return_type, argument_values, auto_broadcast):
        if not self.support_batch:
            if isinstance(return_type, PyObjValueType):
                if isinstance(rv, TensorizedPyObjValues):
                    return TensorValue.from_tensorized_pyobj(rv, return_type)
                return TensorValue.from_scalar(rv, return_type)
            else:
                if isinstance(rv, torch.Tensor):
                    return TensorValue.from_tensor(rv, return_type)
                elif isinstance(rv, (bool, int, float)):
                    return TensorValue.from_scalar(rv, return_type)
                else:
                    raise ValueError(f'Unsupported return type: {type(rv)}')
        else:
            if isinstance(return_type, PyObjValueType):
                raise TypeError('Cannot return a PyObjValueType for a batched function.')
            else:
                if isinstance(rv, torch.Tensor):
                    first_tensor_arg = None
                    for arg in argument_values:
                        if isinstance(arg, TensorValue):
                            first_tensor_arg = arg
                            break
                    if not self.auto_broadcast or not auto_broadcast or first_tensor_arg is None:
                        raise ValueError('Cannot return a raw PyTorch tensor for a batched function without auto_broadcast.')
                    return TensorValue.from_tensor(rv, return_type, batch_variables=first_tensor_arg.batch_variables)
                else:
                    raise ValueError(f'Unsupported return type: {type(rv)}')


class PythonFunctionCrossRef(object):
    def __init__(self, cross_ref_name: str):
        self.cross_ref_name = cross_ref_name


def config_function_implementation(function: Optional[Callable] = None, *, function_quantized: Optional[Callable] = None, support_batch: bool = False, auto_broadcast: bool = True, use_object_names: bool = True, unwrap_values: Optional[bool] = None, include_executor_args: bool = False, is_iterator: bool = False) -> Callable:
    """Configure the implementation of a function in a domain.

    Args:
        function: the function to be wrapped.
        function_quantized: the quantized version of the function (can be None).
        support_batch: whether the function supports batched inputs.
        auto_broadcast: whether the executor should automatically broadcast the arguments before calling the function.
        use_object_names: whether the executor should use object names instead of indices.
        unwrap_values: whether the executor should unwrap the values before calling the function.
        include_executor_args: whether the executor should include itself as the first argument.
        is_iterator: whether the function is an iterator.

    Returns:
        the decorator.
    """

    function_implementation_configs = {
        'function_quantized': function_quantized,
        'support_batch': support_batch,
        'auto_broadcast': auto_broadcast,
        'use_object_names': use_object_names,
        'unwrap_values': unwrap_values,
        'include_executor_args': include_executor_args,
        'is_iterator': is_iterator
    }

    def wrapper(function: Callable, configs=function_implementation_configs):
        return PythonFunctionRef(function, **configs)

    if function is None:
        return wrapper
    return wrapper(function)


def _check_no_quantized_arguments(arguments):
    """A helper function to check that there are no quantized arguments. This function handles the migration from quantized tensor CSP computation to non-quantized tensor CSP computation."""
    # TODO(Jiayuan Mao @ 2023/08/15): remove this after the migration.
    for arg in arguments:
        if isinstance(arg, TensorValue):
            if isinstance(arg.dtype, TensorValueTypeBase):
                if arg.quantized and not arg.dtype.is_intrinsically_quantized():
                    raise RuntimeError('Quantized arguments are not supported.')


class PDSketchExecutor(TensorValueExecutorBase):
    """Planning domain expression executor. The basic concept in planning domain execution is the "state"."""

    def __init__(self, domain: Domain, parser: Optional[ParserBase] = None):
        """Initialize a PDSketch expression executor.

        Args:
            domain: the domain of this executor.
            parser: the parser to be used. This argument is optional. If provided, the execute function can take strings as input.
        """
        super().__init__(domain, parser)
        self._default_visitor = PDSketchExecutionDefaultVisitor(self)
        self._csp_visitor = PDSketchExecutionCSPVisitor(self)
        self._register_default_function_implementations()
        self._effect_update_from_simulation = False
        self._effect_action_index = None

    def _register_default_function_implementations(self):
        for t in self.domain.types.values():
            if isinstance(t, NamedTensorValueType):
                if isinstance(t.base_type, ScalarValueType):
                    self.register_function_implementation(
                        f'type::{t.typename}::equal',
                        PythonFunctionRef(
                            lambda x, y: TensorValue(BOOL, x.batch_variables, torch.eq(x.tensor, y.tensor), x.batch_dims),
                            support_batch=True, auto_broadcast=True, unwrap_values=False
                        )
                    )
                    self.register_function_implementation(
                        f'type::{t.typename}::hash',
                        PythonFunctionRef(lambda x: x.tensor.item(), support_batch=False, unwrap_values=False)
                    )
                elif isinstance(t.base_type, VectorValueType):
                    self.register_function_implementation(
                        f'type::{t.typename}::equal',
                        PythonFunctionRef(
                            lambda x, y: TensorValue(BOOL, x.batch_variables, torch.eq(x.tensor, y.tensor).all(dim=-1), x.batch_dims),
                            support_batch=True, auto_broadcast=True, unwrap_values=False
                        )
                    )

        for fname, cross_ref_name in self.domain.external_function_crossrefs.items():
            self.register_function_implementation(fname, PythonFunctionCrossRef(cross_ref_name))

    _domain: Domain

    @property
    def domain(self) -> Domain:
        return self._domain

    @property
    def effect_update_from_simulation(self) -> bool:
        """A context variable indicating whether the current effect should be updated from simulation, instead of the evaluation of expressions."""
        return self._effect_update_from_simulation

    @property
    def effect_action_index(self) -> Optional[int]:
        return self._effect_action_index

    def parse(self, string: Union[str, Expression], variables: Optional[Sequence[Variable]] = None) -> Expression:
        if isinstance(string, Expression):
            return string
        if self.parser is not None:
            return self.parser.parse_expression(string)
        return self._domain.parse(string, variables)

    _function_implementations: Dict[str, Union[PythonFunctionRef, PythonFunctionCrossRef]]

    @property
    def function_implementations(self) -> Dict[str, Union[PythonFunctionRef, PythonFunctionCrossRef]]:
        return self._function_implementations

    def register_function_implementation(self, name: str, func: Union[Callable, PythonFunctionRef, PythonFunctionCrossRef]):
        if isinstance(func, PythonFunctionRef):
            self._function_implementations[name] = func.set_executor(self)
        elif isinstance(func, PythonFunctionCrossRef):
            self._function_implementations[name] = func
        else:
            self._function_implementations[name] = PythonFunctionRef(func)

    def get_function_implementation(self, name: str) -> PythonFunctionRef:
        while name in self._function_implementations:
            func = self._function_implementations[name]
            if isinstance(func, PythonFunctionCrossRef):
                name = func.cross_ref_name
            else:
                return func
        raise KeyError(f'Function {name} not found.')

    def _execute(self, expression: Expression) -> TensorValueExecutorReturnType:
        if self.csp is not None:
            return self._csp_visitor.visit(expression)
        return self._default_visitor.visit(expression)

    def apply_precondition(self, operator: Union[Operator, OperatorApplier], state: TensorState, *args, csp: Optional[ConstraintSatisfactionProblem] = None) -> bool:
        """Apply the precondition of this operator to the given state.

        Args:
            operator: the operator to be applied.
            state: the state to be applied to.
            args: the arguments of the operator.
            csp: the CSP to be used for optimistic evaluation.

        Returns:
            bool: whether the precondition is satisfied.
        """

        if isinstance(operator, OperatorApplier):
            assert not operator.is_macro
            assert len(args) == 0, 'Operator applier does not support arguments.'
            operator, args = operator.operator, operator.arguments
        else:
            # NB(Jiayuan Mao @ 2023/03/17): sanity check due to the addition of macro operators.
            assert isinstance(operator, Operator)

        bounded_variables = compose_bvdict_args(operator.arguments, args, state=state)
        try:
            with self.with_state(state), self.with_csp(csp), self.with_bounded_variables(bounded_variables):
                for precondition in operator.preconditions:
                    pred_value = self._execute(precondition.bool_expr)
                    rv = pred_value.item()

                    if isinstance(rv, OptimisticValue):
                        if self.csp is not None:
                            self.csp.add_constraint(EqualityConstraint.from_bool(rv, True), note=f'precondition::{precondition.bool_expr.cached_string(-1)}')
                    else:
                        if rv < 0.5:
                            return False
            return True
        except Exception:
            logger.warning(f'Precondition evaluation failed: {precondition.bool_expr}.')
            raise

    def apply_precondition_debug(self, operator: Union[Operator, OperatorApplier], state: TensorState, *args, csp: Optional[ConstraintSatisfactionProblem] = None):
        """Apply the precondition of this operator to the given state, but in a dry-run mode. It will print out the evaluation results of each precondition.

        Example:
            .. code-block:: python

                (succ, state), csp = executor.apply(action, state, csp=csp)
                if succ:
                    pass
                else:
                    executor.apply_precondition_debug(action, state, csp=csp)  # will print out detailed infomation why the precondition is not satisfied.

        Args:
            operator: the operator to be applied.
            state: the state to be applied to.
            args: the arguments of the operator.
            csp: the CSP to be used for optimistic evaluation.
        """
        if isinstance(operator, OperatorApplier):
            assert not operator.is_macro
            assert len(args) == 0, 'Operator applier does not support arguments.'
            operator, args = operator.operator, operator.arguments
        else:
            # NB(Jiayuan Mao @ 2023/03/17): sanity check due to the addition of macro operators.
            assert isinstance(operator, Operator)

        bounded_variables = compose_bvdict_args(operator.arguments, args, state=state)
        try:
            with self.with_state(state), self.with_csp(csp), self.with_bounded_variables(bounded_variables):
                for precondition in operator.preconditions:
                    logger.info(f'Evaluate precondition: {precondition.bool_expr}')
                    pred_value = self._execute(precondition.bool_expr)
                    rv = pred_value.item()
                    logger.info(f'  Result = {rv}')

                    if isinstance(rv, OptimisticValue):
                        if self.csp is not None:
                            constraint = EqualityConstraint.from_bool(rv, True)
                            logger.info(f'  Add constraint: {constraint}')
        except Exception:
            logger.warning(f'Precondition evaluation failed: {precondition.bool_expr}.')
            raise

    def apply_effect(self, operator: Union[Operator, OperatorApplier], state: TensorState, *args, csp: Optional[ConstraintSatisfactionProblem] = None, action_index: Optional[int] = None, clone: bool = True) -> TensorState:
        """Apply the effect of this operator to the given state.

        Args:
            operator: the operator to be applied.
            state: the state to be applied to.
            args: the arguments of the operator.
            csp: the CSP to be used for optimistic evaluation.
            action_index: the index of the action in the trajectory (only effective when ``effect_update_from_simulation`` is active).
            clone: whether to clone the state before applying the effect.

        Returns:
            the new state after applying the effect.
        """
        if isinstance(operator, OperatorApplier):
            assert not operator.is_macro
            assert len(args) == 0, 'Operator applier does not support arguments.'
            operator, args = operator.operator, operator.arguments
        else:
            # NB(Jiayuan Mao @ 2023/03/17): sanity check due to the addition of macro operators.
            assert isinstance(operator, Operator)

        if clone:
            state = state.clone()

        bounded_variables = compose_bvdict_args(operator.arguments, args, state=state)

        try:
            with self.with_state(state), self.with_csp(csp), self.with_bounded_variables(bounded_variables):
                self._effect_action_index = action_index
                for effect in operator.effects:
                    self._effect_update_from_simulation = effect.update_from_simulation
                    self._execute(effect.assign_expr)
            return state
        except Exception:
            self._effect_action_index = None
            self._effect_update_from_simulation = False
            logger.warning(f'Effect application failed: {effect.assign_expr}.')
            raise

    def apply(self, operator: Union[Operator, OperatorApplier], state: TensorState, *args, clone: bool = True, csp: Optional[ConstraintSatisfactionProblem] = None, action_index: Optional[int] = None) -> Tuple[bool, TensorState]:
        """Apply an operator to a state.

        Args:
            operator: the operator to be applied.
            state: the state to be applied to.
            args: the arguments of the operator.
            clone: whether to clone the state before applying the effect.
            csp: the CSP to be used for optimistic evaluation.
            action_index: the index of the action in the trajectory (only effective when ``effect_update_from_simulation`` is active).

        Returns:
            a tuple of (whether the precondition is satisfied, the new state after applying the effect).
        """

        if isinstance(operator, (Operator, MacroOperator)):
            assert not operator.is_macro, 'Macro operators are not supported.'
            operator = operator(*args)
        else:
            assert not operator.operator.is_macro, 'Macro operators are not supported.'

        if self.apply_precondition(operator, state, csp=csp):
            return True, self.apply_effect(operator, state, csp=csp, clone=clone, action_index=action_index)
        return False, state

    def get_controller_args(self, operator: Union[Operator, OperatorApplier], state: TensorState, *args) -> Tuple[TensorValueExecutorReturnType, ...]:
        """Get the arguments of the controller of the given operator.

        Args:
            operator: the operator to be applied.
            state: the state to be applied to.
            args: the arguments of the operator.

        Returns:
            the arguments to the controller.
        """
        if isinstance(operator, OperatorApplier):
            assert not operator.is_macro
            assert len(args) == 0, 'Operator applier does not support arguments.'
            operator, args = operator.operator, operator.arguments
        else:
            # NB(Jiayuan Mao @ 2023/03/17): sanity check due to the addition of macro operators.
            assert isinstance(operator, Operator)

        bounded_variables = compose_bvdict_args(operator.arguments, args, state=state)

        try:
            with self.with_state(state), self.with_bounded_variables(bounded_variables):
                arguments = list()
                for arg in operator.controller.arguments:
                    arguments.append(self._execute(arg))
                return tuple(arguments)
        except Exception:
            logger.warning(f'Controller argument application failed: {arg}.')
            raise

    def new_state(self, object_names: ObjectNameArgument, object_types: ObjectTypeArgument = None, create_context: bool = False) -> Union[TensorState, Tuple[TensorState, 'StateDefinitionHelper']]:
        """Create a new state. This function also creates the state definition helper if the `create_context` argument is True. See the documentation of `StateDefinitionHelper` for more details.

        Args:
            object_names: the object names. It can be a list of strings (the names of the objects), or a dictionary mapping object names to their types. In this case, the `object_types` argument should be None.
            object_types: the object types, which is a list of :class:`~concepts.dsl.dsl_types.ObjectType` instances.
            create_context: whether to create the state definition helper.

        Returns:
            the new state if `create_context` is False, otherwise a tuple of the new state and the state definition helper.
        """

        if isinstance(object_names, dict):
            assert object_types is None
            object_types = tuple(object_names.values())
            object_names = tuple(object_names.keys())

        object_types_list = list()
        for object_type in object_types:
            if isinstance(object_type, str):
                object_types_list.append(self.domain.types[object_type])
            else:
                object_types_list.append(object_type)
        object_types = tuple(object_types_list)

        state = State({}, object_names, object_types)
        if create_context:
            return state, StateDefinitionHelper(self, state)
        return state


class PDSketchExecutionDefaultVisitor(ExpressionVisitor):
    """The underlying default implementation for :class:`PDSketchExecutor`. This function does not handle CSPs (a.k.a. optimistic execution)."""

    def __init__(self, executor: PDSketchExecutor):
        """Initialize a PDExpressionExecutionDefaultVisitor.

        Args:
            executor: the executor that uses this visitor.
        """
        self.executor = executor

    @property
    def csp(self) -> ConstraintSatisfactionProblem:
        return self.executor.csp

    def visit_variable_expression(self, expr: E.VariableExpression) -> TensorValueExecutorReturnType:
        variable = expr.variable
        return self.executor.bounded_variables[variable.dtype.typename][variable.name]

    def visit_object_constant_expression(self, expr: E.ObjectConstantExpression) -> StateObjectReference:
        const = expr.constant
        state = self.executor.state
        assert isinstance(state, NamedObjectTensorState)
        return StateObjectReference(
            const.name,
            state.get_typed_index(const.name, const.dtype.typename)
        )

    def visit_constant_expression(self, expr: E.ConstantExpression) -> TensorValueExecutorReturnType:
        value = expr.constant
        assert isinstance(value, TensorValue)
        return value

    def visit_function_application_expression(self, expr: E.FunctionApplicationExpression) -> TensorValueExecutorReturnType:
        function = expr.function
        return_type = function.return_type
        state = self.executor.state
        assert isinstance(function, Predicate)

        if function.is_generator_placeholder:  # always true branch
            argument_values = self.forward_args(*expr.arguments, force_tuple=True)
            argument_values = expand_argument_values(argument_values)
            _check_no_quantized_arguments(argument_values)

            batched_value = None
            for argv in argument_values:
                if isinstance(argv, TensorValue):
                    batched_value = argv
                    break
            assert batched_value is not None

            rv = torch.ones(
                batched_value.tensor.shape[:batched_value.total_batch_dims],
                dtype=torch.bool, device=batched_value.tensor.device
            )
            assert return_type == BOOL

            # Add "true" asserts to the csp.
            if self.csp is not None and not self.executor.optimistic_execution:
                expr_string = expr.cached_string(-1)
                for ind in _iter_tensor_indices(rv):
                    self.csp.add_constraint(Constraint.from_function(
                        function,
                        [argv.fast_index(tuple(ind)) for argv in argument_values],
                        True
                    ), note=f'{expr_string}::{ind}' if len(ind) > 0 else expr_string)

            return TensorValue(
                BOOL, batched_value.batch_variables,
                rv, batch_dims=state.batch_dims
            )
        elif function.is_cacheable and function.name in state.features:
            argument_values = self.forward_args(*expr.arguments, force_tuple=True)
            argument_values = [v.index if isinstance(v, StateObjectReference) else v for v in argument_values]
            batch_variables = [arg.name for arg, value in zip(expr.arguments, argument_values) if value == QINDEX]
            value = state.features[function.name][tuple(argument_values)]
            return value.rename_batch_variables(batch_variables)
        elif function.is_derived:
            argument_values = self.forward_args(*expr.arguments, force_tuple=True)
            with self.executor.with_bounded_variables({k: v for k, v in zip(function.arguments, argument_values)}):
                return self.visit(function.derived_expression)
        else:
            # dynamic predicate is exactly the same thing as a pre-defined external function.
            argument_values = self.forward_args(*expr.arguments, force_tuple=True)
            # only supports external functions with a single return value.
            return self.forward_external_function('predicate::' + function.name, argument_values, return_type, expression=expr)

    def visit_bool_expression(self, expr: E.BoolExpression, argument_values: Optional[Tuple[TensorValueExecutorReturnType, ...]] = None) -> TensorValueExecutorReturnType:
        if argument_values is None:
            argument_values = self.forward_args(*expr.arguments, force_tuple=True)
            argument_values = expand_argument_values(argument_values)

        _check_no_quantized_arguments(argument_values)

        assert len(argument_values) > 0
        assert all(isinstance(v, TensorValue) for v in argument_values)

        dtype = argument_values[0].dtype
        batch_variables = argument_values[0].batch_variables

        if expr.bool_op is BoolOpType.NOT:
            assert len(argument_values) == 1
            return TensorValue(
                dtype, batch_variables,
                1 - argument_values[0].tensor,
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
        else:
            raise ValueError(f'Unknown bool op type: {expr.bool_op}.')

    def visit_quantification_expression(self, expr: E.QuantificationExpression, value: Optional[TensorValue] = None) -> TensorValueExecutorReturnType:
        if value is None:
            with self.executor.new_bounded_variables({expr.variable: QINDEX}):
                value = self.forward_args(expr.expression)
            assert isinstance(value, TensorValue)
        _check_no_quantized_arguments([value])

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
                MaskedTensorStorage(tensor.amin(dim=variable_index), None, mask),
                batch_dims=self.executor.state.batch_dims
            )
        elif expr.quantification_op is QuantificationOpType.EXISTS:
            return TensorValue(
                dtype, batch_variables,
                MaskedTensorStorage(tensor.amax(dim=variable_index), None, mask),
                batch_dims=self.executor.state.batch_dims
            )
        else:
            raise ValueError(f'Unknown quantifier type: {expr.quantification_op}.')

    def visit_predicate_equal_expression(self, expr: E.PredicateEqualExpression, feature: Optional[TensorValue] = None, value: Optional[TensorValue] = None) -> TensorValueExecutorReturnType:
        if feature is None or value is None:
            feature, value = self.forward_args(expr.predicate, expr.value)
            feature, value = expand_argument_values([feature, value])
        _check_no_quantized_arguments([feature, value])

        if isinstance(feature.dtype, PyObjValueType):
            rv = self.forward_external_function(f'type::{feature.dtype.typename}::equal', [feature, value], BOOL)
        elif isinstance(feature.dtype, NamedTensorValueType):
            rv = self.forward_external_function(f'type::{feature.dtype.typename}::equal', [feature, value], BOOL)
        else:
            raise NotImplementedError(f'Unsupported FeatureEqual computation for dtype {feature.dtype} and {value.dtype}.')

        return rv

    def visit_assign_expression(self, expr: E.AssignExpression):
        state = self.executor.state
        argument_values = self.forward_args(*expr.predicate.arguments, force_tuple=True)
        value = self.forward_args(expr.value)
        argument_values = [v.index if isinstance(v, StateObjectReference) else v for v in argument_values]

        # if state.features[expr.predicate.function.name].quantized:
        #     if not value.quantized:
        #         value = self.executor.value_quantizer.quantize_value(value)
        # else:
        #     if value.quantized:
        #         value = self.executor.value_quantizer.unquantize_value(value)
        _check_no_quantized_arguments([value, *argument_values, state.features[expr.predicate.function.name]])

        state.features[expr.predicate.function.name][tuple(argument_values)] = value

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
        value = self.forward_args(expr.value)
        condition = self.forward_args(expr.condition)
        _check_no_quantized_arguments([value, condition, state.features[expr.predicate.function.name]])

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

    def forward_args(self, *args, force_tuple: bool = False) -> Union[TensorValueExecutorReturnType, Tuple[TensorValueExecutorReturnType, ...]]:
        if len(args) == 1 and not force_tuple:
            return self.visit(args[0])
        else:
            return tuple(self.visit(arg) for arg in args)

    def forward_external_function(
        self, function_name: str, argument_values: Sequence[TensorValueExecutorReturnType],
        return_type: Union[TensorValueTypeBase, PyObjValueType], auto_broadcast: bool = False, expression: Optional[Expression] = None
    ) -> TensorValue:
        external_function = self.executor.get_function_implementation(function_name)
        assert isinstance(external_function, PythonFunctionRef)
        return external_function.forward(argument_values, return_type=return_type, auto_broadcast=auto_broadcast)


class PDSketchExecutionCSPVisitor(PDSketchExecutionDefaultVisitor):
    def __init__(self, executor: PDSketchExecutor):
        super().__init__(executor)

    def forward_external_function(
        self, function_name: str, argument_values: Sequence[TensorValueExecutorReturnType],
        return_type: Union[TensorValueTypeBase, PyObjValueType], auto_broadcast: bool = True, expression: Optional[E.FunctionApplicationExpression] = None
    ) -> TensorValue:
        argument_values = expand_argument_values(argument_values)
        optimistic_masks = [is_optimistic_value(argv.tensor_optimistic_values) for argv in argument_values if argv.tensor_optimistic_values is not None]
        if len(optimistic_masks) > 0:
            optimistic_mask = torch.stack(optimistic_masks, dim=-1).any(dim=-1)

            # TODO(Jiayuan Mao @ 2023/01/26): fix this hack, add a proper flag to indicate whether a function has to be executed in the simulator.
            if not self.executor.has_function_implementation('predicate::' + function_name):
                optimistic_mask[...] = True

            rv = super().forward_external_function(function_name, argument_values, return_type=return_type, auto_broadcast=auto_broadcast, expression=expression)

            if optimistic_mask.sum().item() == 0:
                return rv

            rv.init_tensor_optimistic_values()

            if self.executor.optimistic_execution:
                rv.tensor_optimistic_values[optimistic_mask.nonzero(as_tuple=True)] = OPTIM_MAGIC_NUMBER_MAGIC
            else:
                expr_string = expression.cached_string(-1)
                for ind in optimistic_mask.nonzero().tolist():
                    ind = tuple(ind)
                    new_identifier = self.executor.csp.new_var(return_type)
                    rv.tensor_optimistic_values[ind] = new_identifier
                    self.csp.add_constraint(Constraint.from_function(
                        expression.function,
                        [argv.fast_index(ind) for argv in argument_values],
                        new_identifier
                    ), note=f'{expr_string}::{ind}' if len(ind) > 0 else expr_string)
            return rv

            # TODO(Jiayuan Mao @ 2023/08/16): implement a faster verion of the computation.
            # if optimistic_mask.sum().item() == 0:
            #     pass  # just do the standard execution.
            # else:
            #     retain_mask = torch.logical_not(optimistic_mask)
            #     rv = torch.zeros(
            #         argument_values[0].tensor.shape,
            #         dtype=torch.int64,
            #         device=argument_values[0].tensor.device
            #     )

            #     if retain_mask.sum().item() > 0:
            #         argument_values_r = [TensorValue(argv.dtype, ['?x'], argv.tensor[retain_mask], 0, quantized=argv.quantized) for argv in argument_values]
            #         # No need to broadcast again.
            #         rv_r = super().forward_external_function(function_name, argument_values_r, return_type, auto_broadcast=False)
            #         if not rv_r.quantized:
            #             rv_r = executor.value_quantizer.quantize_value(rv_r)
            #         rv[retain_mask] = rv_r.tensor

            #     expr_string = expression.cached_string(-1)
            #     for ind in optimistic_mask.nonzero().tolist():
            #         ind = tuple(ind)
            #         new_identifier = self.executor.csp.new_var(return_type)
            #         rv[ind] = new_identifier
            #         self.csp.add_constraint(Constraint.from_function(
            #             expression.function,
            #             [argv.tensor[ind].item() if argv.quantized else argv.tensor[ind] for argv in argument_values],
            #             new_identifier
            #         ), note=f'{expr_string}::{ind}' if len(ind) > 0 else expr_string)

            #     return TensorValue(
            #         expression.function.return_type, argument_values[0].batch_variables if len(argument_values) > 0 else [],
            #         rv,
            #         batch_dims=executor.state.batch_dims, quantized=True
            #     )

        return super().forward_external_function(function_name, argument_values, return_type=return_type, auto_broadcast=auto_broadcast, expression=expression)

    def visit_function_application_expression(self, expr: E.FunctionApplicationExpression) -> TensorValueExecutorReturnType:
        return super().visit_function_application_expression(expr)

    def visit_bool_expression(self, expr: E.BoolExpression, argument_values: Optional[Tuple[TensorValueExecutorReturnType, ...]] = None) -> TensorValueExecutorReturnType:
        if argument_values is None:
            argument_values = self.forward_args(*expr.arguments, force_tuple=True)
            argument_values = list(expand_argument_values(argument_values))

        for argv in argument_values:
            assert argv.dtype == BOOL, 'Boolean expression only supports boolean values in CSP mode.'

        optimistic_masks = [is_optimistic_value(argv.tensor_optimistic_values) for argv in argument_values if argv.tensor_optimistic_values is not None]
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
            value_transposed
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

        rv = super().visit_predicate_equal_expression(expr, feature, value)
        optimistic_mask = torch.logical_or(is_optimistic_value(feature.tensor_optimistic_values), is_optimistic_value(value.tensor_optimistic_values))

        if optimistic_mask.sum().item() == 0:
            return rv

        rv.init_tensor_optimistic_values()
        if self.executor.optimistic_execution:
            rv.tensor_optimistic_values[optimistic_mask.nonzero(as_tuple=True)] = OPTIM_MAGIC_NUMBER_MAGIC
        else:
            expr_string = expr.cached_string(-1)
            for ind in optimistic_mask.nonzero().tolist():
                ind = tuple(ind)
                this_argv = feature.fast_index(ind), value.fast_index(ind)
                new_identifier = self.csp.new_var(BOOL)
                rv.tensor_optimistic_values[ind] = new_identifier
                self.csp.add_constraint(EqualityConstraint(
                    *[cvt_opt_value(v, feature.dtype) for v in this_argv],
                    OptimisticValue(BOOL, new_identifier)
                ), note=f'{expr_string}::{ind}' if len(ind) > 0 else expr_string)
        return rv

    def visit_assign_expression(self, expr: E.AssignExpression) -> Any:
        if self.executor.effect_update_from_simulation:
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
                        SimulationFluentConstraintFunction(self.executor.effect_action_index, expr.predicate.function, entry_values),
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
        if self.executor.effect_update_from_simulation:
            raise NotImplementedError('Conditional assign is not supported in simulation mode.')
        if self.executor.optimistic_execution:
            raise RuntimeError('Optimistic execution is not supported for conditional assign.')

        state = self.executor.state
        argument_values = self.forward_args(*expr.predicate.arguments, force_tuple=True)
        value = self.forward_args(expr.value)
        condition = self.forward_args(expr.condition)
        _check_no_quantized_arguments([value, condition, state.features[expr.predicate.function.name]])

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


def expand_argument_values(argument_values: Sequence[Union[TensorValue, int, str, slice, StateObjectReference]]) -> List[TensorValue]:
    """Expand a list of argument values to the same batch size.

    Args:
        argument_values: a list of argument values.

    Returns:
        the result list of argument values. All return values will have the same batch size.
    """
    has_slot_var = False
    for arg in argument_values:
        if isinstance(arg, TensorValue):
            for var in arg.batch_variables:
                if var == '??':
                    has_slot_var = True
                    break
    if has_slot_var:
        return list(argument_values)

    if len(argument_values) < 2:
        return list(argument_values)

    argument_values = list(argument_values)
    batch_variables = list()
    batch_sizes = list()
    for arg in argument_values:
        if isinstance(arg, TensorValue):
            for var in arg.batch_variables:
                if var not in batch_variables:
                    batch_variables.append(var)
                    batch_sizes.append(arg.get_variable_size(var))
        else:
            assert isinstance(arg, (int, str, slice, StateObjectReference)), arg

    masks = list()
    for i, arg in enumerate(argument_values):
        if isinstance(arg, TensorValue):
            argument_values[i] = arg.expand(batch_variables, batch_sizes)
            if argument_values[i].tensor_mask is not None:
                masks.append(argument_values[i].tensor_mask)

    if len(masks) > 0:
        final_mask = torch.stack(masks, dim=-1).amin(dim=-1)
        for arg in argument_values:
            if isinstance(arg, TensorValue):
                arg.tensor_mask = final_mask
                arg._mask_certified_flag = True  # now we have corrected the mask.
    return argument_values


class _StateDefPredicate(object):
    def __init__(self, function, arguments):
        self.function = function
        self.arguments = arguments


class _StateDefPredicateApplier(object):
    def __init__(self, function):
        self.function = function

    def __call__(self, *args):
        return _StateDefPredicate(self.function, args)


class StateDefinitionHelper(object):
    """A helper class to define the planning domain state (:class:`~concepts.pdsketch.domain.State`). Typically you should use :meth:`~PDSketchExecutor.new_state` to create an instance of this helper class.

    The most important feature of this class is that it supports a human-friendly definition of Boolean predicates. For example, you can define the grounding values for a predicate ``p`` as follows:

    .. code-block:: python

        state, ctx = executor.new_state({'a': ObjectType('person'), 'b': ObjectType('person')})
        state.define_predicates([
            ctx.p('a', 'b')
        ])  # define the predicate p(a, b) as True. All other values are False by convention.

    """

    def __init__(self, executor: PDSketchExecutor, state: State):
        """Initialize a new instance of the state definition helper.

        Args:
            executor: the executor.
            state: a (possibly empty) state to be modified.
        """
        self.executor = executor
        self.domain = executor.domain
        self.state = state

    def get_predicate(self, name: str) -> _StateDefPredicateApplier:
        """Get a predicate definition helper by name. This can be used to define the grounding values of a predicate, e.g., ``ctx.get_predicate('p')('a', 'b')``.

        Args:
            name: the name of the predicate.

        Returns:
            a predicate definition helper.
        """
        if name in self.domain.functions:
            return _StateDefPredicateApplier(self.domain.functions[name])
        else:
            name = name.replace('_', '-')
            if name in self.domain.functions:
                return _StateDefPredicateApplier(self.domain.functions[name])
            else:
                raise NameError(f'Predicate {name} is not defined in the domain.')

    def __getattr__(self, name):
        return self.get_predicate(name)

    def define_predicates(self, predicates: Sequence[_StateDefPredicate]):
        """Define a list of grounding values of predicates. See the example in :class:`StateDefinitionHelper`.

        Args:
            predicates: a list of grounding values, created by ``ctx.p('a', 'b')``.
        """
        for function in self.domain.functions.values():
            if function.return_type != BOOL:
                continue
            if function.name in self.state.features.all_feature_names:
                continue

            if function.is_state_variable:
                sizes = list()
                for arg_def in function.arguments:
                    sizes.append(len(self.state.object_type2name[arg_def.typename]) if arg_def.typename in self.state.object_type2name else 0)
                self.state.features[function.name] = TensorValue.make_empty(BOOL, [var.name for var in function.arguments], tuple(sizes))

        for pred in predicates:
            assert isinstance(pred, _StateDefPredicate)
            assert pred.function.return_type == BOOL
            name = pred.function.name
            arguments = [self.state.get_typed_index(arg) for arg in pred.arguments]
            self.state.features[name].tensor[tuple(arguments)] = 1

    def define_feature(self, feature_name: str, tensor_or_mapping: Union[torch.Tensor, TensorizedPyObjValues, Mapping[Union[str, Tuple[str, ...]], Union[bool, int, float, torch.Tensor, TensorizedPyObjValues]]]):
        """Define a relational feature directly with :class:`torch.Tensor` objects. For example,

        .. code-block:: python

            state, ctx = executor.new_state({'a': ObjectType('person'), 'b': ObjectType('person')})
            # assume p is a Boolean predicate with the signature person x person -> bool
            state.define_feature('p', torch.tensor([[1, 0], [0, 1]]), quantized=True)

        Args:
            feature_name: the name of the feature.
            tensor_or_mapping: a tensor or a mapping from argument values to tensors. If a tensor is given, it is assumed that the tensor has exactly the same shape as the tensor of the feature.
                If the input is a mapping, the keys of the mapping are tuples (entry indices), and the values are tensors. The tensors will be filled into an all-zero tensor of the same shape as the feature tensor.
        """
        function = self.domain.functions[feature_name]
        sizes = list()
        for arg_def in function.arguments:
            sizes.append(len(self.state.object_type2name[arg_def.typename]) if arg_def.typename in self.state.object_type2name else 0)
        sizes = tuple(sizes)
        batch_variables = [var.name for var in function.arguments]
        return_type = function.return_type

        if isinstance(return_type, PyObjValueType):
            if isinstance(tensor_or_mapping, TensorizedPyObjValues):
                self.state.features[feature_name] = TensorValue.from_tensorized_pyobj(tensor_or_mapping, return_type, batch_variables)
                return
        else:
            if isinstance(tensor_or_mapping, torch.Tensor):
                self.state.features[feature_name] = TensorValue.from_tensor(tensor_or_mapping, return_type, batch_variables)
                return

        self.state.features[feature_name] = feature = TensorValue.make_empty(return_type, [var.name for var in function.arguments], sizes)
        for key, value in tensor_or_mapping.items():
            if isinstance(key, tuple):
                args = [self.state.get_typed_index(arg) for arg in key]
            else:
                assert isinstance(key, str)
                args = [self.state.get_typed_index(key)]
            feature[tuple(args)] = value

    def define_pyobj_feature(self, feature_name: str, pyobj_list: List[Any]):
        """Define a feature with a list of Python objects. The objects will be converted to tensors using the underlying :class:`~concepts.dsl.executors.tensor_value_executor.PyObjStore` of the executor.

        Args:
            feature_name: the name of the feature.
            pyobj_list: a list of Python objects (they can also be nested lists).
        """

        function = self.domain.functions[feature_name]
        sizes = list()
        for arg_def in function.arguments:
            sizes.append(len(self.state.object_type2name[arg_def.typename]) if arg_def.typename in self.state.object_type2name else 0)
        sizes = tuple(sizes)
        batch_variables = [var.name for var in function.arguments]
        return_type = function.return_type

        value = TensorValue.from_tensorized_pyobj(
            TensorizedPyObjValues(return_type, pyobj_list, sizes),
            dtype=return_type,
            batch_variables=batch_variables
        )

        self.state.features[feature_name] = value

    def set_value(self, feature_name, arguments: Union[Sequence[str], Sequence[int]], value: Union[torch.Tensor, bool, int, float, Any]):
        """Set a single entry in the feature representation. When the feature tensor has not been created, it will be created automatically.
        Note that, when creating the feature tensor, we will use ``quantized`` to determine whether the feature should be quantized or not in the state.

        .. code-block:: python

            state, ctx = executor.new_state({'a': ObjectType('person'), 'b': ObjectType('person')})
            # assume p is a Boolean predicate with the signature person x person -> bool
            state.set_value('p', ('a', 'b'), 1, quantized=True)

        Args:
            feature_name: the name of the feature.
            arguments: the argument values of the entry.
            value: the value of the entry.
        """

        function = self.domain.functions[feature_name]
        return_type = function.return_type

        if feature_name not in self.state.features:
            sizes = list()
            for arg_def in function.arguments:
                sizes.append(len(self.state.object_type2name[arg_def.typename]) if arg_def.typename in self.state.object_type2name else 0)
            sizes = tuple(sizes)
            self.state.features[feature_name] = TensorValue.make_empty(return_type, [var.name for var in function.arguments], sizes)

        assert len(arguments) == len(function.arguments)
        arguments = [self.state.get_typed_index(arg, arg_def.dtype.typename) if isinstance(arg, str) else arg for arg, arg_def in zip(arguments, function.arguments)]
        self.state.features[feature_name][tuple(arguments)] = value


class GeneratorManager(object):
    """The :class:`GeneratorManager` is used to manage calls to generators in the function dmoain. It is particularly useful for
    keep tracking of historical values generated by the generators."""

    def __init__(self, executor: PDSketchExecutor, store_history: bool = True):
        """Initialize the generator manager.

        Args:
            executor: the executor.
            store_history: whether to store the historical values generated by the generators.
        """

        self.executor = executor
        self.generator_calls = defaultdict(list)
        self.generator_calls_successful = defaultdict(list)
        self.generator_calls_count = defaultdict(int)

        self._store_history = store_history

    executor: PDSketchExecutor
    """The executor."""

    generator_calls: Dict[str, List[Tuple[Tuple[Any, ...], Tuple[Any, ...]]]]
    """Mappings from generator names to the list of calls made to the generator, including a tuple of the arguments and a tuple of the return values."""

    generator_calls_successful: Dict[str, List[bool]]
    """Mappings from generator names to the list of Boolean values indicating whether the generated values lead to successful solution."""

    @property
    def store_history(self) -> bool:
        """Whether to store the historical values generated by the generators."""
        return self._store_history

    def call(self, g: Generator, max_generator_trials: int, args: Tuple[Any, ...], constraint_list: Optional[List[Constraint]] = None) -> Iterator[Tuple[Tuple[str, int], Any]]:
        """Call a generator.

        Args:
            g: the generator.
            max_generator_trials: the maximum number of trials to generate values.
            args: the arguments of the generator.
            constraint_list: the list of constraints to be satisfied by the generated values. This will be passed to the generator function if the list contains more than one constraint.

        Yields:
            A tuple of (index, generated value). The index is a tuple of (generator_name, value_index).
        """

        generator_name = g.function.name
        generator = wrap_singletime_function_to_iterator(
            self.executor.get_function_implementation(generator_name),
            max_generator_trials
        )
        if constraint_list is not None or not isinstance(constraint_list, list):
            generator = generator(*args, return_type=g.output_type)
        else:
            generator = generator(*args, constraint_list, return_type=g.output_type)

        self.generator_calls_count[generator_name] += 1
        first = True
        for result in generator:
            if self._store_history:
                self.generator_calls[generator_name].append((args, result))
                self.generator_calls_successful[generator_name].append(False)
            if not first:
                self.generator_calls_count[generator_name] += 1
            else:
                first = False
            index = generator_name, len(self.generator_calls[generator_name]) - 1

            if not isinstance(result, tuple) and g.function.ftype.is_singular_return:
                result = (result, )

            if not g.function.ftype.is_singular_return:
                assert len(result) == len(generator.function.return_type)

            yield index, result

    def mark_success(self, assignment_dict: AssignmentDict):
        """Mark the values in an assignment dictionary as successful.

        Args:
            assignment_dict: the assignment dictionary.
        """
        assert self._store_history, 'Cannot mark success if history is not stored.'
        for _, value in assignment_dict.items():
            if value.generator_index is not None:
                name, index = value.generator_index
                self.generator_calls_successful[name][index] = True

    def export_generator_calls(self) -> Dict[str, List[Tuple[Tuple[Any, ...], Tuple[Any, ...], bool]]]:
        """Export the generator calls.

        Returns:
            a dictionary mapping from generator names to the list of calls made to the generator, including a tuple of the arguments
            and a tuple of the return values, and a Boolean value indicating whether the generated values lead to successful solution.
        """

        output_dict = defaultdict(list)
        for name, calls in self.generator_calls.items():
            for index, (args, result) in enumerate(calls):
                output_dict[name].append((args, result, self.generator_calls_successful[name][index]))
        return output_dict

    def export_generator_stats(self, divide_by: float = 1) -> str:
        """Export the generator statistics.

        Returns:
            a string containing the generator statistics.
        """

        rows = list()
        for name, count in self.generator_calls_count.items():
            rows.append((name, count / divide_by))
        rows.append(('Total', sum(count / divide_by for count in self.generator_calls_count.values())))
        return tabulate(rows, headers=['Generator', 'Calls'])


def wrap_singletime_function_to_iterator(function: PythonFunctionRef, max_examples: int) -> Callable[..., Iterator[Any]]:
    """Wrap a function that returns a single value to an iterator function.

    Args:
        function: the function.
        max_examples: the maximum number of examples.

    Returns:
        the iterator function.
    """

    if function.is_iterator:
        @functools.wraps(function)
        def wrapped(*args, **kwargs) -> Iterator[Any]:
            try:
                yield from itertools.islice(function.iter_from(*args, **kwargs), max_examples)
            except Exception as e:
                logger.warning(f'Exception raised when calling generator {function}: {e}')

        return wrapped

    @functools.wraps(function)
    def wrapped(*args, **kwargs) -> Iterator[Any]:
        for _ in range(max_examples):
            rv = function(*args, **kwargs)
            if rv is None:
                break
            yield rv

    return wrapped

