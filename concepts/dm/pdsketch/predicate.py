#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : predicate.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/26/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import contextlib
from typing import Any, Optional, Union, Sequence, Tuple, Set, Dict, Callable

from jacinle.utils.enum import JacEnum
from jacinle.utils.meta import repr_from_str

import concepts.dsl.expression as E
from concepts.dsl.dsl_types import ObjectType, Variable
from concepts.dsl.dsl_functions import FunctionType, Function
from concepts.dsl.expression import Expression, ExpressionDefinitionContext
from concepts.dsl.expression_utils import iter_exprs
from concepts.dsl.expression import FunctionApplicationExpression, VariableExpression, ObjectConstantExpression, ValueOutputExpression, VariableAssignmentExpression
from concepts.dsl.expression import ListCreationExpression, ListExpansionExpression, ListFunctionApplicationExpression
from concepts.dsl.expression import BoolExpression, QuantificationExpression, FindAllExpression, BoolOpType, ObjectCompareExpression, ValueCompareExpression
from concepts.dsl.expression import PredicateEqualExpression, AssignExpression, ConditionalSelectExpression, DeicticSelectExpression, ConditionalAssignExpression, DeicticAssignExpression
from concepts.dsl.expression_visitor import IdentityExpressionVisitor

__all__ = ['Predicate', 'flatten_expression', 'get_used_state_variables', 'is_simple_bool', 'split_simple_bool', 'get_simple_bool_predicate']


class FunctionEvaluationDefinitionMode(JacEnum):
    """The evaluation mode of a function definition. This enum has three values:

    - ``FUNCTIONAL``: the function is a pure function.
    - ``SIMULATION``: the function is a simulation-dependent function, i.e., it is a function that can only be evaluated given the current state in simulation.
    - ``EXECUTION``: the function is an execution-dependent function, i.e., it is a function that can only be evaluated given the current state in execution.
    """

    FUNCTIONAL = 'functional'
    SIMULATION = 'simulation'
    EXECUTION = 'execution'

    @classmethod
    def from_bools(cls, simulation: bool, execution: bool):
        if simulation:
            assert not execution, 'Cannot set both simulation and execution mode.'
            return cls.SIMULATION
        elif execution:
            return cls.EXECUTION
        else:
            return cls.FUNCTIONAL

    def get_prefix(self) -> str:
        return f'[[{self.value}]]'


class Predicate(Function):
    """A predicate is a special function that can part of the observation or state space.
    Predicate does not support overloaded function types. Currently, predicates are only used in planning domains."""

    def __init__(
        self,
        name: str,
        ftype: FunctionType,
        derived_expression: Optional[str] = None,
        overridden_call: Optional[Callable] = None,
        observation: Optional[bool] = None,
        state: Optional[bool] = None,
        generator_placeholder: Optional[bool] = None,
        inplace_generators: Optional[Sequence[str]] = None,
        simulation: bool = False,
        execution: bool = False,
        is_sgc_function: bool = False
    ):
        """Initialize a predicate.

        Args:
            name: the name of the predicate.
            ftype: the function type of the predicate.
            derived_expression: the derived expression of the predicate.
            overridden_call: the overridden call of the predicate.
            observation: whether the predicate is an observation variable.
            state: whether the predicate is a state variable.
            generator_placeholder: whether the predicate is a generator placeholder.
            inplace_generators: the names of the inplace defined generators associated with this predicate. This value is stored for macro operator extension.
            simulation: whether the predicate is a simulation-dependent function.
            execution: whether the predicate is an execution-dependent function.
            is_sgc_function: whether the predicate is a state-goal-constraint function.
        """
        assert isinstance(name, str)
        super().__init__(name, ftype, derived_expression=derived_expression, overridden_call=overridden_call)

        # self.is_derived has been computed in super().__init__
        self.is_static = False
        self.is_cacheable = self._guess_is_cacheable()

        self.is_observation_variable = observation if observation is not None else self.is_cacheable
        self.is_state_variable = state if state is not None else self.is_cacheable
        self.is_generator_placeholder = generator_placeholder if generator_placeholder is not None else False
        self.inplace_generators = tuple(inplace_generators) if inplace_generators is not None else tuple()
        self.evaluation_mode = FunctionEvaluationDefinitionMode.from_bools(simulation, execution)
        self.is_sgc_function = is_sgc_function

        # from concepts.dm.pdsketch.strips.ao_discretization import AOFeatureCodebook
        # self.ao_discretization: Optional[AOFeatureCodebook] = None  # for AODiscretization
        self.ao_discretization = None

        self._check_flags_sanity()

    is_static: bool
    """Whether the predicate is a static predicate. I.e., the predicate will never be changed."""

    is_cacheable: bool
    """Whether the predicate can be cached. Specifically, if it contains only "ObjectTypes" as arguments."""

    is_observation_variable: bool
    """Whether the predicate is an observation variable."""

    is_state_variable: bool
    """Whether the predicate is a state variable."""

    is_generator_placeholder: bool
    """Whether the predicate is a generator placeholder."""

    inplace_generators: Tuple[str, ...]
    """The names of the inplace defined generators associated with this predicate."""

    evaluation_mode: FunctionEvaluationDefinitionMode
    """The evaluation mode of the predicate."""

    is_sgc_function: bool
    """Whether the predicate is a state-goal-constraint function."""

    # ao_discretization: Optional['AOFeatureCodebook']
    # """The discretization codebook for this predicate."""

    def _guess_is_cacheable(self) -> bool:
        """Return whether the function can be cached. Specifically, if it contains only "ObjectTypes" as arguments, it can be statically evaluated."""
        for arg_def in self.arguments:
            if not isinstance(arg_def.dtype, ObjectType):
                return False
        return True

    def mark_static(self, flag: bool = True):
        """Mark a predicate as static (i.e., its grounded value will never change).

        Args:
            flag: Whether to mark the predicate as static.
        """
        self.is_static = flag

    def _check_flags_sanity(self):
        if self.is_observation_variable:
            assert self.is_cacheable and not self.is_derived
        if self.is_state_variable:
            assert self.is_cacheable
            if self.is_derived and self.evaluation_mode is FunctionEvaluationDefinitionMode.FUNCTIONAL:
                for predicate_def in get_used_state_variables(self.derived_expression):
                    assert predicate_def.is_observation_variable and not predicate_def.is_state_variable
            else:
                assert self.is_cacheable
        if self.is_generator_placeholder:
            assert not self.is_derived and not self.is_state_variable and not self.is_observation_variable

    @property
    def flags(self) -> Dict[str, bool]:
        """Return the flags of the predicate, which is a dictionary of {flag_name: flag_value}."""
        return {
            'is_derived': self.is_derived,
            'is_static': self.is_static,
            'is_cacheable': self.is_cacheable,
            'is_observation_variable': self.is_observation_variable,
            'is_state_variable': self.is_state_variable,
            'is_generator_placeholder': self.is_generator_placeholder,
        }

    def rename(self, new_name: str) -> 'Predicate':
        """Rename the predicate."""
        return Predicate(
            new_name, self.ftype,
            derived_expression=self.derived_expression,
            overridden_call=self.overridden_call,
            observation=self.is_observation_variable,
            state=self.is_state_variable,
            generator_placeholder=self.is_generator_placeholder,
            inplace_generators=self.inplace_generators
        )

    def __str__(self) -> str:
        flags = list()

        if self.is_observation_variable:
            flags.append('observation')
        if self.is_state_variable:
            flags.append('state')
        if self.is_generator_placeholder:
            flags.append('gen')
        if self.is_cacheable:
            flags.append('cacheable')
        if self.is_static:
            flags.append('static')
        flags_string = '[' + ', '.join(flags) + ']' if len(flags) > 0 else ''
        arguments = ', '.join([str(arg) for arg in self.arguments])
        if self.is_sgc_function:
            arguments = 'SGC, ' + arguments
        return_type_name = self.return_type.typename
        if self.is_generator_function:
            return_type_name = f'Gen[{return_type_name}]'
        fmt = f'{self.name}{flags_string}({arguments}) -> {return_type_name}'
        if self.derived_expression is not None:
            fmt += ' {\n'
            fmt += '  ' + str(self.derived_expression)
            fmt += '\n}'
        return fmt

    __repr__ = repr_from_str


def flatten_expression(
    expr: Expression,
    mappings: Optional[Dict[Union[FunctionApplicationExpression, VariableExpression], Union[Variable, ValueOutputExpression]]] = None,
    ctx: Optional[ExpressionDefinitionContext] = None,
    flatten_cacheable_expression: bool = True,
) -> Union[VariableExpression, ObjectConstantExpression, ValueOutputExpression, VariableAssignmentExpression]:
    """Flatten an expression by replacing certain variables or function applications with sub-expressions.
    The input mapping is a dictionary of {expression: sub-expression}. There are two cases:

    - The expression is a :class:`~concepts.dsl.expression.VariableExpression`, and the sub-expression is a
        :class:`~concepts.dsl.dsl_types.Variable` or a :class:`~concepts.dsl.expression.ValueOutputExpression`. In this case,
        the variable expression will is the sub-expression used for replacing the variable.
    - The expression is a :class:`~concepts.dsl.expression.FunctionApplicationExpression`, and the sub-expression is a
        :class:`~concepts.dsl.dsl_types.Variable`. Here, the function
        application expression must be a "simple" function application expression, i.e., it contains only variables
        as arguments. The Variable will replace the entire function application expression.

    Args:
        expr: the expression to flatten.
        mappings: a dictionary of {expression: sub-expression} to replace the expression with the sub-expression.
        ctx: a :class:`~concepts.dsl.expression.ExpressionDefinitionContext`.
        flatten_cacheable_expression: whether to flatten cacheable expressions. If False, cacheable function applications will be kept as-is.

    Returns:
        the flattened expression.
    """

    if mappings is None:
        mappings = dict()

    if ctx is None:
        ctx = ExpressionDefinitionContext()

    with ctx.as_default():
        return _FlattenExpressionVisitor(ctx, mappings, flatten_cacheable_expression).visit(expr)


class _FlattenExpressionVisitor(IdentityExpressionVisitor):
    def __init__(
        self,
        ctx: ExpressionDefinitionContext,
        mappings: Dict[Union[FunctionApplicationExpression, VariableExpression], Union[Variable, ValueOutputExpression]],
        flatten_cacheable_expression: bool = True,
    ):
        self.ctx = ctx
        self.mappings = mappings
        self.flatten_cacheable_expression = flatten_cacheable_expression

    def visit_variable_expression(self, expr: VariableExpression) -> Union[VariableExpression, ValueOutputExpression]:
        rv = expr
        for k, v in self.mappings.items():
            if isinstance(k, VariableExpression):
                if k.name == expr.name:
                    if isinstance(v, Variable):
                        rv = VariableExpression(v)
                    else:
                        rv = v
                    break
        return rv

    def visit_function_application_expression(self, expr: Union[FunctionApplicationExpression, ListFunctionApplicationExpression]) -> Union[VariableExpression, ValueOutputExpression]:
        # Case 1: the function application will be replaced by something in the mappings.
        for k, v in self.mappings.items():
            if isinstance(k, FunctionApplicationExpression):
                if expr.function.name == k.function.name and all(
                    isinstance(a1, VariableExpression) and isinstance(a2, VariableExpression) and a1.name == a2.name for a1, a2 in zip(expr.arguments, k.arguments)
                ):
                    assert isinstance(v, Variable)
                    return VariableExpression(v)

        # Case 2 contains three sub-cases:
        #   (1) the function is not a derived function
        #   (2) the function corresponds to a state variable
        #   (3) the function is a cacheable function and we do not want to flatten it.
        if not expr.function.is_derived or (isinstance(expr.function, Predicate) and expr.function.is_state_variable) or (not self.flatten_cacheable_expression and expr.function.ftype.is_cacheable):
            return type(expr)(expr.function, [self.visit(e) for e in expr.arguments])

        # Case 3: the function is a derived function and we want to flatten it.
        for arg in expr.function.arguments:
            if not isinstance(arg, Variable):
                raise TypeError(f'Cannot flatten function application {expr} because it contains non-variable arguments.')

        # (1) First resolve the arguments.
        argvs = [self.visit(e) for e in expr.arguments]

        # (2) Make a backup of the current context, and then create a new context using the arguments.
        old_mappings = self.mappings
        self.mappings = dict()
        for arg, argv in zip(expr.function.arguments, argvs):
            if isinstance(arg, Variable):
                self.mappings[VariableExpression(arg)] = argv

        # (3) Flatten the derived expression.
        with self.ctx.with_variables(*expr.function.arguments):
            rv = self.visit(expr.function.derived_expression)

        # (4) Restore the old context.
        self.mappings = old_mappings

        # (5) Flatten the result again, using the old context + mappings.
        return self.visit(rv)
        # return type(rv)(rv.function, [self.visit(e) for e in rv.arguments])

    def visit_list_creation_expression(self, expr: ListCreationExpression) -> ListCreationExpression:
        return type(expr)([self.visit(e) for e in expr.arguments])

    def visit_list_expansion_expression(self, expr: E.ListExpansionExpression) -> ListExpansionExpression:
        return type(expr)(self.visit(expr.expression))

    def visit_list_function_application_expression(self, expr: ListFunctionApplicationExpression) -> Union[VariableExpression, ValueOutputExpression]:
        return self.visit_function_application_expression(expr)

    def visit_bool_expression(self, expr: BoolExpression) -> BoolExpression:
        return BoolExpression(expr.bool_op, [self.visit(child) for child in expr.arguments])

    def visit_object_compare_expression(self, expr: ObjectCompareExpression) -> ObjectCompareExpression:
        return ObjectCompareExpression(expr.compare_op, self.visit(expr.lhs), self.visit(expr.rhs))

    def visit_value_compare_expression(self, expr: ValueCompareExpression) -> ValueCompareExpression:
        return ValueCompareExpression(expr.compare_op, self.visit(expr.lhs), self.visit(expr.rhs))

    def visit_quantification_expression(self, expr: QuantificationExpression) -> QuantificationExpression:
        with self.make_dummy_variable(expr.variable) as dummy_variable:
            return QuantificationExpression(expr.quantification_op, dummy_variable, self.visit(expr.expression))

    def visit_find_all_expression(self, expr: FindAllExpression) -> FindAllExpression:
        with self.make_dummy_variable(expr.variable) as dummy_variable:
            return E.FindAllExpression(dummy_variable, self.visit(expr.expression))

    def visit_predicate_equal_expression(self, expr: PredicateEqualExpression) -> PredicateEqualExpression:
        return type(expr)(self.visit(expr.predicate), self.visit(expr.value))

    def visit_assign_expression(self, expr: AssignExpression) -> AssignExpression:
        return type(expr)(self.visit(expr.predicate), self.visit(expr.value))

    def visit_conditional_select_expression(self, expr: ConditionalSelectExpression) -> ConditionalSelectExpression:
        return type(expr)(self.visit(expr.predicate), self.visit(expr.condition))

    def visit_deictic_select_expression(self, expr: DeicticSelectExpression) -> DeicticSelectExpression:
        with self.make_dummy_variable(expr.variable) as dummy_variable:
            return type(expr)(dummy_variable, self.visit(expr.expression))

    def visit_conditional_assign_expression(self, expr: ConditionalAssignExpression) -> ConditionalAssignExpression:
        return type(expr)(self.visit(expr.predicate), self.visit(expr.value), self.visit(expr.condition))

    def visit_deictic_assign_expression(self, expr: DeicticAssignExpression) -> DeicticAssignExpression:
        with self.make_dummy_variable(expr.variable) as dummy_variable:
            return type(expr)(dummy_variable, self.visit(expr.expression))

    def visit_constant_expression(self, expr: Expression) -> Expression:
        return expr

    def visit_object_constant_expression(self, expr: Expression) -> Expression:
        return expr

    @contextlib.contextmanager
    def make_dummy_variable(self, variable: Variable):
        dummy_variable = self.ctx.gen_random_named_variable(variable.dtype)
        dummy_variable_expr = VariableExpression(variable)

        old_mapping = self.mappings.get(dummy_variable_expr, None)
        self.mappings[dummy_variable_expr] = dummy_variable

        yield dummy_variable

        self.mappings[dummy_variable_expr] = old_mapping


def get_used_state_variables(expr: ValueOutputExpression) -> Set[Predicate]:
    """Return the set of state variables used in the given expression.

    Args:
        expr: the expression to be analyzed.

    Returns:
        the set of state variables (the :class:`Predicate` objects) used in the given expression.
    """

    assert isinstance(expr, ValueOutputExpression), (
        'Only value output expression has well-defined used-state-variables.\n'
        'For value assignment expressions, please separately process the targets, conditions, and values.'
    )

    used_svs = set()

    def dfs(this):
        nonlocal used_svs
        for e in iter_exprs(this):
            if isinstance(e, FunctionApplicationExpression):
                if isinstance(e.function, Predicate) and e.function.is_state_variable:
                    used_svs.add(e.function)
                elif e.function.derived_expression is not None:
                    dfs(e.function.derived_expression)

    dfs(expr)
    return used_svs


def is_simple_bool(expr: Expression) -> bool:
    """Check if the expression is a simple Boolean expression. That is, it is either a Boolean state variable,
    or the negation of a Boolean state variable.

    Args:
        expr: the expression to check.

    Returns:
        True if the expression is a simple boolean expression, False otherwise.
    """

    if isinstance(expr, FunctionApplicationExpression) and isinstance(expr.function, Predicate) and expr.function.is_state_variable:
        return True
    if isinstance(expr, BoolExpression) and expr.bool_op is BoolOpType.NOT:
        return is_simple_bool(expr.arguments[0])
    return False


def split_simple_bool(expr: Expression, initial_negated: bool = False) -> Tuple[Optional[FunctionApplicationExpression], bool]:
    """
    If the expression is a simple Boolean expression (see :func:`is_simple_bool`),
    it returns the feature definition and a boolean indicating whether the expression is negated.

    Args:
        expr (Expression): the expression to be checked.
        initial_negated (bool, optional): whether outer context of the feature expression is a negated function.

    Returns:
        a tuple of the feature application and a boolean indicating whether the expression is negated.
        The first element is None if the feature is not a simple Boolean feature application.
    """
    if isinstance(expr, FunctionApplicationExpression) and isinstance(expr.function, Predicate) and expr.function.is_state_variable:
        return expr, initial_negated
    if isinstance(expr, BoolExpression) and expr.bool_op is BoolOpType.NOT:
        return split_simple_bool(expr.arguments[0], not initial_negated)
    return None, initial_negated


def get_simple_bool_predicate(expr: Expression) -> Predicate:
    """If the expression is a simple bool (see :func:`is_simple_bool`), it returns the underlying predicate.

    Args:
        expr (Expression): the expression, assumed to be a simple Boolean expression.

    Returns:
        the underlying predicate.
    """
    if isinstance(expr, FunctionApplicationExpression) and isinstance(expr.function, Predicate) and expr.function.is_state_variable:
        return expr.function
    assert isinstance(expr, BoolExpression) and expr.bool_op is BoolOpType.NOT
    return get_simple_bool_predicate(expr.arguments[0])
