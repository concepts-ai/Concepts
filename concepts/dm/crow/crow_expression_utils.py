#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : crow_expression_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/17/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import numpy as np
import torch

from typing import Optional, Any, Union, Sequence, Tuple, Set, Dict
from jacinle.utils.meta import stmap

from concepts.dsl.dsl_types import Variable
from concepts.dsl.expression import Expression, ExpressionDefinitionContext
from concepts.dsl.expression import FunctionApplicationExpression, VariableExpression, ObjectOrValueOutputExpression, VariableAssignmentExpression, ValueOutputExpression
from concepts.dsl.expression import ListFunctionApplicationExpression, BoolExpression, BoolOpType
from concepts.dsl.expression_utils import iter_exprs, FlattenExpressionVisitor
from concepts.dsl.tensor_value import TensorValue
from concepts.dsl.tensor_state import StateObjectReference

from concepts.dm.crow.crow_function import CrowFeature
from concepts.dm.crow.controller import CrowControllerApplier

__all__ = [
    'make_plan_serializable',
    'crow_flatten_expression', 'crow_replace_expression_variables',
    'crow_get_used_state_variables', 'crow_is_simple_bool', 'crow_split_simple_bool',
    'crow_get_simple_bool_predicate',
]


def make_plan_serializable(plan: Sequence[CrowControllerApplier], json_compatible: bool = False) -> Tuple[Union[Dict[str, Any], str, list]]:
    """Make a serializable version of the plan.

    Args:
        plan: the plan to be serialized.
        json_compatible: whether to make the plan JSON-compatible. If True, the plan will be converted to a JSON-compatible format, which will flatten all numpy arrays and torch tensors to lists.

    Returns:
        the serialized plan.
    """
    def prim(v, json_compatible=json_compatible):
        if isinstance(v, CrowControllerApplier):
            return {'name': v.name, 'arguments': stmap(prim, v.arguments)}
        elif isinstance(v, StateObjectReference):
            return v.name
        elif isinstance(v, TensorValue):
            return prim(v.tensor)
        else:
            if json_compatible:
                if isinstance(v, np.ndarray):
                    return v.tolist()
                elif isinstance(v, torch.Tensor):
                    return v.cpu().numpy().tolist()
                elif hasattr(v, '__dict__'):
                    return {'class': v.__class__.__name__, 'data': stmap(prim, v.__dict__)}
                else:
                    return v
            return v

    return tuple(prim(v) for v in plan)


def crow_replace_expression_variables(
    expr: Expression,
    mappings: Optional[Dict[Union[FunctionApplicationExpression, VariableExpression], Union[Variable, ObjectOrValueOutputExpression]]] = None,
    ctx: Optional[ExpressionDefinitionContext] = None,
) -> Union[ObjectOrValueOutputExpression, VariableAssignmentExpression]:
    """Replace variables in an expression with other expressions. Allowed replacements are:

    - Replace a :class:`~concepts.dsl.expression.VariableExpression` with a :class:`~concepts.dsl.dsl_types.Variable` or a :class:`~concepts.dsl.expression.ValueOutputExpression`.
    - Replace a :class:`~concepts.dsl.expression.FunctionApplicationExpression` with a :class:`~concepts.dsl.dsl_types.Variable` or a :class:`~concepts.dsl.expression.ValueOutputExpression`.

    Args:
        expr: the expression to replace variables.
        mappings: a dictionary of {expression: sub-expression} to replace the expression with the sub-expression.
        ctx: a :class:`~concepts.dsl.expression.ExpressionDefinitionContext`.
    """
    return crow_flatten_expression(expr, mappings, ctx, deep=False, flatten_cacheable_expression=False)


def crow_flatten_expression(
    expr: Expression,
    mappings: Optional[Dict[Union[FunctionApplicationExpression, VariableExpression], Union[Variable, ObjectOrValueOutputExpression]]] = None,
    ctx: Optional[ExpressionDefinitionContext] = None,
    deep: bool = True,
    flatten_cacheable_expression: bool = True,
) -> Union[ObjectOrValueOutputExpression, VariableAssignmentExpression]:
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
        deep: whether to recursively flatten the expression (expand derived functions). Default is True.
        flatten_cacheable_expression: whether to flatten cacheable expressions. If False, cacheable function applications will be kept as-is.

    Returns:
        the flattened expression.
    """

    if mappings is None:
        mappings = dict()

    if ctx is None:
        ctx = ExpressionDefinitionContext()

    with ctx.as_default():
        return CrowFlattenExpressionVisitor(ctx, mappings, deep=deep, flatten_cacheable_expression=flatten_cacheable_expression).visit(expr)


class CrowFlattenExpressionVisitor(FlattenExpressionVisitor):
    def __init__(
        self,
        ctx: ExpressionDefinitionContext,
        mappings: Dict[Union[FunctionApplicationExpression, VariableExpression], Union[Variable, ValueOutputExpression]],
        deep: bool = True,
        flatten_cacheable_expression: bool = True,
    ):
        super().__init__(ctx, mappings, deep)
        self.flatten_cacheable_expression = flatten_cacheable_expression

    def visit_function_application_expression(self, expr: Union[FunctionApplicationExpression, ListFunctionApplicationExpression]) -> Union[VariableExpression, ValueOutputExpression]:
        # Case 1: the function application will be replaced by something in the mappings.
        for k, v in self.mappings.items():
            if isinstance(k, FunctionApplicationExpression):
                if expr.function.name == k.function.name and all(
                    isinstance(a1, VariableExpression) and isinstance(a2, VariableExpression) and a1.name == a2.name for a1, a2 in zip(expr.arguments, k.arguments)
                ):
                    assert isinstance(v, Variable)
                    return VariableExpression(v)

        if not self.deep:
            return type(expr)(expr.function, [self.visit(e) for e in expr.arguments])

        # Case 2 contains three sub-cases:
        #   (1) the function is not a derived function
        #   (2) the function corresponds to a state variable
        #   (3) the function is a cacheable function and we do not want to flatten it.
        if not expr.function.is_derived or (isinstance(expr.function, CrowFeature) and expr.function.is_state_variable) or (not self.flatten_cacheable_expression and expr.function.ftype.is_cacheable):
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


def crow_get_used_state_variables(expr: ValueOutputExpression) -> Set[CrowFeature]:
    """Return the set of state variables used in the given expression.

    Args:
        expr: the expression to be analyzed.

    Returns:
        the set of state variables (the :class:`~concepts.dm.crow.function.Feature` objects) used in the given expression.
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
                if isinstance(e.function, CrowFeature) and e.function.is_state_variable:
                    used_svs.add(e.function)
                elif e.function.derived_expression is not None:
                    dfs(e.function.derived_expression)

    dfs(expr)
    return used_svs


def crow_is_simple_bool(expr: Expression) -> bool:
    """Check if the expression is a simple Boolean expression. That is, it is either a Boolean state variable,
    or the negation of a Boolean state variable.

    Args:
        expr: the expression to check.

    Returns:
        True if the expression is a simple boolean expression, False otherwise.
    """
    if isinstance(expr, FunctionApplicationExpression) and isinstance(expr.function, CrowFeature) and expr.function.is_state_variable:
        return True
    if isinstance(expr, BoolExpression) and expr.bool_op is BoolOpType.NOT:
        return crow_is_simple_bool(expr.arguments[0])
    return False


def crow_split_simple_bool(expr: Expression, initial_negated: bool = False) -> Tuple[Optional[FunctionApplicationExpression], bool]:
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
    if isinstance(expr, FunctionApplicationExpression) and isinstance(expr.function, CrowFeature) and expr.function.is_state_variable:
        return expr, initial_negated
    if isinstance(expr, BoolExpression) and expr.bool_op is BoolOpType.NOT:
        return crow_split_simple_bool(expr.arguments[0], not initial_negated)
    return None, initial_negated


def crow_get_simple_bool_predicate(expr: Expression) -> CrowFeature:
    """If the expression is a simple bool (see :func:`is_simple_bool`), it returns the underlying predicate.

    Args:
        expr: the expression, assumed to be a simple Boolean expression.

    Returns:
        the underlying predicate.
    """
    if isinstance(expr, FunctionApplicationExpression) and isinstance(expr.function, CrowFeature) and expr.function.is_state_variable:
        return expr.function
    assert isinstance(expr, BoolExpression) and expr.bool_op is BoolOpType.NOT
    return crow_get_simple_bool_predicate(expr.arguments[0])

