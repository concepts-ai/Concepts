#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : regression_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/21/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Optional, Union, Sequence, Tuple, List

import concepts.dsl.expression as E
from concepts.dsl.dsl_types import Variable
from concepts.dsl.expression import ObjectOrValueOutputExpression
from concepts.dsl.tensor_value import TensorValue

from concepts.dm.crow.behavior import CrowBehavior
from concepts.dm.crow.behavior import CrowUntrackExpression, CrowBindExpression, CrowAssertExpression
from concepts.dm.crow.behavior import CrowRuntimeAssignmentExpression, CrowFeatureAssignmentExpression
from concepts.dm.crow.behavior import CrowBehaviorOrderingSuite
from concepts.dm.crow.behavior_utils import format_behavior_statement
from concepts.dm.crow.controller import CrowControllerApplicationExpression
from concepts.dm.crow.planners.regression_planning import ScopedCrowExpression, SupportedCrowExpressionType

__all__ = ['replace_variable_with_value', 'format_regression_statement', 'canonize_bounded_variables', 'split_simple_sequential']


def replace_variable_with_value(expr: ObjectOrValueOutputExpression, scope) -> ObjectOrValueOutputExpression:
    """Replace the variables in the expression with the values in the scope.

    Args:
        expr: the expression to edit.
        scope: the scope containing the variable-value pairs.

    Returns:
        the expression with all variables replaced.
    """
    if isinstance(expr, E.VariableExpression):
        if isinstance(scope[expr.variable.name], TensorValue):
            return E.ConstantExpression(scope[expr.variable.name])
        return E.ObjectConstantExpression(scope[expr.variable.name])
    elif isinstance(expr, E.FunctionApplicationExpression):
        return E.FunctionApplicationExpression(expr.function, [replace_variable_with_value(x, scope) for x in expr.arguments])
    else:
        return expr


def format_regression_statement(stmt: ScopedCrowExpression, scopes: Optional[dict]) -> str:
    """Format a regression statement into a human-readable string.

    Args:
        stmt: the regression statement.
        scopes: the scopes of the regression statement.

    Returns:
        the formatted string.
    """
    assert isinstance(stmt.statement, SupportedCrowExpressionType)
    inner_stmt = stmt.statement
    scope_id = stmt.scope_id
    return format_behavior_statement(inner_stmt, scopes=scopes, scope_id=scope_id) + '@' + str(scope_id)


def canonize_bounded_variables(scopes, scope_id):
    """Canonize the bounded variables in the scope. This function is used to resolve the scope references in the regression trace.

    Args:
        scopes: a set of scopes of the regression trace.
        scope_id: the scope id to resolve.

    Returns:
        the resolved scope in `scopes[scope_id]`.
    """
    scope = scopes[scope_id].copy()
    for var, value in scope.items():
        if isinstance(value, Variable):
            assert value.scope > -1
            for i in range(100):
                if isinstance(value, Variable):
                    value = scopes[value.scope][value.name]
                else:
                    break
            else:
                raise RuntimeError('Too deep scope reference.')
            scope[var] = value
    return scope


def split_simple_sequential(program: Sequence[Union[CrowBehavior, CrowBehaviorOrderingSuite]], scope_id: int) -> Tuple[
    Sequence[Union[CrowBehavior, CrowBehaviorOrderingSuite]],
    List[ScopedCrowExpression]
]:
    """Split a program into a complex part and a simple part. The simple part is a list of statements that can be executed sequentially.
    This function extracts a suffix of the program containing only :class:`~concepts.dm.crow.behavior.CrowAssertExpression`,
    :class:`~concepts.dm.crow.behavior.CrowControllerApplicationExpression`, :class:`~concepts.dm.crow.behavior.CrowBindExpression`,
    :class:`~concepts.dm.crow.behavior.CrowUntrackExpression`, :class:`~concepts.dm.crow.behavior.CrowRuntimeAssignmentExpression`,
    and :class:`~concepts.dm.crow.behavior.CrowFeatureAssignmentExpression`.

    Args:
        program: the program to split.
        scope_id: the scope id of the program.

    Returns:
        the complex part and the simple part.
    """
    simple_part = list()
    for i in reversed(range(len(program))):
        if isinstance(program[i], (CrowAssertExpression, CrowControllerApplicationExpression, CrowBindExpression, CrowUntrackExpression, CrowRuntimeAssignmentExpression, CrowFeatureAssignmentExpression)):
            simple_part.append(ScopedCrowExpression(program[i], scope_id))
        else:
            break
    complex_part = program[:len(program) - len(simple_part)]
    return complex_part, list(reversed(simple_part))


