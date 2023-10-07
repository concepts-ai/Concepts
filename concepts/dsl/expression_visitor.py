#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : expression_visitor.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/30/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""A visitor for iterating over expressions."""

from typing import Any

import concepts.dsl.expression as E
from concepts.dsl.expression import Expression


class ExpressionVisitor(object):
    """A visitor for iterating over expressions."""

    def visit(self, expr: Expression) -> Any:
        """The main entry point of the visitor. It will call the corresponding method for the given expression type.

        Args:
            expr: the expression to visit.

        Returns:
            the result of the visit.
        """
        if isinstance(expr, E.VariableExpression):
            return self.visit_variable_expression(expr)
        elif isinstance(expr, E.ObjectConstantExpression):
            return self.visit_object_constant_expression(expr)
        elif isinstance(expr, E.ConstantExpression):
            return self.visit_constant_expression(expr)
        elif isinstance(expr, E.FunctionApplicationExpression):
            return self.visit_function_application_expression(expr)
        elif isinstance(expr, E.ConditionalSelectExpression):
            return self.visit_conditional_select_expression(expr)
        elif isinstance(expr, E.DeicticSelectExpression):
            return self.visit_deictic_select_expression(expr)
        elif isinstance(expr, E.BoolExpression):
            return self.visit_bool_expression(expr)
        elif isinstance(expr, E.QuantificationExpression):
            return self.visit_quantification_expression(expr)
        elif isinstance(expr, E.GeneralizedQuantificationExpression):
            return self.visit_generalized_quantification_expression(expr)
        elif isinstance(expr, E.PredicateEqualExpression):
            return self.visit_predicate_equal_expression(expr)
        elif isinstance(expr, E.AssignExpression):
            return self.visit_assign_expression(expr)
        elif isinstance(expr, E.ConditionalAssignExpression):
            return self.visit_conditional_assign_expression(expr)
        elif isinstance(expr, E.DeicticAssignExpression):
            return self.visit_deictic_assign_expression(expr)
        else:
            raise TypeError(f'Unknown expression type: {type(expr)}.')

    def visit_variable_expression(self, expr: E.VariableExpression) -> Any:
        raise NotImplementedError()

    def visit_object_constant_expression(self, expr: E.ObjectConstantExpression) -> Any:
        raise NotImplementedError()

    def visit_constant_expression(self, expr: E.ConstantExpression) -> Any:
        raise NotImplementedError()

    def visit_function_application_expression(self, expr: E.FunctionApplicationExpression) -> Any:
        raise NotImplementedError()

    def visit_conditional_select_expression(self, expr: E.ConditionalSelectExpression) -> Any:
        raise NotImplementedError()

    def visit_deictic_select_expression(self, expr: E.DeicticSelectExpression) -> Any:
        raise NotImplementedError()

    def visit_bool_expression(self, expr: E.BoolExpression) -> Any:
        raise NotImplementedError()

    def visit_quantification_expression(self, expr: E.QuantificationExpression) -> Any:
        raise NotImplementedError()

    def visit_generalized_quantification_expression(self, expr: E.GeneralizedQuantificationExpression) -> Any:
        raise NotImplementedError()

    def visit_predicate_equal_expression(self, expr: E.PredicateEqualExpression) -> Any:
        raise NotImplementedError()

    def visit_assign_expression(self, expr: E.AssignExpression) -> Any:
        raise NotImplementedError()

    def visit_conditional_assign_expression(self, expr: E.ConditionalAssignExpression) -> Any:
        raise NotImplementedError()

    def visit_deictic_assign_expression(self, expr: E.DeicticAssignExpression) -> Any:
        raise NotImplementedError()

