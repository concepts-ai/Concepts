#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : csp_parser.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/26/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os.path as osp

from lark import Tree, v_args

from typing import Any

import concepts.dsl.expression as E

from concepts.dm.pdsketch.domain import Domain
from concepts.dsl.constraint import Constraint, EqualityConstraint, NamedConstraintSatisfactionProblem
from concepts.dsl.expression import ExpressionDefinitionContext
from concepts.dsl.expression_visitor import ExpressionVisitor
from concepts.dsl.value import ListValue
from concepts.dsl.tensor_value import TensorValue
from concepts.dm.pdsketch.parsers.pdsketch_parser import PDSketchParser, PDSketchTransformer

inline_args = v_args(inline=True)

__all__ = ['PDSketchCSPParser', 'PDSketchCSPProblemTransformer', 'load_csp_problem_file', 'load_csp_problem_string']


class PDSketchCSPParser(PDSketchParser):
    """Parser for PDSketch domain and problem files. Users should not use this class directly.
    Instead, use the following functions:

    - :func:`load_domain_file`
    - :func:`load_domain_string`
    - :func:`load_csp_problem_file`
    - :func:`load_csp_problem_string`
    """

    grammar_file = osp.join(osp.dirname(__file__), 'pdsketch-v2.grammar')
    """The grammar definition for PDSketch."""

    def make_csp_problem(self, tree: Tree, domain: Domain, ignore_unknown_predicates: bool = False) -> NamedConstraintSatisfactionProblem:
        """Construct a PDSketch problem from a tree."""
        assert tree.children[0].data == 'definition'
        transformer = PDSketchCSPProblemTransformer(domain, ignore_unknown_predicates=ignore_unknown_predicates)
        transformer.transform(tree)
        problem = transformer.problem
        return problem


class PDSketchCSPProblemTransformer(PDSketchTransformer):
    def __init__(self, domain: Domain, ignore_unknown_predicates: bool = False):
        super().__init__(domain, ignore_unknown_predicates=ignore_unknown_predicates)
        self.problem = NamedConstraintSatisfactionProblem()
        self.variables = list()

    @inline_args
    def constants_definition(self, *args):
        for constant in args:
            self.problem.new_constant_var(constant.name, constant.dtype)

    @inline_args
    def variables_definition(self, *args):
        for variable in args:
            self.problem.new_named_actionable_var(variable.name, variable.dtype)
            self.variables.append(variable)

    @inline_args
    def constraints_definition(self, function):
        with ExpressionDefinitionContext(*self.variables, domain=self.domain).as_default():
            function = function.compose()
        visitor = ExpressionToCSPConverter(self.problem, self.domain)
        if E.is_and_expr(function):
            for constraint in function.arguments:
                rv = visitor.visit(constraint)
                self.problem.add_constraint(EqualityConstraint(rv, TensorValue.TRUE))


_parser = PDSketchCSPParser()


def load_csp_problem_file(filename: str, domain: Domain, ignore_unknown_predicates: bool = False) -> NamedConstraintSatisfactionProblem:
    """Load a PDSketch CSP problem from a file."""
    tree = _parser.load(filename)

    with ExpressionDefinitionContext(domain=domain).as_default():
        problem = _parser.make_csp_problem(tree, domain, ignore_unknown_predicates=ignore_unknown_predicates)

    return problem


def load_csp_problem_string(string: str, domain: Domain, ignore_unknown_predicates: bool = False) -> NamedConstraintSatisfactionProblem:
    """Load a PDSketch CSP problem from a string."""
    tree = _parser.loads(string)
    problem = _parser.make_csp_problem(tree, domain, ignore_unknown_predicates=ignore_unknown_predicates)
    return problem


class ExpressionToCSPConverter(ExpressionVisitor):
    def __init__(self, csp: NamedConstraintSatisfactionProblem, domain: Domain):
        self.csp = csp
        self.domain = domain

    def visit_variable_expression(self, expr: E.VariableExpression) -> Any:
        return self.csp.name2optimistic_value[expr.name]

    def visit_object_constant_expression(self, expr: E.ObjectConstantExpression) -> Any:
        return self.csp.name2optimistic_value[expr.name]

    def visit_constant_expression(self, expr: E.ConstantExpression) -> Any:
        return expr.constant

    def visit_function_application_expression(self, expr: E.FunctionApplicationExpression) -> Any:
        arguments = [self.visit(arg) for arg in expr.arguments]
        if expr.function.is_derived:
            raise NotImplementedError()
        rv = self.csp.new_var(expr.function.return_type, wrap=True)
        self.csp.add_constraint(Constraint(expr.function, arguments, rv, note=str(expr)))
        return rv

    def visit_conditional_select_expression(self, expr: E.ConditionalSelectExpression) -> Any:
        raise NotImplementedError()

    def visit_deictic_select_expression(self, expr: E.DeicticSelectExpression) -> Any:
        raise NotImplementedError()

    def visit_bool_expression(self, expr: E.BoolExpression) -> Any:
        arguments = [self.visit(arg) for arg in expr.arguments]
        rv = self.csp.new_var(expr.return_type, wrap=True)
        self.csp.add_constraint(Constraint(expr.bool_op, arguments, rv, note=str(expr)))
        return rv

    def visit_list_creation_expression(self, expr: E.ListCreationExpression) -> Any:
        return ListValue(expr.return_type, [self.visit(arg) for arg in expr.arguments])
