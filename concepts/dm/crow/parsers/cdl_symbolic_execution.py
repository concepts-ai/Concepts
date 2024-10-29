#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : cdl_symbolic_execution.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/15/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from dataclasses import dataclass, field
from typing import Any, Optional, Union, Tuple, List, Dict

import jacinle
from jacinle.utils.enum import JacEnum

import concepts.dsl.expression as E
from concepts.dm.crow.behavior import (
    CrowAchieveExpression, CrowAssertExpression, CrowBehaviorApplicationExpression, CrowBindExpression,
    CrowFeatureAssignmentExpression, CrowRuntimeAssignmentExpression, CrowUntrackExpression
)
from concepts.dm.crow.behavior import CrowBehaviorBodyItem, CrowBehaviorBodyPrimitiveBase, CrowBehaviorBodySuiteBase, CrowBehaviorCommit
from concepts.dm.crow.behavior import CrowBehaviorConditionSuite, CrowBehaviorForeachLoopSuite, CrowBehaviorOrderingSuite, CrowBehaviorWhileLoopSuite
from concepts.dm.crow.controller import CrowControllerApplicationExpression
from concepts.dm.crow.crow_expression_utils import crow_replace_expression_variables
from concepts.dm.crow.crow_generator import CrowGeneratorApplicationExpression
from concepts.dsl.dsl_types import AutoType, UnnamedPlaceholder, Variable


class SymbolicExecutionMode(JacEnum):
    """The execution mode of the symbolic execution engine. Different modes have different supported instructions.

    - FUNCTIONAL: the functional mode, which supports only assignments, function calls, and return statements.
    - DERIVED: the derived mode, which supports only function calls and if-else conditions.
    - BEHAVIOR: the behavior mode corresponds to the "body" of a behavior rule, which supports everything except for return statements and feature assignments.
    - EFFECT: the effect mode corresponds to the "effect" of a behavior rule, which supports only variable / feature assignments and function calls.
    """
    FUNCTIONAL = 'functional'
    DERIVED = 'derived'
    BEHAVIOR = 'behavior'
    EFFECT = 'effect'
    HEURISTIC = 'heuristic'

    @property
    def support_achieve_statements(self) -> bool:
        return self in (SymbolicExecutionMode.BEHAVIOR, SymbolicExecutionMode.HEURISTIC)

    @property
    def support_misc_behavior_body_statements(self):
        return self in (SymbolicExecutionMode.BEHAVIOR, )

    @property
    def support_ordering_statements(self) -> bool:
        return self in (SymbolicExecutionMode.BEHAVIOR, )

    @property
    def support_assign_statements(self) -> bool:
        return self in (SymbolicExecutionMode.EFFECT, SymbolicExecutionMode.HEURISTIC)

    @property
    def support_if_statements(self) -> bool:
        return self in (SymbolicExecutionMode.FUNCTIONAL, SymbolicExecutionMode.DERIVED, SymbolicExecutionMode.BEHAVIOR, SymbolicExecutionMode.EFFECT, SymbolicExecutionMode.HEURISTIC)

    @property
    def support_foreach_statements(self) -> bool:
        return self in (SymbolicExecutionMode.BEHAVIOR, SymbolicExecutionMode.EFFECT, SymbolicExecutionMode.HEURISTIC)

    @property
    def support_while_statements(self) -> bool:
        return self in (SymbolicExecutionMode.BEHAVIOR, SymbolicExecutionMode.HEURISTIC)

    @property
    def support_expr_statements(self) -> bool:
        return self in (SymbolicExecutionMode.DERIVED, )

    @property
    def support_return_statements(self) -> bool:
        return self in (SymbolicExecutionMode.FUNCTIONAL, SymbolicExecutionMode.DERIVED)



@dataclass
class ArgumentsList(object):
    """A list of argument values. They can be variables, function calls, or other expressions."""
    arguments: Tuple[Union['Suite', E.ValueOutputExpression, E.ListExpansionExpression, E.VariableExpression, bool, int, float, complex, str], ...]


@dataclass
class FunctionCall(object):
    """A function call. This is used as the intermediate representation of the parsed expressions.
    Note that this includes not only function calls but also primitive operators and control flow statements.
    """

    name: str
    args: ArgumentsList
    annotations: Optional[Dict[str, Any]] = None

    def __str__(self):
        annotation_str = ''
        if self.annotations is not None:
            annotation_str = f'[[' + ', '.join(f'{k}={v}' for k, v in self.annotations.items()) + ']] '
        arg_strings = [str(arg) for arg in self.args.arguments]
        if sum(len(arg) for arg in arg_strings) > 80:
            arg_strings = [jacinle.indent_text(arg) for arg in arg_strings]
            return f'{annotation_str}{self.name}:\n' + '\n'.join(arg_strings)
        return f'{annotation_str}{self.name}(' + ', '.join(arg_strings) + ')'

    def __repr__(self):
        return f'FunctionCall{{{str(self)}}}'


@dataclass
class Suite(object):
    """A suite of statements. This is used as the intermediate representation of the parsed expressions."""

    items: Tuple[Any, ...]
    local_variables: Dict[str, Any] = field(default_factory=dict)
    tracker: Optional['FunctionCallSymbolicExecutor2'] = None

    def _init_tracker(self, mode: SymbolicExecutionMode):
        self.tracker = FunctionCallSymbolicExecutor2(self, dict(), mode).run()

    def get_effect_statements(self) -> List[Union[CrowFeatureAssignmentExpression, CrowBehaviorForeachLoopSuite, CrowBehaviorConditionSuite]]:
        self._init_tracker(SymbolicExecutionMode.EFFECT)
        return self.tracker.statements

    def get_behavior_body_statements(self) -> List[Any]:
        self._init_tracker(SymbolicExecutionMode.BEHAVIOR)
        return self.tracker.statements

    def get_heuristic_statements(self) -> List[Any]:
        self._init_tracker(SymbolicExecutionMode.HEURISTIC)
        return self.tracker.statements

    def get_derived_expression(self) -> Optional[Union[E.ValueOutputExpression, Tuple[E.ValueOutputExpression, ...]]]:
        self._init_tracker(SymbolicExecutionMode.DERIVED)
        if len(self.tracker.statements) > 0:
            raise ValueError('No body statements are allowed in a derived expression.')
        return self.tracker.return_expression

    def __str__(self):
        if len(self.items) == 0:
            return 'Suite{}'
        if len(self.items) == 1:
            return f'Suite{{{self.items[0]}}}'
        return 'Suite{\n' + '\n'.join(jacinle.indent_text(str(item)) for item in self.items) + '\n}'

    def __repr__(self):
        return self.__str__()


class FunctionCallSymbolicExecutor(object):
    """This class is used to track the function calls and other statements in a suite. It supports simulating the execution of the program
    and generating the post-condition of the program."""

    def __init__(self, suite: Suite, init_local_variables: Optional[Dict[str, Any]] = None):
        """Initialize the function call tracker.

        Args:
            suite: the suite to be tracked.
            init_local_variables: the initial local variables. If None, an empty dictionary will be used.
        """
        self.suite = suite
        self.local_variables = dict() if init_local_variables is None else init_local_variables

        self.assign_expressions = list()
        self.assign_expression_signatures = dict()
        self.expr_expressions = list()
        self.behavior_expressions = list()
        self.return_expression = None

    local_variables_stack: List[Dict[str, Any]]
    """The assignments of local variables."""

    assign_expressions: List[Union[CrowFeatureAssignmentExpression, CrowBehaviorForeachLoopSuite, CrowBehaviorConditionSuite]]
    """A list of assign expressions."""

    assign_expression_signatures: Dict[Tuple[str, ...], E.VariableAssignmentExpression]
    """A dictionary of assign expressions, indexed by their signatures."""

    check_expressions: List[E.ValueOutputExpression]
    """A list of check expressions."""

    expr_expressions: List[E.ValueOutputExpression]
    """A list of expr expressions (i.e. bare expressions in the body)."""

    return_expression: Optional[E.ValueOutputExpression]
    """The return expression. This is either None or a single expression."""

    behavior_expressions: List[CrowBehaviorBodyItem]

    def _g(
        self, expr: Union[E.Expression, UnnamedPlaceholder, CrowBehaviorBodyItem]
    ) -> Union[E.Expression, UnnamedPlaceholder, CrowBehaviorBodyItem]:
        if isinstance(expr, (CrowControllerApplicationExpression, CrowBehaviorApplicationExpression, CrowGeneratorApplicationExpression)):
            return expr
        if isinstance(expr, (CrowBehaviorBodyPrimitiveBase, CrowBehaviorBodySuiteBase)):
            return expr

        if not isinstance(expr, E.Expression):
            raise ValueError(f'Invalid expression: {expr}')
        return crow_replace_expression_variables(expr, {
            E.VariableExpression(Variable(k, AutoType)): v
            for k, v in self.local_variables.items()
        })

    def _get_deictic_signature(self, e, known_deictic_vars=tuple()) -> Optional[Tuple[str, ...]]:
        if isinstance(e, E.DeicticAssignExpression):
            known_deictic_vars = known_deictic_vars + (e.variable.name,)
            return self._get_deictic_signature(e.expression, known_deictic_vars)
        elif isinstance(e, E.AssignExpression):
            args = [x.name if x.name not in known_deictic_vars else '?' for x in e.predicate.arguments]
            return tuple((e.predicate.function.name, *args))
        else:
            return None

    def _mark_assign(self, *exprs: E.VariableAssignmentExpression, annotations: Optional[Dict[str, Any]] = None):
        if annotations is None:
            annotations = dict()
        for expr in exprs:
            signature = self._get_deictic_signature(expr)
            if signature is not None:
                if signature in self.assign_expression_signatures:
                    raise ValueError(f'Duplicate assign expression: {expr} vs {self.assign_expression_signatures[signature]}')
                self.assign_expressions.append((expr, annotations))
                self.assign_expression_signatures[signature] = expr
            else:
                self.assign_expressions.append((expr, annotations))

    @jacinle.log_function(verbose=False)
    def run(self):
        """Simulate the execution of the program and generates an equivalent return statement.
        This function handles if-else conditions. However, loops are not allowed."""

        # jacinle.log_function.print('Current suite:', self.suite)
        from concepts.dm.crow.parsers.cdl_parser import FunctionCall

        current_return_statement = None
        current_return_statement_condition_neg = None

        for item in self.suite.items:
            assert isinstance(item, FunctionCall), f'Invalid item in suite: {item}'

            if item.name == 'assign':
                if isinstance(item.args.arguments[0], E.FunctionApplicationExpression) and item.args.arguments[1] is Ellipsis:
                    # TODO(Jiayuan Mao @ 2024/07/17): implement this for the new CrowFeatureAssignExpression.
                    self.assign_expressions.append(CrowFeatureAssignmentExpression(
                        self._g(item.args.arguments[0]),
                        E.NullExpression(item.args.arguments[0].return_type),
                        **item.annotations if item.annotations is not None else dict()
                    ))
                elif isinstance(item.args.arguments[0], (E.ListFunctionApplicationExpression, E.FunctionApplicationExpression)) and isinstance(item.args.arguments[1], E.ObjectOrValueOutputExpression):
                    self.assign_expressions.append(CrowFeatureAssignmentExpression(
                        self._g(item.args.arguments[0]), self._g(item.args.arguments[1]),
                        **item.annotations if item.annotations is not None else dict()
                    ))
                elif isinstance(item.args.arguments[0], E.VariableExpression) and isinstance(item.args.arguments[1], E.ObjectOrValueOutputExpression):
                    if item.annotations is None or 'symbol' not in item.annotations:  # Runtime assign / assignment to feature variables.
                        if item.args.arguments[0].name not in self.local_variables:
                            self.local_variables[item.args.arguments[0].name] = E.VariableExpression(item.args.arguments[0])
                        self.behavior_expressions.append(CrowRuntimeAssignmentExpression(
                            item.args.arguments[0].variable,
                            self._g(item.args.arguments[1])
                        ))
                    else:
                        self.local_variables[item.args.arguments[0].name] = self._g(item.args.arguments[1])
                else:
                    raise ValueError(f'Invalid assignment: {item}. Types: {type(item.args.arguments[0])}, {type(item.args.arguments[1])}.')
            elif item.name == 'check':
                assert isinstance(item.args.arguments[0], (E.ValueOutputExpression, E.VariableExpression)), f'Invalid check expression: {item.args.arguments[0]}'
                self.check_expressions.append(self._g(item.args.arguments[0]))
            elif item.name == 'expr':
                if isinstance(item.args.arguments[0], (CrowControllerApplicationExpression, CrowBehaviorApplicationExpression, E.ListExpansionExpression)):
                    self.behavior_expressions.append(self._g(item.args.arguments[0]))
                else:
                    assert isinstance(
                        item.args.arguments[0],
                        (E.ValueOutputExpression, E.VariableExpression, CrowGeneratorApplicationExpression)
                    ), f'Invalid expr expression: {item.args.arguments[0]}'
                    self.expr_expressions.append(self._g(item.args.arguments[0]))
            elif item.name == 'bind':
                arguments = item.args.arguments[0].items
                body = item.args.arguments[1]
                self.behavior_expressions.append(CrowBindExpression(arguments, body))
            elif item.name == 'achieve':
                term = item.args.arguments[0]
                self.behavior_expressions.append(CrowAchieveExpression(term, **item.annotations if item.annotations is not None else dict()))
            elif item.name == 'pachieve':
                term = item.args.arguments[0]
                self.behavior_expressions.append(CrowAchieveExpression(term, is_policy_achieve=True, **item.annotations if item.annotations is not None else dict()))
            elif item.name == 'untrack':
                if len(item.args.arguments) == 0:
                    self.behavior_expressions.append(CrowUntrackExpression(E.NullExpression(BOOL)))
                else:
                    term = item.args.arguments[0]
                    self.behavior_expressions.append(CrowUntrackExpression(term))
            elif item.name == 'assert':
                term = item.args.arguments[0]
                self.behavior_expressions.append(CrowAssertExpression(term, **item.annotations if item.annotations is not None else dict()))
            elif item.name == 'commit':
                self.behavior_expressions.append(CrowBehaviorCommit(**item.annotations if item.annotations is not None else dict()))
            elif item.name == 'return':
                assert isinstance(item.args.arguments[0], (E.ValueOutputExpression, E.VariableExpression)), f'Invalid return expression: {item.args.arguments[0]}'
                self.return_expression = _make_conditional_return(current_return_statement, current_return_statement_condition_neg, self._g(item.args.arguments[0]))
                break
            elif item.name == 'ordering':
                suite = item.args.arguments[1]
                tracker = FunctionCallSymbolicExecutor(suite, self.local_variables.copy()).run()
                self.local_variables = tracker.local_variables
                behavior_expressions = tracker.behavior_expressions
                if item.args.arguments[0] == 'promotable unordered':
                    prog = CrowBehaviorOrderingSuite('promotable', (CrowBehaviorOrderingSuite('unordered', behavior_expressions),))
                elif item.args.arguments[0] == 'promotable sequential':
                    prog = CrowBehaviorOrderingSuite('promotable', behavior_expressions)
                elif item.args.arguments[0] == 'critical unordered':
                    prog = CrowBehaviorOrderingSuite('critical', (CrowBehaviorOrderingSuite('unordered', behavior_expressions),))
                elif item.args.arguments[0] == 'critical sequential':
                    prog = CrowBehaviorOrderingSuite('critical', behavior_expressions)
                else:
                    assert ' ' not in item.args.arguments[0], f'Invalid ordering type: {item.args.arguments[0]}'
                    prog = CrowBehaviorOrderingSuite(item.args.arguments[0], behavior_expressions)
                self.behavior_expressions.append(prog)
            elif item.name == 'if':
                condition = self._g(item.args.arguments[0])
                neg_condition = E.NotExpression(condition)
                assert isinstance(condition, E.ValueOutputExpression), f'Invalid condition: {condition}. Type: {type(condition)}.'
                t_suite = item.args.arguments[1]
                f_suite = item.args.arguments[2]

                t_tracker = FunctionCallSymbolicExecutor(t_suite, self.local_variables.copy()).run()
                f_tracker = FunctionCallSymbolicExecutor(f_suite, self.local_variables.copy()).run()

                assert set(t_tracker.local_variables.keys()) == set(f_tracker.local_variables.keys()), f'Local variables in the true and false branches are not consistent: {t_tracker.local_variables.keys()} vs {f_tracker.local_variables.keys()}'
                new_local_variables = t_tracker.local_variables
                for k, v in t_tracker.local_variables.items():
                    if f_tracker.local_variables[k] != v:
                        new_local_variables[k] = E.ConditionExpression(condition, v, f_tracker.local_variables[k])
                self.local_variables = new_local_variables

                if len(t_tracker.assign_expressions) > 0 or len(f_tracker.assign_expressions) > 0:
                    self.assign_expressions.append(CrowBehaviorConditionSuite(condition, t_tracker.assign_expressions, f_tracker.assign_expressions if len(f_tracker.assign_expressions) > 0 else None))

                for expr in t_tracker.check_expressions:
                    self.check_expressions.append(_make_conditional_implies(condition, expr))
                for expr in f_tracker.check_expressions:
                    raise RuntimeError(f'Check statements in the false branch are not supported: {expr}')
                    # TODO(Jiayuan Mao @ 2024/07/17): implement false-branch check statements.
                    self.check_expressions.append(_make_conditional_implies(neg_condition, expr))

                if len(t_tracker.expr_expressions) != len(f_tracker.expr_expressions):
                    raise ValueError(f'Number of bare expressions in the true and false branches are not consistent: {len(t_tracker.expr_expressions)} vs {len(f_tracker.expr_expressions)}')
                if len(t_tracker.expr_expressions) == 0:
                    pass
                elif len(t_tracker.expr_expressions) == 1:
                    self.expr_expressions.append(E.ConditionExpression(condition, t_tracker.expr_expressions[0], f_tracker.expr_expressions[0]))
                else:
                    raise ValueError(f'Multiple bare expressions in the true and false branches are not supported: {t_tracker.expr_expressions} vs {f_tracker.expr_expressions}')

                if len(t_tracker.behavior_expressions) > 0:
                    self.behavior_expressions.append(CrowBehaviorConditionSuite(condition, t_tracker.behavior_expressions, f_tracker.behavior_expressions))

                if t_tracker.return_expression is not None and f_tracker.return_expression is not None:
                    # Both branches have return statements.
                    statement = E.ConditionExpression(condition, t_tracker.return_expression, f_tracker.return_expression)
                    self.return_expression = _make_conditional_return(current_return_statement, current_return_statement_condition_neg, statement)
                    break
                elif t_tracker.return_expression is not None:
                    current_return_statement = _make_conditional_return(current_return_statement, current_return_statement_condition_neg, t_tracker.return_expression)
                    current_return_statement_condition_neg = E.NotExpression(condition)
                elif f_tracker.return_expression is not None:
                    current_return_statement = _make_conditional_return(current_return_statement, current_return_statement_condition_neg, f_tracker.return_expression)
                    current_return_statement_condition_neg = condition
                else:
                    pass
            elif item.name == 'while':
                condition = self._g(item.args.arguments[0])
                suite = item.args.arguments[1]

                tracker = FunctionCallSymbolicExecutor(suite, self.local_variables.copy()).run()

                for k in tracker.local_variables:
                    if k in self.local_variables and self.local_variables[k] != tracker.local_variables[k]:
                        raise ValueError(f'Local variable {k} is assigned in the while statement but has been assigned before: {self.local_variables[k]} vs {tracker.local_variables[k]}')

                for expr in tracker.assign_expressions:
                    raise ValueError(f'Assign statements are not allowed in a while statement: {expr}')

                for expr in tracker.check_expressions:
                    raise ValueError(f'Check statements are not allowed in a while statement: {expr}')

                if len(tracker.expr_expressions) > 0:
                    raise ValueError(f'Expr statements are not allowed in a while statement: {tracker.expr_expressions}')

                if len(tracker.behavior_expressions) > 0:
                    self.behavior_expressions.append(CrowBehaviorWhileLoopSuite(condition, tracker.behavior_expressions, **item.annotations if item.annotations is not None else dict()))

                if tracker.return_expression is not None:
                    raise ValueError(f'Return statement is not allowed in a while statement: {tracker.return_expression}')
            elif item.name == 'foreach':
                suite = item.args.arguments[1]
                tracker = FunctionCallSymbolicExecutor(suite, self.local_variables.copy()).run()

                for k in tracker.local_variables:
                    if k in self.local_variables and self.local_variables[k] != tracker.local_variables[k]:
                        raise ValueError(f'Local variable {k} is assigned in the foreach statement but has been assigned before: {self.local_variables[k]} vs {tracker.local_variables[k]}')

                statements = tracker.assign_expressions
                if len(statements) > 0:
                    for var in item.args.arguments[0].items:
                        statements = [CrowBehaviorForeachLoopSuite(var, statements)]
                    self.assign_expressions.extend(statements)

                for expr in tracker.check_expressions:
                    for var in item.args.arguments[0].items:
                        expr = E.ForallExpression(var, expr)
                    self.check_expressions.append(expr)

                if len(tracker.expr_expressions) == 0:
                    pass
                else:
                    if len(tracker.expr_expressions) == 1:
                        merged = tracker.expr_expressions[0]
                    else:
                        merged = E.AndExpression(*tracker.expr_expressions)
                    for var in item.args.arguments[0].items:
                        merged = E.ForallExpression(var, merged)
                    self.expr_expressions.append(merged)

                if len(tracker.behavior_expressions) > 0:
                    assert len(item.args.arguments[0].items) == 1, f'Invalid number of variables in the foreach statement: {item.args.arguments[0].items}. Currently only one variable is supported.'
                    self.behavior_expressions.append(CrowBehaviorForeachLoopSuite(item.args.arguments[0].items[0], tracker.behavior_expressions))

                if tracker.return_expression is not None:
                    raise ValueError(f'Return statement is not allowed in a foreach statement: {tracker.return_expression}')
            elif item.name == 'foreach_in':
                variables = item.args.arguments[0].items
                values = item.args.arguments[1].items
                suite = item.args.arguments[2]

                if len(variables) != 1 or len(values) != 1:
                    raise NotImplementedError(f'Currently only one variable and one value are supported in a foreach_in statement: {variables} vs {values}')

                tracker = FunctionCallSymbolicExecutor(suite, self.local_variables.copy()).run()
                for k in tracker.local_variables:
                    if k in self.local_variables and self.local_variables[k] != tracker.local_variables[k]:
                        raise ValueError(f'Local variable {k} is assigned in the foreach_in statement but has been assigned before: {self.local_variables[k]} vs {tracker.local_variables[k]}')

                statements = tracker.assign_expressions
                if len(statements) > 0:
                    for var, value in reversed(list(zip(variables, values))):
                        statements = [CrowBehaviorForeachLoopSuite(var, statements, loop_in_expression=value)]
                    self.assign_expressions.extend(statements)

                if len(tracker.check_expressions) > 0:
                    raise NotImplementedError(f'Check statements are not allowed in a foreach_in statement: {tracker.check_expressions}')

                if len(tracker.expr_expressions) > 0:
                    raise NotImplementedError(f'Expr statements are not allowed in a foreach_in statement: {tracker.expr_expressions}')

                if len(tracker.behavior_expressions) > 0:
                    # TODO(Jiayuan Mao @ 2024/03/12): implement the rest parts of action statements.
                    expressions = tracker.behavior_expressions
                    for var, value in reversed(list(zip(variables, values))):
                        expressions = [CrowBehaviorForeachLoopSuite(var, expressions, loop_in_expression=value)]
                    self.behavior_expressions.extend(expressions)

                if tracker.return_expression is not None:
                    raise ValueError(f'Return statement is not allowed in a foreach_in statement: {tracker.return_expression}')
            elif item.name == 'pass':
                pass

        # jacinle.log_function.print('Local variables:', self.local_variables)
        # jacinle.log_function.print('Assign expressions:', self.assign_expressions)
        # jacinle.log_function.print('Check expressions:', self.check_expressions)
        # jacinle.log_function.print('Expr expressions:', self.expr_expressions)
        # jacinle.log_function.print('Return expression:', self.return_expression)
        return self


def _make_conditional_implies(condition: E.ValueOutputExpression, test: E.ValueOutputExpression):
    if isinstance(test, E.BoolExpression) and test.op == E.BoolOpType.IMPLIES:
        if isinstance(test.arguments[0], E.BoolExpression) and test.arguments[0].op == E.BoolOpType.AND:
            return E.ImpliesExpression(E.AndExpression(condition, *test.arguments[0].arguments), test.arguments[1])
        else:
            return E.ImpliesExpression(E.AndExpression(condition, test.arguments[0]), test.arguments[1])
    else:
        return E.ImpliesExpression(condition, test)


def _make_conditional_return(current_stmt: Optional[E.ValueOutputExpression], current_condition_neg: Optional[E.ValueOutputExpression], new_stmt: E.ValueOutputExpression):
    if current_stmt is None:
        return new_stmt
    return E.ConditionExpression(current_condition_neg, new_stmt, current_stmt)


def _make_conditional_assign(assign_stmt: E.VariableAssignmentExpression, condition: E.ValueOutputExpression):
    if isinstance(assign_stmt, E.AssignExpression):
        return E.ConditionalAssignExpression(assign_stmt.predicate, assign_stmt.value, condition)
    elif isinstance(assign_stmt, E.ConditionalAssignExpression):
        if isinstance(assign_stmt.condition, E.BoolExpression) and assign_stmt.condition.op == E.BoolOpType.AND:
            return E.ConditionalAssignExpression(assign_stmt.predicate, assign_stmt.value, E.AndExpression(condition, *assign_stmt.condition.arguments))
        else:
            return E.ConditionalAssignExpression(assign_stmt.predicate, assign_stmt.value, E.AndExpression(condition, assign_stmt.condition))
    elif isinstance(assign_stmt, E.DeicticAssignExpression):
        return E.DeicticAssignExpression(assign_stmt.variable, _make_conditional_assign(assign_stmt.expression, condition))
    else:
        raise ValueError(f'Invalid assign statement: {assign_stmt}')


class FunctionCallSymbolicExecutor2(object):
    """This class is used to track the function calls and other statements in a suite. It supports simulating the execution of the program
    and generating the post-condition of the program."""

    def __init__(self, suite: Suite, init_local_variables: Optional[Dict[str, Any]] = None, mode: SymbolicExecutionMode = SymbolicExecutionMode.FUNCTIONAL):
        """Initialize the function call tracker.

        Args:
            suite: the suite to be tracked.
            init_local_variables: the initial local variables. If None, an empty dictionary will be used.
        """
        self.suite = suite
        self.local_variables = dict() if init_local_variables is None else init_local_variables
        self.mode = mode

        self.statements = list()
        self.return_expression = None

    def fork(self, suite: Suite, init_local_variables: Optional[Dict[str, Any]] = None) -> 'FunctionCallSymbolicExecutor2':
        """Make a "fork" of the current tracker.

        Args:
            suite: the new suite to be tracked.
            init_local_variables: the new initial local variables. If None, the current local variables will be used.

        Returns:
            A new instance of the tracker.
        """
        return FunctionCallSymbolicExecutor2(suite, init_local_variables if init_local_variables is not None else self.local_variables.copy(), mode=self.mode)

    suite: Suite
    """The suite to be tracked."""

    local_variables_stack: List[Dict[str, Any]]
    """The assignments of local variables."""

    mode: SymbolicExecutionMode

    statements: List[Union[
        CrowFeatureAssignmentExpression, CrowRuntimeAssignmentExpression, CrowControllerApplicationExpression,
        CrowBehaviorOrderingSuite, CrowBehaviorWhileLoopSuite, CrowBehaviorForeachLoopSuite, CrowBehaviorConditionSuite
    ]]

    return_expression: Optional[E.ValueOutputExpression]
    """The return expression. This is either None or a single expression."""

    def _replace_variables(
        self, expr: Union[E.Expression, UnnamedPlaceholder, CrowBehaviorBodyItem]
    ) -> Union[E.Expression, UnnamedPlaceholder, CrowBehaviorBodyItem]:
        if isinstance(expr, (CrowControllerApplicationExpression, CrowBehaviorApplicationExpression, CrowGeneratorApplicationExpression)):
            return expr
        if isinstance(expr, (CrowBehaviorBodyPrimitiveBase, CrowBehaviorBodySuiteBase)):
            return expr

        if not isinstance(expr, E.Expression):
            raise ValueError(f'Invalid expression: {expr}')
        return crow_replace_expression_variables(expr, {
            E.VariableExpression(Variable(k, AutoType)): v
            for k, v in self.local_variables.items()
        })

    def set_unique_return(self, return_expression: E.ValueOutputExpression):
        """Set the return expression. This function is used to set the return expression when it is unique."""
        assert self.return_expression is None, f'The return expression has already been set: {self.return_expression}'
        self.return_expression = return_expression

    @jacinle.log_function(verbose=False)
    def run(self):
        """Simulate the execution of the program and generates an equivalent return statement.
        This function handles if-else conditions. However, loops are not allowed."""

        # jacinle.log_function.print('Current suite:', self.suite)
        from concepts.dm.crow.parsers.cdl_parser import FunctionCall

        current_return_statement = None
        current_return_statement_condition_neg = None

        for item in self.suite.items:
            assert isinstance(item, FunctionCall), f'Invalid item in suite: {item}'
            annotations = item.annotations if item.annotations is not None else dict()
            if item.name == 'assign':
                target, value = item.args.arguments
                if isinstance(target, E.FunctionApplicationExpression) and value is Ellipsis:
                    if not self.mode.support_assign_statements:
                        raise ValueError(f'Feature assign statements are not allowed in the current mode: {self.mode}')
                    self.statements.append(CrowFeatureAssignmentExpression(target, E.NullExpression(target.return_type), **annotations))
                elif isinstance(target, E.FunctionApplicationExpression) and isinstance(value, E.ObjectOrValueOutputExpression):
                    if not self.mode.support_assign_statements:
                        raise ValueError(f'Feature assign statements are not allowed in the current mode: {self.mode}')
                    self.statements.append(CrowFeatureAssignmentExpression(target, value, **annotations))
                elif isinstance(target, E.VariableExpression) and isinstance(value, E.ObjectOrValueOutputExpression):
                    use_symbol_definition = 'symbol' in item.annotations
                    if self.mode is SymbolicExecutionMode.DERIVED:
                        use_symbol_definition = True

                    if use_symbol_definition:
                        self.local_variables[target.name] = self._replace_variables(item.args.arguments[1])
                    else:  # Runtime assign / assignment to feature variables.
                        if target.name not in self.local_variables:  # Only put a VariableExpression in the local variables.
                            self.local_variables[target.name] = target
                        self.statements.append(CrowRuntimeAssignmentExpression(target.variable, value))
                else:
                    raise ValueError(f'Invalid assignment: {item}. Types: {type(target)}, {type(item.args.arguments[1])}.')
            elif item.name == 'expr':
                if isinstance(item.args.arguments[0], (CrowControllerApplicationExpression, CrowBehaviorApplicationExpression, E.ListExpansionExpression)):
                    if not self.mode.support_misc_behavior_body_statements:
                        raise ValueError(f'Behavior body statements are not allowed in the current mode: {self.mode}')
                    self.statements.append(self._replace_variables(item.args.arguments[0]))
                else:
                    if not self.mode.support_expr_statements:
                        raise ValueError(f'Expr statements are not allowed in the current mode: {self.mode}')
                    assert isinstance(
                        item.args.arguments[0],
                        (E.ValueOutputExpression, E.VariableExpression, CrowGeneratorApplicationExpression)
                    ), f'Invalid expr expression: {item.args.arguments[0]}'
                    self.set_unique_return(self._replace_variables(item.args.arguments[0]))
            elif item.name in ('bind', 'untrack', 'assert', 'commit'):
                if not self.mode.support_misc_behavior_body_statements:
                    raise ValueError(f'Behavior body statement {item} are not allowed in the current mode: {self.mode}')
                if item.name == 'bind':
                    arguments = item.args.arguments[0].items
                    body = item.args.arguments[1]
                    self.statements.append(CrowBindExpression(arguments, body))
                elif item.name == 'untrack':
                    term = item.args.arguments[0]
                    self.statements.append(CrowUntrackExpression(term))
                elif item.name == 'assert':
                    term = item.args.arguments[0]
                    self.statements.append(CrowAssertExpression(term, **annotations))
                elif item.name == 'commit':
                    self.statements.append(CrowBehaviorCommit(**annotations))
            elif item.name in ('achieve', 'pachieve'):
                if not self.mode.support_achieve_statements:
                    raise ValueError(f'Achieve statements are not allowed in the current mode: {self.mode}')
                if item.name == 'achieve':
                    term = item.args.arguments[0]
                    self.statements.append(CrowAchieveExpression(term, **annotations))
                elif item.name == 'pachieve':
                    term = item.args.arguments[0]
                    self.statements.append(CrowAchieveExpression(term, is_policy_achieve=True, **annotations))
            elif item.name == 'return':
                if not self.mode.support_return_statements:
                    raise ValueError(f'Return statements are not allowed in the current mode: {self.mode}')
                assert isinstance(item.args.arguments[0], (E.ValueOutputExpression, E.VariableExpression)), f'Invalid return expression: {item.args.arguments[0]}'
                self.return_expression = _make_conditional_return(current_return_statement, current_return_statement_condition_neg, self._replace_variables(item.args.arguments[0]))
                break
            elif item.name == 'ordering':
                if not self.mode.support_ordering_statements:
                    raise ValueError(f'Ordering statements are not allowed in the current mode: {self.mode}')
                suite = item.args.arguments[1]
                tracker = self.fork(suite).run()
                self.local_variables = tracker.local_variables
                statements = tracker.statements
                if item.args.arguments[0] == 'promotable unordered':
                    prog = CrowBehaviorOrderingSuite('promotable', (CrowBehaviorOrderingSuite('unordered', statements),))
                elif item.args.arguments[0] == 'promotable sequential':
                    prog = CrowBehaviorOrderingSuite('promotable', statements)
                elif item.args.arguments[0] == 'critical unordered':
                    prog = CrowBehaviorOrderingSuite('critical', (CrowBehaviorOrderingSuite('unordered', statements),))
                elif item.args.arguments[0] == 'critical sequential':
                    prog = CrowBehaviorOrderingSuite('critical', statements)
                else:
                    assert ' ' not in item.args.arguments[0], f'Invalid ordering type: {item.args.arguments[0]}'
                    prog = CrowBehaviorOrderingSuite(item.args.arguments[0], statements)
                self.statements.append(prog)
            elif item.name == 'if':
                if not self.mode.support_if_statements:
                    raise ValueError(f'If statements are not allowed in the current mode: {self.mode}')

                condition = self._replace_variables(item.args.arguments[0])
                assert isinstance(condition, E.ValueOutputExpression), f'Invalid condition: {condition}. Type: {type(condition)}.'
                t_suite = item.args.arguments[1]
                f_suite = item.args.arguments[2]

                t_tracker = self.fork(t_suite).run()
                f_tracker = self.fork(f_suite).run()

                assert set(t_tracker.local_variables.keys()) == set(f_tracker.local_variables.keys()), f'Local variables in the true and false branches are not consistent: {t_tracker.local_variables.keys()} vs {f_tracker.local_variables.keys()}'
                new_local_variables = t_tracker.local_variables
                for k, v in t_tracker.local_variables.items():
                    if f_tracker.local_variables[k] != v:
                        new_local_variables[k] = E.ConditionExpression(condition, v, f_tracker.local_variables[k])
                self.local_variables = new_local_variables

                if len(t_tracker.statements) > 0:
                    self.statements.append(CrowBehaviorConditionSuite(condition, t_tracker.statements, f_tracker.statements))

                if t_tracker.return_expression is not None and f_tracker.return_expression is not None:
                    # Both branches have return statements.
                    statement = E.ConditionExpression(condition, t_tracker.return_expression, f_tracker.return_expression)
                    self.return_expression = _make_conditional_return(current_return_statement, current_return_statement_condition_neg, statement)
                    break
                elif t_tracker.return_expression is not None:
                    current_return_statement = _make_conditional_return(current_return_statement, current_return_statement_condition_neg, t_tracker.return_expression)
                    current_return_statement_condition_neg = E.NotExpression(condition)
                elif f_tracker.return_expression is not None:
                    current_return_statement = _make_conditional_return(current_return_statement, current_return_statement_condition_neg, f_tracker.return_expression)
                    current_return_statement_condition_neg = condition
            elif item.name == 'while':
                if not self.mode.support_while_statements:
                    raise ValueError(f'While statements are not allowed in the current mode: {self.mode}')

                condition = self._replace_variables(item.args.arguments[0])
                suite = item.args.arguments[1]

                tracker = self.fork(suite).run()
                for k in tracker.local_variables:
                    if k in self.local_variables and self.local_variables[k] != tracker.local_variables[k]:
                        raise ValueError(f'Local variable {k} is assigned in the while statement but has been assigned before: {self.local_variables[k]} vs {tracker.local_variables[k]}')

                if len(tracker.statements) > 0:
                    self.statements.append(CrowBehaviorWhileLoopSuite(condition, tracker.statements, **annotations))
                if tracker.return_expression is not None:
                    raise ValueError(f'Return statement is not allowed in a while statement: {tracker.return_expression}')
            elif item.name == 'foreach':
                if not self.mode.support_foreach_statements:
                    raise ValueError(f'Foreach statements are not allowed in the current mode: {self.mode}')

                variables = item.args.arguments[0].items
                suite = item.args.arguments[1]
                tracker = self.fork(suite).run()
                for k in tracker.local_variables:
                    if k in self.local_variables and self.local_variables[k] != tracker.local_variables[k]:
                        raise ValueError(f'Local variable {k} is assigned in the foreach statement but has been assigned before: {self.local_variables[k]} vs {tracker.local_variables[k]}')

                statements = tracker.statements
                if len(statements) > 0:
                    for var in variables:
                        statements = [CrowBehaviorForeachLoopSuite(var, statements)]
                    self.statements.extend(statements)
                if tracker.return_expression is not None:
                    raise ValueError(f'Return statement is not allowed in a foreach statement: {tracker.return_expression}')
            elif item.name == 'foreach_in':
                if not self.mode.support_foreach_statements:
                    raise ValueError(f'Foreach_in statements are not allowed in the current mode: {self.mode}')

                variables = item.args.arguments[0].items
                values = item.args.arguments[1].items
                suite = item.args.arguments[2]
                if len(variables) != 1 or len(values) != 1:
                    raise NotImplementedError(f'Currently only one variable and one value are supported in a foreach_in statement: {variables} vs {values}')

                tracker = self.fork(suite).run()
                for k in tracker.local_variables:
                    if k in self.local_variables and self.local_variables[k] != tracker.local_variables[k]:
                        raise ValueError(f'Local variable {k} is assigned in the foreach_in statement but has been assigned before: {self.local_variables[k]} vs {tracker.local_variables[k]}')

                statements = tracker.statements
                if len(statements) > 0:
                    for var, value in reversed(list(zip(variables, values))):
                        statements = [CrowBehaviorForeachLoopSuite(var, statements, loop_in_expression=value)]
                    self.statements.extend(statements)
                if tracker.return_expression is not None:
                    raise ValueError(f'Return statement is not allowed in a foreach_in statement: {tracker.return_expression}')
            elif item.name == 'pass':
                pass
            else:
                raise NotImplementedError(f'Unsupported item: {item}')

        # jacinle.log_function.print('Local variables:', self.local_variables)
        # jacinle.log_function.print('Assign expressions:', self.assign_expressions)
        # jacinle.log_function.print('Check expressions:', self.check_expressions)
        # jacinle.log_function.print('Expr expressions:', self.expr_expressions)
        # jacinle.log_function.print('Return expression:', self.return_expression)
        return self
