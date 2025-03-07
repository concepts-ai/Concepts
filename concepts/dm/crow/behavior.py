#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : behavior.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/16/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import warnings
from typing import Optional, Union, Iterator, Sequence, Tuple, Dict

from jacinle.utils.enum import JacEnum
from jacinle.utils.printing import indent_text
from jacinle.utils.meta import repr_from_str

from concepts.dsl.dsl_types import ObjectType, ValueType, Variable, ObjectConstant, BOOL
from concepts.dsl.expression import NullExpression, ObjectOrValueOutputExpression, ValueOutputExpression, VariableExpression, FunctionApplicationExpression
from concepts.dsl.tensor_value import TensorValue
from concepts.dm.crow.crow_function import CrowFunctionEvaluationMode
from concepts.dm.crow.controller import CrowControllerApplicationExpression
from concepts.dm.crow.crow_generator import CrowGeneratorApplicationExpression

__all__ = [
    'CrowBehaviorBodyPrimitiveBase', 'CrowBehaviorCommit', 'CrowAchieveExpression', 'CrowUntrackExpression', 'CrowMemQueryExpression', 'CrowBindExpression', 'CrowAssertExpression', 'CrowRuntimeAssignmentExpression', 'CrowFeatureAssignmentExpression',
    'CrowBehaviorBodySuiteBase', 'CrowBehaviorConditionSuite', 'CrowBehaviorWhileLoopSuite', 'CrowBehaviorForeachLoopSuite', 'CrowBehaviorStatementOrdering', 'CrowBehaviorOrderingSuite',
    'CrowBehaviorBodyItem', 'CrowBehavior', 'CrowBehaviorApplier', 'CrowBehaviorApplicationExpression', 'CrowBehaviorEffectApplicationExpression',
    'CrowEffectApplier'
]


CrowBehaviorBodyItem = Union[
    'CrowBehaviorCommit',
    'CrowAchieveExpression',
    'CrowUntrackExpression',
    'CrowAssertExpression',
    'CrowBindExpression',
    'CrowMemQueryExpression',
    'CrowControllerApplicationExpression',
    'CrowRuntimeAssignmentExpression',
    'CrowFeatureAssignmentExpression',
    'CrowBehaviorConditionSuite',
    'CrowBehaviorWhileLoopSuite',
    'CrowBehaviorForeachLoopSuite',
    'CrowBehaviorOrderingSuite',
    'CrowBehaviorApplicationExpression'
]


class CrowBehaviorBodyPrimitiveBase(object):
    pass


class CrowBehaviorCommit(CrowBehaviorBodyPrimitiveBase):
    def __init__(self, sketch: bool = True, csp: bool = True, execution: bool = False):
        self.sketch = sketch
        self.csp = csp
        self.execution = execution

    execution: bool
    """Whether to commit the execution: the execution is always going to achieve the subgoal."""

    sketch: bool
    """Whether to commit the sketch: the sketch is always going to achieve the subgoal."""

    csp: bool
    """Whether to commit the CSP variables: the CSP is always going to achieve the subgoal."""

    @property
    def flags(self):
        return dict(sketch=self.sketch, csp=self.csp, execution=self.execution)

    @property
    def flags_str(self):
        return ', '.join(f'{key}={value}' for key, value in self.flags.items())

    def __str__(self):
        return f'Commit({self.flags_str})'

    def __repr__(self):
        return str(self)

    @classmethod
    def execution_only(cls):
        return cls(sketch=False, csp=False, execution=True)


class CrowAchieveExpression(CrowBehaviorBodyPrimitiveBase):
    def __init__(self, goal: ValueOutputExpression, once: bool = False, is_policy_achieve: bool = False, ordered: bool = False, serializable: bool = False, refinement_compressible: bool = True):
        """Initialize an achieve expression.

        Args:
            goal: the goal of the achieve expression.
            once: whether the goal should be achieved only once. By default, the goal will be held until the end of the behavior.
        """
        self.goal = goal
        self.once = once
        self.is_policy_achieve = is_policy_achieve
        self.ordered = ordered
        self.serializable = serializable
        self.refinement_compressible = refinement_compressible

    goal: ValueOutputExpression
    """The goal of the achieve expression."""

    once: bool
    """Whether the goal should be achieved only once."""

    @property
    def flags(self):
        return dict(ordered=self.ordered, serializable=self.serializable, refinement_compressible=self.refinement_compressible)

    @property
    def flags_str(self):
        return ', '.join(f'{key}={value}' for key, value in self.flags.items())

    @property
    def op_str(self):
        if self.is_policy_achieve:
            if self.once:
                return f'pachieve_once[{self.flags_str}]'
            return f'pachieve_hold[{self.flags_str}]'
        if self.once:
            return 'achieve_once'
        return 'achieve_hold'

    def __str__(self):
        if self.is_policy_achieve:
            if self.once:
                return f'PolicyAchieveOnce({self.goal}, ordered={self.ordered}, serializable={self.serializable}, refinement_compressible={self.refinement_compressible})'
            return f'PolicyAchieveHold({self.goal}, ordered={self.ordered}, serializable={self.serializable}, refinement_compressible={self.refinement_compressible})'
        if self.once:
            return f'AchieveOnce({self.goal})'
        return f'AchieveHold({self.goal})'

    def __repr__(self):
        return str(self)


class CrowUntrackExpression(CrowBehaviorBodyPrimitiveBase):
    def __init__(self, goal: Union[ValueOutputExpression, NullExpression]):
        self.goal = goal

    goal: Union[ValueOutputExpression, NullExpression]
    """The goal of the achieve expression."""

    def __str__(self):
        return f'Untrack({self.goal})'

    def __repr__(self):
        return str(self)


class CrowAssertExpression(CrowBehaviorBodyPrimitiveBase):
    def __init__(self, bool_expr: ValueOutputExpression, once: bool = True):
        self.bool_expr = bool_expr
        self.once = once

    bool_expr: ValueOutputExpression
    """The boolean expression of the assert expression."""

    once: bool
    """Whether the assert should be checked only once."""

    @property
    def op_str(self):
        if self.once:
            return 'assert_once'
        return 'assert_hold'

    def __str__(self):
        if self.once:
            return f'AssertOnce({self.bool_expr})'
        return f'AssertHold({self.bool_expr})'

    def __repr__(self):
        return str(self)


class CrowBindExpression(CrowBehaviorBodyPrimitiveBase):
    def __init__(self, variables: Sequence[Variable], goal: Union[ValueOutputExpression, CrowGeneratorApplicationExpression]):
        self.variables = tuple(variables)
        self.goal = goal
        self.is_object_bind = self._check_object_bind()

    variables: Tuple[Variable, ...]
    """The variables to bind."""

    goal: Union[ValueOutputExpression, CrowGeneratorApplicationExpression]
    """The goal of the bind expression."""

    def _check_object_bind(self) -> bool:
        """Check if all variables to be bound are object variables. Note that we currently do not support mixed bind."""
        is_object_bind = [isinstance(var.dtype, ObjectType) for var in self.variables]
        # if any(is_object_bind) and not all(is_object_bind):
        #     raise ValueError(f'Mixed bind is not supported: {self.variables}')
        return all(is_object_bind)

    def __str__(self):
        return f'Bind({", ".join(str(var) for var in self.variables)} <- {self.goal})'

    def __repr__(self):
        return str(self)


class CrowMemQueryExpression(CrowBehaviorBodyPrimitiveBase):
    def __init__(self, query: ObjectOrValueOutputExpression):
        self.query = query

    query: ObjectOrValueOutputExpression
    """The query to be executed."""

    def __str__(self):
        return f'MemQuery({self.query})'

    def __repr__(self):
        return str(self)


class CrowRuntimeAssignmentExpression(CrowBehaviorBodyPrimitiveBase):
    def __init__(self, variable: Variable, value: ObjectOrValueOutputExpression):
        self.variable = variable
        self.value = value

    variable: Variable
    """The variable to assign."""

    value: ObjectOrValueOutputExpression
    """The value to assign."""

    def __str__(self):
        return f'Assign({self.variable} <- {self.value})'

    def __repr__(self):
        return str(self)


class CrowFeatureAssignmentExpression(CrowBehaviorBodyPrimitiveBase):
    def __init__(self, feature: Union[FunctionApplicationExpression], value: Union[ObjectOrValueOutputExpression, NullExpression], simulation: bool = False, execution: bool = False, vision_update: bool = False):
        self.feature = feature
        self.value = value
        self.evaluation_mode = CrowFunctionEvaluationMode.from_bools(simulation=simulation, execution=execution)
        self.vision_update = vision_update

    feature: Union[FunctionApplicationExpression]
    """The feature to assign."""

    value: Union[ObjectOrValueOutputExpression, NullExpression]
    """The value to assign."""

    evaluation_mode: CrowFunctionEvaluationMode
    """The evaluation mode of the assignment."""

    vision_update: bool
    """Whether the feature assignment should be override by visual perception during execution time."""

    def __str__(self):

        return f'AssignFeature{self.evaluation_mode.get_prefix()}({self.feature} <- {self.value})'

    def get_prefix(self) -> str:
        tags = list()
        if self.evaluation_mode == CrowFunctionEvaluationMode.FUNCTIONAL:
            pass
        else:
            tags.append(self.evaluation_mode.get_short_name())
        if self.vision_update:
            tags.append('vision_update')
        if len(tags) == 0:
            return ''
        return f'[{", ".join(tags)}]'

    def __repr__(self):
        return str(self)


class CrowBehaviorBodySuiteBase(object):
    def __init__(self, statements: Sequence[CrowBehaviorBodyItem]):
        self._statements = tuple(statements)

    @property
    def statements(self) -> Tuple[CrowBehaviorBodyItem, ...]:
        """The statements in a behavior body suite."""
        return self._statements

    def __str__(self):
        return '\n'.join(map(str, self.statements))

    def __repr__(self):
        return f'{self.__class__.__name__}{{\n{indent_text(self.__str__())}\n}}'


class CrowBehaviorConditionSuite(CrowBehaviorBodySuiteBase):
    def __init__(self, condition: ValueOutputExpression, statements: Sequence[CrowBehaviorBodyItem], else_statements: Optional[Sequence[CrowBehaviorBodyItem]] = None):
        super().__init__(statements)
        self.condition = condition
        self.else_statements = else_statements

    condition: ValueOutputExpression
    """The condition of the behavior body suite."""

    else_statements: Optional[Sequence[CrowBehaviorBodyItem]]
    """The else statement of the behavior body suite."""

    def __str__(self):
        if self.else_statements is not None:
            else_statement_str = '\n'.join(map(str, self.else_statements))
            return f'if {self.condition}:\n{indent_text(super().__str__())}\nelse:\n{indent_text(else_statement_str)}'
        return f'if {self.condition}:\n{indent_text(super().__str__())}'


class CrowBehaviorWhileLoopSuite(CrowBehaviorBodySuiteBase):
    def __init__(self, condition: ValueOutputExpression, statements: Sequence[CrowBehaviorBodyItem], max_depth: int = 10):
        super().__init__(statements)
        self.condition = condition
        self.max_depth = max_depth

    condition: ValueOutputExpression
    """The condition of the behavior body suite."""

    max_depth: int
    """The maximum depth of the while loop."""

    def __str__(self):
        return f'while {self.condition}:\n{indent_text(super().__str__())}'


class CrowBehaviorForeachLoopSuite(CrowBehaviorBodySuiteBase):
    def __init__(self, variable: Variable, statements: Sequence[CrowBehaviorBodyItem], loop_in_expression: Optional[ObjectOrValueOutputExpression] = None):
        super().__init__(statements)
        self.variable = variable
        self.loop_in_expression = loop_in_expression

    variable: Variable
    """The variable (including its type) to iterate over."""

    loop_in_expression: Optional[ObjectOrValueOutputExpression]
    """The expression that generates the set of values to iterate over."""

    @property
    def is_foreach_in_expression(self) -> bool:
        return self.loop_in_expression is not None

    def __str__(self):
        if self.loop_in_expression is not None:
            return f'foreach {self.variable} in {self.loop_in_expression}:\n{indent_text(super().__str__())}'
        return f'foreach {self.variable}:\n{indent_text(super().__str__())}'


class CrowBehaviorStatementOrdering(JacEnum):
    """The ordering of statements in a behavior body ordering suite. There are four types of ordering:

    - SEQUENTIAL: the statements are executed sequentially.
    - UNORDERED: the statements are executed in an unordered way. All achieve expressions would be achieved, but the order is not specified.
    - PROMOTABLE: the statements are executed in a promotable way. In particular, all achieve expressions would be achieved, but they might be promoted to the front of the behavior sequence.
    - PREAMBLE: the statements are executed in a preamble way. This must be the first statement in the behavior body and it specifies the preamble of the behavior, which will be executed at the place where the behavior is refined.
    - CRITICAL: the statements are executed in a critical way. This can only be used inside the promotable body and it specifies a critical section.
    """

    SEQUENTIAL = 'sequential'
    UNORDERED = 'unordered'
    PROMOTABLE = 'promotable'
    PREAMBLE = 'preamble'
    CRITICAL = 'critical'
    ALTERNATIVE = 'alternative'


class CrowBehaviorOrderingSuite(CrowBehaviorBodySuiteBase):
    def __init__(self, order: Union[str, CrowBehaviorStatementOrdering], statements: Sequence[CrowBehaviorBodyItem], variable_scope_identifier: Optional[int] = None, _skip_simplify: bool = False):
        order = CrowBehaviorStatementOrdering.from_string(order)

        # NB(Jiayuan Mao @ 2024/07/16): when the order is PROMOTABLE and there is only a single CRITICAL suite, we will wrap it with a SEQUENTIAL suite.
        # This forbids us executing "pop_right_statement" on the CRITICAL suite.
        if order is CrowBehaviorStatementOrdering.PROMOTABLE:
            if len(statements) == 1 and isinstance(statements[0], CrowBehaviorOrderingSuite) and statements[0].order == CrowBehaviorStatementOrdering.CRITICAL:
                statements = [CrowBehaviorOrderingSuite(CrowBehaviorStatementOrdering.SEQUENTIAL, [statements[0]], variable_scope_identifier, _skip_simplify=True)]
                _skip_simplify = True

        super().__init__(statements)
        self.order = order
        self.variable_scope_identifier = variable_scope_identifier

        if not _skip_simplify:
            self._simplify_statements()

    ORDER = CrowBehaviorStatementOrdering

    order: CrowBehaviorStatementOrdering
    """The ordering of the behavior body suite."""

    variable_scope_identifier: Optional[int]
    """The identifier of the variable scope. This is only used during search."""

    def is_empty(self) -> bool:
        return len(self.statements) == 0

    def _simplify_statements(self):
        """Simplify the statements in the suite."""
        simplified_statements = []
        if len(self.statements) == 1:
            if isinstance(self.statements[0], CrowBehaviorOrderingSuite):
                if self.order in (CrowBehaviorStatementOrdering.UNORDERED, CrowBehaviorStatementOrdering.SEQUENTIAL):
                    self.order = self.statements[0].order
                    self.variable_scope_identifier = self.statements[0].variable_scope_identifier
                    self._statements = self._statements[0].statements
            return

        for stmt in self.statements:
            if isinstance(stmt, CrowBehaviorOrderingSuite):
                if stmt.order == self.order and self.variable_scope_identifier == stmt.variable_scope_identifier:
                    simplified_statements.extend(stmt.statements)
                else:
                    simplified_statements.append(stmt)
            else:
                simplified_statements.append(stmt)
        self._statements = tuple(simplified_statements)

    def clone(self, variable_scope_identifier: int):
        statements = [stmt.clone(variable_scope_identifier) if isinstance(stmt, CrowBehaviorOrderingSuite) else stmt for stmt in self.statements]
        return CrowBehaviorOrderingSuite(self.order, statements, variable_scope_identifier, _skip_simplify=True)

    def is_single_statement(self) -> bool:
        if len(self.statements) != 1:
            return False
        if isinstance(self.statements[0], CrowBehaviorOrderingSuite):
            return self.statements[0].is_single_statement()
        return True

    def pop_right_statement(self) -> Iterator[Tuple[Optional['CrowBehaviorOrderingSuite'], CrowBehaviorBodyItem, int]]:
        assert len(self.statements) > 0, 'Cannot pop from an empty suite.'
        if self.order == CrowBehaviorStatementOrdering.SEQUENTIAL:
            # NB(Jiayuan Mao @ 2024/07/16): If the statement is a CRITICAL suite, we just pop the entire suite.
            if isinstance(self.statements[-1], CrowBehaviorOrderingSuite) and self.statements[-1].order is CrowBehaviorStatementOrdering.CRITICAL:
                stmt = self.statements[-1]
                if len(self.statements) == 1:
                    yield None, stmt, self.variable_scope_identifier
                else:
                    yield CrowBehaviorOrderingSuite(self.order, self.statements[:-1], variable_scope_identifier=self.variable_scope_identifier), stmt, self.variable_scope_identifier
            elif isinstance(self.statements[-1], CrowBehaviorOrderingSuite):
                for rest, rstmt, scope in self.statements[-1].pop_right_statement():
                    if rest is not None:
                        yield CrowBehaviorOrderingSuite(self.order, self.statements[:-1] + (rest,), variable_scope_identifier=self.variable_scope_identifier), rstmt, scope
                    else:
                        if len(self.statements) == 1:
                            yield None, rstmt, scope
                        else:
                            yield CrowBehaviorOrderingSuite(self.order, self.statements[:-1], variable_scope_identifier=self.variable_scope_identifier), rstmt, scope
            else:
                if len(self.statements) == 1:
                    yield None, self.statements[0], self.variable_scope_identifier
                else:
                    yield CrowBehaviorOrderingSuite(self.order, self.statements[:-1], variable_scope_identifier=self.variable_scope_identifier), self.statements[-1], self.variable_scope_identifier
        elif self.order == CrowBehaviorStatementOrdering.UNORDERED:
            for i, stmt in enumerate(self.statements):
                # NB(Jiayuan Mao @ 2024/07/16): If the statement is a CRITICAL suite, we just pop the entire suite.
                if isinstance(stmt, CrowBehaviorOrderingSuite) and stmt.order is CrowBehaviorStatementOrdering.CRITICAL:
                    if len(self.statements) == 1:
                        yield None, stmt, self.variable_scope_identifier
                    else:
                        yield CrowBehaviorOrderingSuite(self.order, self.statements[:i] + self.statements[i + 1:], variable_scope_identifier=self.variable_scope_identifier), stmt, self.variable_scope_identifier
                elif isinstance(stmt, CrowBehaviorOrderingSuite):
                    for rest, rstmt, scope in stmt.pop_right_statement():
                        if rest is not None:
                            yield CrowBehaviorOrderingSuite(self.order, self.statements[:i] + (rest,) + self.statements[i + 1:], variable_scope_identifier=self.variable_scope_identifier), rstmt, scope
                        else:
                            if len(self.statements) == 1:
                                yield None, rstmt, scope
                            else:
                                yield CrowBehaviorOrderingSuite(self.order, self.statements[:i] + self.statements[i + 1:], variable_scope_identifier=self.variable_scope_identifier), rstmt, scope
                else:
                    if len(self.statements) == 1:
                        yield None, stmt, self.variable_scope_identifier
                    else:
                        yield CrowBehaviorOrderingSuite(self.order, self.statements[:i] + self.statements[i + 1:], variable_scope_identifier=self.variable_scope_identifier), stmt, self.variable_scope_identifier
        elif self.order == CrowBehaviorStatementOrdering.ALTERNATIVE:
            for i, stmt in enumerate(self.statements):
                if isinstance(stmt, CrowBehaviorOrderingSuite):
                    yield from stmt.pop_right_statement()
                else:
                    yield None, stmt, self.variable_scope_identifier
        elif self.order in (CrowBehaviorStatementOrdering.PROMOTABLE, CrowBehaviorStatementOrdering.CRITICAL):
            raise ValueError(f'Cannot pop from a {self.order} suite.')

    def unwrap_critical(self):
        if len(self.statements) == 1:
            return self.statements[0]
        return CrowBehaviorOrderingSuite(CrowBehaviorStatementOrdering.SEQUENTIAL, self.statements, variable_scope_identifier=self.variable_scope_identifier)

    def unwrap_alternative(self) -> Iterator['CrowBehaviorBodyItem']:
        yield from self.statements

    def iter_statements(self):
        for stmt in self.statements:
            if isinstance(stmt, CrowBehaviorOrderingSuite):
                yield from stmt.iter_statements()
            else:
                yield stmt

    def iter_statements_with_scope(self):
        for stmt in self.statements:
            if isinstance(stmt, CrowBehaviorOrderingSuite):
                yield from stmt.iter_statements_with_scope()
            else:
                yield stmt, self.variable_scope_identifier

    def split_preamble_and_promotable(self) -> Tuple[Optional[Tuple[CrowBehaviorBodyItem, ...]], Optional[Tuple[CrowBehaviorBodyItem, ...]], Tuple[CrowBehaviorBodyItem, ...]]:
        """Split the body into three parts: the preamble part, the promotable part, and the rest part."""
        preamble = None
        promotable = None
        rest = self.statements

        contains_preamble = any(isinstance(item, CrowBehaviorOrderingSuite) and item.order == CrowBehaviorStatementOrdering.PREAMBLE for item in rest)
        contains_promotable = any(isinstance(item, CrowBehaviorOrderingSuite) and item.order == CrowBehaviorStatementOrdering.PROMOTABLE for item in rest)

        if contains_preamble:  # handles {preamble, promotable, rest} or {preamble, rest}
            if len(rest) > 0 and isinstance(rest[0], CrowBehaviorOrderingSuite) and rest[0].order == CrowBehaviorStatementOrdering.PREAMBLE:
                preamble = rest[0].statements
                rest = rest[1:]
            if len(rest) > 0 and isinstance(rest[0], CrowBehaviorOrderingSuite) and rest[0].order == CrowBehaviorStatementOrdering.PROMOTABLE:
                promotable = rest[0].statements
                rest = rest[1:]
        elif contains_promotable:  # handles {<statements>, promotable, rest}
            preamble = list()
            for i, item in enumerate(rest):
                if isinstance(item, CrowBehaviorOrderingSuite) and item.order == CrowBehaviorStatementOrdering.PROMOTABLE:
                    promotable = item.statements
                    break
                preamble.append(item)
            rest = rest[i + 1:]
            if len(preamble) == 0:
                preamble = None
        return preamble, promotable, rest

    def split_promotable(self) -> Tuple[Optional[Tuple[CrowBehaviorBodyItem, ...]], Tuple[CrowBehaviorBodyItem, ...]]:
        """Split the body into two parts: the promotable part and the rest part."""
        warnings.warn('This method is deprecated. Use split_preamble_and_promotable instead.', DeprecationWarning)
        body = self.statements
        if len(body) > 0 and isinstance(body[0], CrowBehaviorOrderingSuite) and body[0].order == CrowBehaviorStatementOrdering.PROMOTABLE:
            return body[0].statements, body[1:]
        return None, body

    def get_flatten_body(self) -> Tuple[CrowBehaviorBodyItem, ...]:
        """Get the flatten body of the behavior. It only flattens the PROMOTABLE body. Note that this function does NOT recursively flatten the body.
        Instead, it returns a list of items that is the concatenation of the preamble section (left), the promotable section (middle) and the rest of the body (right).

        Returns:
            the flatten body.
        """
        flatten_body = []
        for item in self.statements:
            if isinstance(item, CrowBehaviorOrderingSuite) and item.order == CrowBehaviorStatementOrdering.PREAMBLE:
                flatten_body.extend(item.statements)
            elif isinstance(item, CrowBehaviorOrderingSuite) and item.order == CrowBehaviorStatementOrdering.PROMOTABLE:
                flatten_body.extend(item.statements)
            else:
                flatten_body.append(item)
        return tuple(flatten_body)

    def __str__(self):
        if self.variable_scope_identifier is None:
            return f'{self.order.value}{{\n{indent_text(super().__str__())}\n}}'
        return f'{self.order.value}@{self.variable_scope_identifier}{{\n{indent_text(super().__str__())}\n}}'

    @classmethod
    def make_sequential(cls, *statements: Union[CrowBehaviorBodyItem, Tuple[CrowBehaviorBodyItem, ...]], variable_scope_identifier: Optional[int] = None, _skip_simplify: bool = False):
        if len(statements) == 1 and isinstance(statements[0], (tuple, list)):
            statements = statements[0]
        return cls(CrowBehaviorStatementOrdering.SEQUENTIAL, statements, variable_scope_identifier, _skip_simplify=_skip_simplify)

    @classmethod
    def make_unordered(cls, *statements: Union[CrowBehaviorBodyItem, Tuple[CrowBehaviorBodyItem, ...]], variable_scope_identifier: Optional[int] = None, _skip_simplify: bool = False):
        if len(statements) == 1 and isinstance(statements[0], (tuple, list)):
            statements = statements[0]
        return cls(CrowBehaviorStatementOrdering.UNORDERED, statements, variable_scope_identifier, _skip_simplify=_skip_simplify)

    @classmethod
    def make_promotable(cls, *statements: Union[CrowBehaviorBodyItem, Tuple[CrowBehaviorBodyItem, ...]], variable_scope_identifier: Optional[int] = None, _skip_simplify: bool = False):
        if len(statements) == 1 and isinstance(statements[0], (tuple, list)):
            statements = statements[0]
        return cls(CrowBehaviorStatementOrdering.PROMOTABLE, statements, variable_scope_identifier, _skip_simplify=_skip_simplify)


class CrowBehavior(object):
    """A behavior definition in the CROW planner has the following components:

    - name: the name of the behavior.
    - arguments: the arguments of the behavior.
    - goal: the goal of the behavior.
    - body: the body of the behavior.
    - preconditions: the preconditions of the behavior. This is equivalent to having an "assert" statement at the beginning of the body.
    - effects: the effects of the behavior. This section will update the state of the world.
    - heuristic: the heuristic of the behavior. This is used to guide the search process.
    - minimize: the expression to minimize. This is used to guide the search process.
    """

    def __init__(
        self, name: str, arguments: Sequence[Variable],
        goal: Optional[ValueOutputExpression], body: CrowBehaviorOrderingSuite,
        effect_body: Optional[CrowBehaviorOrderingSuite] = None,
        heuristic: Optional[CrowBehaviorOrderingSuite] = None,
        minimize: Optional[ValueOutputExpression] = None,
        always: bool = False,
        python_effect: bool = False
    ):
        """Initialize a new behavior.

        Args:
            name: the name of the behavior.
            arguments: the arguments of the behavior.
            goal: the goal of the behavior.
            body: the body of the behavior.
            effect_body: the body of the effects.
            heuristic: the heuristic of the behavior.
            minimize: the expression to minimize.
            always: whether the behavior is always feasible (it can always achieve the specified goal in any states). Legacy option and unused.
            python_effect: whether the effect is declared in Python.
        """
        self.name = name
        self.arguments = tuple(arguments)
        self.goal = goal if goal is not None else NullExpression(BOOL)
        self.body = body
        self.effect_body = effect_body if effect_body is not None else CrowBehaviorOrderingSuite.make_sequential()
        self.heuristic = heuristic
        self.minimize = minimize
        self.always = always
        self.python_effect = python_effect

        self._check_body()

    name: str
    """The name of the behavior."""

    arguments: Tuple[Variable, ...]
    """The arguments of the behavior."""

    goal: ValueOutputExpression
    """The goal of the behavior."""

    body: CrowBehaviorOrderingSuite
    """The body of the behavior."""

    effect_body: CrowBehaviorOrderingSuite
    """The effects of the behavior."""

    heuristic: Optional[CrowBehaviorOrderingSuite]
    """The heuristic of the behavior."""

    minimize: Optional[ValueOutputExpression]
    """The expression to minimize."""

    always: bool
    """Whether the behavior is always feasible (it can always achieve the specified goal in any states)."""

    python_effect: bool
    """Whether the effect is declared in Python."""

    @property
    def argument_names(self) -> Tuple[str, ...]:
        """The names of the arguments."""
        return tuple(arg.name for arg in self.arguments)

    @property
    def argument_types(self) -> Tuple[Union[ObjectType, ValueType], ...]:
        """The types of the arguments."""
        return tuple(arg.dtype for arg in self.arguments)

    def _check_body(self):
        if not isinstance(self.body, CrowBehaviorOrderingSuite):
            raise ValueError(f'Invalid body type: {type(self.body)}')
        if self.body.order != CrowBehaviorStatementOrdering.SEQUENTIAL:
            raise ValueError(f'Invalid body ordering: {self.body.order}')
        if not isinstance(self.effect_body, CrowBehaviorOrderingSuite):
            raise ValueError(f'Invalid effect body type: {type(self.effect_body)}')
        if self.effect_body.order != CrowBehaviorStatementOrdering.SEQUENTIAL:
            raise ValueError(f'Invalid effect body ordering: {self.effect_body.order}')

        body = self.body.statements
        # TODO(Jiayuan Mao @ 2024/07/13): Now we will allow additional statements before the promotable body, and they will be automatically declared as the preamble body.

        nr_promotable_bodies = sum(isinstance(item, CrowBehaviorOrderingSuite) and item.order == CrowBehaviorStatementOrdering.PROMOTABLE for item in body)
        if nr_promotable_bodies > 1:
            raise ValueError(f'Multiple promotable bodies are not allowed: {nr_promotable_bodies}')
        contains_preamble = any(isinstance(item, CrowBehaviorOrderingSuite) and item.order == CrowBehaviorStatementOrdering.PREAMBLE for item in body)
        if contains_preamble:
            expected_promotable_body_position = 1
        else:
            expected_promotable_body_position = None

        for i, item in enumerate(body):
            if isinstance(item, (CrowBindExpression, CrowMemQueryExpression, CrowRuntimeAssignmentExpression, CrowAssertExpression, CrowControllerApplicationExpression)):
                continue
            elif isinstance(item, (CrowAchieveExpression, CrowUntrackExpression)):
                pass
            elif isinstance(item, CrowBehaviorOrderingSuite) and item.order in (CrowBehaviorStatementOrdering.UNORDERED, CrowBehaviorStatementOrdering.SEQUENTIAL):
                self._check_regular_body(item.order, item.statements)
            elif isinstance(item, CrowBehaviorOrderingSuite) and item.order == CrowBehaviorStatementOrdering.CRITICAL:
                raise ValueError(f'Invalid critical body item: {type(item)}. Critical body can only be used inside the promotable body.')
            elif isinstance(item, CrowBehaviorOrderingSuite) and item.order == CrowBehaviorStatementOrdering.PREAMBLE:
                if i != 0:
                    raise ValueError(f'Preamble body can only be the first statement in the main body.')
                self._check_preamble_body(body[i].statements)
            elif isinstance(item, CrowBehaviorOrderingSuite) and item.order == CrowBehaviorStatementOrdering.PROMOTABLE:
                if expected_promotable_body_position is not None and i != expected_promotable_body_position:
                    raise ValueError(f'Promotable body can only be the first statement (or the second if there is a preamble suite) in the main body.')
                self._check_promotable_body(body[i].statements)

    def _check_preamble_body(self, body):
        pass

    def _check_promotable_body(self, body):
        for item in body:
            if isinstance(item, CrowBehaviorOrderingSuite) and item.order == CrowBehaviorStatementOrdering.PROMOTABLE:
                raise ValueError(f'Promotable body can only be the first statement in the main body.')
            elif isinstance(item, CrowBehaviorOrderingSuite) and item.order in (CrowBehaviorStatementOrdering.UNORDERED, CrowBehaviorStatementOrdering.SEQUENTIAL):
                self._check_promotable_body(item.statements)
            elif isinstance(item, CrowBehaviorOrderingSuite) and item.order == CrowBehaviorStatementOrdering.CRITICAL:
                self._check_critical_body(item.statements)
            # elif isinstance(item, CrowBehaviorForeachLoopSuite):
            #     raise ValueError(f'Foreach loop is not allowed in the promotable body.')
            elif isinstance(item, CrowBehaviorWhileLoopSuite):
                raise ValueError(f'While loop is not allowed in the promotable body.')
            elif isinstance(item, CrowBehaviorConditionSuite):
                raise ValueError(f'Condition suite is not allowed in the promotable body.')

    def _check_regular_body(self, order, body):
        for item in body:
            if isinstance(item, CrowBehaviorOrderingSuite) and item.order == CrowBehaviorStatementOrdering.PROMOTABLE:
                raise ValueError(f'Promotable body can only be the first statement in the main body.')
            elif isinstance(item, CrowBehaviorOrderingSuite) and item.order == CrowBehaviorStatementOrdering.CRITICAL:
                raise ValueError(f'Critical body can only be used inside the promotable body.')
            elif isinstance(item, CrowBehaviorOrderingSuite) and item.order in (CrowBehaviorStatementOrdering.UNORDERED, CrowBehaviorStatementOrdering.SEQUENTIAL):
                self._check_regular_body(item.order, item.statements)

    def _check_critical_body(self, body):
        for item in body:
            if isinstance(item, CrowBehaviorOrderingSuite) and item.order == CrowBehaviorStatementOrdering.CRITICAL:
                raise ValueError(f'Critical body can not be nested.')
            elif isinstance(item, CrowBehaviorOrderingSuite) and item.order in (CrowBehaviorStatementOrdering.UNORDERED, CrowBehaviorStatementOrdering.SEQUENTIAL):
                self._check_critical_body(item.statements)
            elif isinstance(item, CrowBehaviorOrderingSuite) and item.order == CrowBehaviorStatementOrdering.PROMOTABLE:
                raise ValueError(f'Promotable body can not be used inside the critical body.')
            # elif isinstance(item, CrowBehaviorForeachLoopSuite):
            #     raise ValueError(f'Foreach loop is not allowed in the critical body.')
            elif isinstance(item, CrowBehaviorWhileLoopSuite):
                raise ValueError(f'While loop is not allowed in the critical body.')
            elif isinstance(item, CrowBehaviorConditionSuite):
                raise ValueError(f'Condition suite is not allowed in the critical body.')

    def is_sequential_only(self) -> bool:
        """Check if the behavior body contains only sequential statements (i.e., no preamble and promotable)."""
        for item in self.body.statements:
            if isinstance(item, CrowBehaviorOrderingSuite) and item.order in (CrowBehaviorStatementOrdering.PREAMBLE, CrowBehaviorStatementOrdering.PROMOTABLE):
                return False
        return True

    def assign_body_program_scope(self, scope_id: int) -> 'CrowBehaviorOrderingSuite':
        """Assign the program scope to the body."""
        return self.body.clone(scope_id)

    def short_str(self):
        return f'{self.name}({", ".join(map(str, self.arguments))})'

    def long_str(self):
        flag_string = ''
        if self.python_effect:
            flag_string = '[[python_effect]]'

        effect_string = '\n'.join(map(str, self.effect_body.statements))
        fmt = f'behavior {flag_string}{self.name}({", ".join(map(str, self.arguments))}):\n'
        fmt += f'  goal: {indent_text(str(self.goal)).lstrip()}\n'
        fmt += f'  body:\n{indent_text(str(self.body), 2)}\n'
        if len(effect_string) > 0:
            fmt += f'  effects:\n{indent_text(effect_string, 2)}\n'
        if self.heuristic is not None:
            fmt += f'  heuristic:\n{indent_text(str(self.heuristic), 2)}'
        return fmt

    def __str__(self):
        return self.short_str()

    __repr__ = repr_from_str


class CrowBehaviorApplier(object):
    def __init__(self, behavior: CrowBehavior, arguments: Sequence[Union[str, TensorValue]]):
        self.behavior = behavior
        self.arguments = tuple(arguments)

    behavior: CrowBehavior
    """The behavior to be applied."""

    arguments: Tuple[Union[str, TensorValue], ...]
    """The arguments of the controller application."""

    def __str__(self):
        return f'{self.behavior.name}({", ".join(str(arg) for arg in self.arguments)})'

    __repr__ = repr_from_str


class CrowBehaviorApplicationExpression(CrowBehaviorBodyPrimitiveBase):
    def __init__(self, behavior: CrowBehavior, arguments: Sequence[ObjectOrValueOutputExpression]):
        self.behavior = behavior
        self.arguments = tuple(arguments)

    behavior: CrowBehavior
    """The behavior to be applied."""

    arguments: Tuple[ObjectOrValueOutputExpression, ...]
    """The arguments of the controller application."""

    def __str__(self):
        return f'{self.behavior.name}({", ".join(str(arg) for arg in self.arguments)})'

    __repr__ = repr_from_str


class CrowBehaviorEffectApplicationExpression(CrowBehaviorBodyPrimitiveBase):
    def __init__(self, behavior: CrowBehavior, arguments: Optional[Sequence[ObjectOrValueOutputExpression]] = None):
        self.behavior = behavior

        if arguments is None:
            self.arguments = tuple(VariableExpression(var) for var in behavior.arguments)
        else:
            self.arguments = tuple(arguments)

    behavior: CrowBehavior
    """The behavior to be applied."""

    arguments: Tuple[ObjectOrValueOutputExpression, ...]
    """The arguments of the controller application."""

    def __str__(self):
        return f'{self.behavior.name}({", ".join(str(arg) for arg in self.arguments)})'

    __repr__ = repr_from_str


class CrowEffectApplier(object):
    """An effect applier is a wrapper around a behavior/controller that represents the application of the effects of a behavior or a controller.

    Recall that in CDL, the effects (updates to the planner state) is not associated with the controller itself, but higher-level behaviors. Therefore,
    we can not directly track the state updates if we only have a sequence of controller function applications. To address this issue, we introduce the
    concept of effect applier.

    At planning time, if the option `include_effect_appliers` is set to True, in the generated `controller_actions` list, we will include both the controller
    applications and the effect appliers, indicating the time when the state updates happens. The effect applier data structure records the statements
    that updates the state in the planning world and the `bounded_variables` that is used to execute the statements.

    To execute an effect applier based on a state, we can call:

    .. code-block:: python

        executor = ...
        state = ...
        effect_applier = ...

        from concepts.dm.crow.behavior_utils import execute_effect_applier
        execute_effect_applier(executor, effect_applier, state)

    To pretty print an effect applier, we simply call `str(effect_applier)`.

    The `cdl-plan` tool also provides a command `--print-effect-appliers`. If this flag is set, the tool will print the effect appliers in the generated plan.

    .. code-block:: bash

        cdl-plan blocksworld-problem-sussman-with-pragma.cdl --print-effect-appliers

    """
    def __init__(
        self, statements: Sequence[Union[CrowFeatureAssignmentExpression, CrowBehaviorForeachLoopSuite, CrowBehaviorConditionSuite]],
        bounded_variables: Dict[str, Union[ObjectConstant, TensorValue]], *,
        global_constraints: Optional[Dict[int, Sequence[ValueOutputExpression]]] = None, local_constraints: Optional[Sequence[ValueOutputExpression]] = None
    ):
        self.statements = tuple(statements)
        self.bounded_variables = bounded_variables
        self.global_constraints = global_constraints
        self.local_constraints = local_constraints

    global_constraints: Optional[Dict[int, Tuple[Tuple[ValueOutputExpression, ...], dict]]]
    """The global constraints of the controller application."""

    local_constraints: Optional[Tuple[Tuple[ValueOutputExpression, ...], dict]]
    """The local constraints of the controller application."""

    def set_constraints(self, global_constraints: Dict[int, Sequence[ValueOutputExpression]], global_scopes: Dict[int, dict], scope_id: int):
        # TODO(Jiayuan Mao @ 2025/01/14): set the scopes associated with these constraints...
        self.global_constraints = {k: (tuple(v), global_scopes[k]) for k, v in global_constraints.items()}
        self.local_constraints = self.global_constraints.get(scope_id, None)
        return self

    def short_str(self):
        from concepts.dm.crow.behavior_utils import format_behavior_statement
        fmt = '\n'.join(format_behavior_statement(x, scope=self.bounded_variables) for x in self.statements)
        return f'effect_apply{{\n{indent_text(fmt)}\n}}'

    def long_str(self) -> str:
        from concepts.dm.crow.behavior_utils import format_behavior_statement
        fmt = self.short_str()
        if self.global_constraints is not None:
            global_constraints_str = indent_text('\n'.join(f'{k}: {{{", ".join(str(format_behavior_statement(c, scope=scope)) for c in constraints)}}}' for k, (constraints, scope) in self.global_constraints.items()), 2)
            fmt += f'\n  with global constraints:\n{global_constraints_str}'
        if self.local_constraints is not None:
            local_constraints_str = '{' + ', '.join(str(format_behavior_statement(c, scope=self.local_constraints[1])) for c in self.local_constraints[0]) + '}'
            fmt += f'\n  with local constraints: {local_constraints_str}'
        return fmt

    def __str__(self):
        return self.short_str()

    __repr__ = repr_from_str
