#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : behavior_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/17/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import itertools
from typing import Any, Optional, Union, Iterator, Sequence, Tuple, List, Dict, NamedTuple
from jacinle.utils.printing import indent_text

from concepts.dsl.dsl_types import AutoType, QINDEX, Variable, ObjectConstant
from concepts.dsl.value import ListValue
from concepts.dsl.constraint import ConstraintSatisfactionProblem, OptimisticValue
from concepts.dsl.tensor_value import TensorValue
from concepts.dsl.tensor_state import StateObjectReference, StateObjectList
from concepts.dsl.expression import Expression, ExpressionDefinitionContext, ValueOutputExpression, ObjectOrValueOutputExpression, FunctionApplicationExpression, VariableExpression, VariableAssignmentExpression, ConstantExpression, ObjectConstantExpression, is_null_expression
from concepts.dsl.expression import AssignExpression, BoolExpression, BoolOpType
from concepts.dsl.expression_utils import surface_fol_downcast, find_free_variables

from concepts.dm.crow.controller import CrowController, CrowControllerApplicationExpression
from concepts.dm.crow.crow_expression_utils import crow_replace_expression_variables
from concepts.dm.crow.behavior import CrowBehavior, CrowAchieveExpression, CrowUntrackExpression, CrowBindExpression, CrowMemQueryExpression, CrowAssertExpression
from concepts.dm.crow.behavior import CrowRuntimeAssignmentExpression, CrowFeatureAssignmentExpression, CrowBehaviorApplicationExpression, CrowBehaviorEffectApplicationExpression
from concepts.dm.crow.behavior import CrowBehaviorOrderingSuite, CrowBehaviorCommit
from concepts.dm.crow.behavior import CrowBehaviorForeachLoopSuite, CrowBehaviorWhileLoopSuite, CrowBehaviorConditionSuite, CrowEffectApplier
from concepts.dm.crow.crow_domain import CrowDomain, CrowState
from concepts.dm.crow.executors.crow_executor import CrowExecutor

__all__ = [
    'ApplicableBehaviorItem', 'match_applicable_behaviors', 'match_policy_applicable_behaviors',
    'crow_replace_expression_variables_ext', 'format_behavior_statement', 'format_behavior_program',
    'execute_effect_statements', 'execute_behavior_effect_body', 'execute_effect_applier',
    'execute_object_bind', 'execute_additive_heuristic_program'
]


class ApplicableBehaviorItem(NamedTuple):
    """A behavior item that is applicable to the given goal."""

    behavior: CrowBehavior
    """The behavior that is applicable."""

    bounded_variables: Union[Dict[str, Union[Variable, ObjectConstant, TensorValue]], Dict[str, ObjectOrValueOutputExpression]]
    """The bounded variables that are used to instantiate the behavior. If deferred_execution is True, the values are ObjectOrValueOutputExpression, otherwise Variable or ObjectConstant."""

    defered_execution: bool = False
    """Whether the behavior should be executed in deferred mode."""


def match_applicable_behaviors(
    domain: CrowDomain, state: CrowState, goal: ValueOutputExpression, goal_scope: Dict[str, Any],
    return_all_candidates: bool = True
) -> List[ApplicableBehaviorItem]:
    candidate_behaviors = list()
    for behavior in domain.behaviors.values():
        goal_expr = behavior.goal
        if is_null_expression(goal_expr):
            continue
        if (variable_binding := surface_fol_downcast(goal_expr, goal)) is None:
            continue

        # TODO(Jiayuan Mao @ 2024/07/18): think about if this "promoted" variable binding is correct.
        # My current intuition is that, if a variable got changed to a different value later, this procedure will break...
        for variable, value in variable_binding.items():
            if isinstance(value, Variable):
                # assert value.name in goal_scope, f"Variable {value.name} not found in goal scope."
                if value.name in goal_scope:
                    variable_binding[variable] = goal_scope[value.name]
        candidate_behaviors.append(ApplicableBehaviorItem(behavior, variable_binding))

    # TODO: implement return_all_candidates
    return candidate_behaviors


def match_policy_applicable_behaviors(
    domain: CrowDomain, state: CrowState, goal: ValueOutputExpression, goal_scope: Dict[str, Any],
    return_all_candidates: bool = True,
    pachieve_kwargs: Dict[str, Any] = None,
) -> List[ApplicableBehaviorItem]:
    candidate_behaviors = match_applicable_behaviors(domain, state, goal, goal_scope, return_all_candidates)
    if len(candidate_behaviors) > 0:
        return candidate_behaviors

    free_variables = find_free_variables(goal)
    bounded_variables = {var.name: var for var in free_variables}
    if isinstance(goal, BoolExpression) and goal.bool_op is BoolOpType.AND:
        subgoals = [CrowAchieveExpression(subgoal, once=False, is_policy_achieve=True, **pachieve_kwargs) for subgoal in goal.arguments]
        if pachieve_kwargs['ordered']:
            goal_set = CrowBehaviorOrderingSuite.make_sequential(subgoals)
        else:
            goal_set = CrowBehaviorOrderingSuite.make_unordered(subgoals)
        if pachieve_kwargs['serializable']:
            program = CrowBehaviorOrderingSuite.make_sequential(goal_set)
        else:
            program = CrowBehaviorOrderingSuite.make_sequential(CrowBehaviorOrderingSuite.make_promotable(goal_set), _skip_simplify=True)
        return [ApplicableBehaviorItem(CrowBehavior('__pachive__', free_variables, None, program), bounded_variables)]
    elif isinstance(goal, BoolExpression) and goal.bool_op is BoolOpType.OR:
        subgoals = [CrowAchieveExpression(subgoal, once=False, is_policy_achieve=True, **pachieve_kwargs) for subgoal in goal.arguments]
        if pachieve_kwargs['serializable']:
            subgoals = [CrowBehaviorOrderingSuite.make_sequential(subgoal) for subgoal in subgoals]
        else:
            subgoals = [CrowBehaviorOrderingSuite.make_sequential(CrowBehaviorOrderingSuite.make_promotable(subgoal), _skip_simplify=True) for subgoal in subgoals]
        return [ApplicableBehaviorItem(CrowBehavior('__pachive__', free_variables, None, program), bounded_variables) for program in subgoals]
    else:
        raise ValueError(f"Goal type {goal} is not supported.")


def crow_replace_expression_variables_ext(
    expr: Union[Expression, CrowBindExpression, CrowMemQueryExpression, CrowAssertExpression, CrowRuntimeAssignmentExpression, CrowFeatureAssignmentExpression, CrowControllerApplicationExpression, CrowAchieveExpression, CrowUntrackExpression, CrowBehaviorApplicationExpression, CrowBehaviorEffectApplicationExpression],
    mappings: Optional[Dict[Union[FunctionApplicationExpression, VariableExpression], Union[Variable, ValueOutputExpression]]] = None,
    ctx: Optional[ExpressionDefinitionContext] = None,
) -> Union[ObjectOrValueOutputExpression, VariableAssignmentExpression, CrowBindExpression, CrowMemQueryExpression, CrowAssertExpression, CrowRuntimeAssignmentExpression, CrowFeatureAssignmentExpression, CrowControllerApplicationExpression, CrowAchieveExpression, CrowUntrackExpression, CrowBehaviorApplicationExpression, CrowBehaviorEffectApplicationExpression]:
    if isinstance(expr, Expression):
        return crow_replace_expression_variables(expr, mappings, ctx)
    if isinstance(expr, CrowBindExpression):
        return CrowBindExpression(expr.variables, crow_replace_expression_variables(expr.goal, mappings, ctx))
    if isinstance(expr, CrowMemQueryExpression):
        return CrowMemQueryExpression(crow_replace_expression_variables(expr.goal, mappings, ctx))
    if isinstance(expr, CrowAssertExpression):
        return CrowAssertExpression(crow_replace_expression_variables(expr.bool_expr, mappings, ctx), once=expr.once)
    if isinstance(expr, CrowRuntimeAssignmentExpression):
        return CrowRuntimeAssignmentExpression(expr.variable, crow_replace_expression_variables(expr.value, mappings, ctx))
    if isinstance(expr, CrowFeatureAssignmentExpression):
        return CrowFeatureAssignmentExpression(crow_replace_expression_variables(expr.feature, mappings, ctx), crow_replace_expression_variables(expr.value, mappings, ctx))
    if isinstance(expr, CrowControllerApplicationExpression):
        return CrowControllerApplicationExpression(expr.controller, [crow_replace_expression_variables(arg, mappings, ctx) for arg in expr.arguments])
    if isinstance(expr, CrowAchieveExpression):
        return CrowAchieveExpression(crow_replace_expression_variables(expr.goal, mappings, ctx), once=expr.once)
    if isinstance(expr, CrowUntrackExpression):
        return CrowUntrackExpression(crow_replace_expression_variables(expr.goal, mappings, ctx))
    if isinstance(expr, CrowBehaviorApplicationExpression):
        return CrowBehaviorApplicationExpression(expr.behavior, [crow_replace_expression_variables(arg, mappings, ctx) for arg in expr.arguments])
    if isinstance(expr, CrowBehaviorEffectApplicationExpression):
        return CrowBehaviorEffectApplicationExpression(expr.behavior, [crow_replace_expression_variables(arg, mappings, ctx) for arg in expr.arguments])
    raise ValueError(f'Invalid expression ({type(expr)}): {expr}')


def format_behavior_statement(
    program: Union[
        Expression, CrowBehaviorOrderingSuite, CrowBindExpression, CrowMemQueryExpression, CrowAssertExpression, CrowControllerApplicationExpression, CrowFeatureAssignmentExpression,
        CrowAchieveExpression, CrowUntrackExpression, CrowBehaviorEffectApplicationExpression, CrowBehaviorApplicationExpression
    ], scopes: Optional[dict] = None, scope_id: Optional[int] = None, scope: Optional[dict] = None
) -> str:
    if isinstance(program, CrowBehaviorOrderingSuite):
        return format_behavior_program(program, scopes)

    if scopes is None and scope is None and scope_id is not None:
        if isinstance(program, Expression):
            return str(program) + f'@({scope_id})'
        if isinstance(program, CrowBindExpression):
            return f'bind@{scope_id} {", ".join(str(var) for var in program.variables)} where {crow_replace_expression_variables(program.goal)}'
        if isinstance(program, CrowMemQueryExpression):
            return f'mem_query@{scope_id} {crow_replace_expression_variables(program.query)}'
        if isinstance(program, CrowAssertExpression):
            return f'{program.op_str}@{scope_id} {str(program.bool_expr)}'
        if isinstance(program, CrowRuntimeAssignmentExpression):
            return f'assign@{scope_id} {program.variable} = {program.value}'
        if isinstance(program, CrowControllerApplicationExpression):
            return f'do@{scope_id} {str(program)}'
        if isinstance(program, CrowFeatureAssignmentExpression):
            return f'assign_feature@{scope_id} {program.feature} = {program.value}'
        if isinstance(program, CrowAchieveExpression):
            return f'{program.op_str}@{scope_id} {str(program.goal)}'
        if isinstance(program, CrowUntrackExpression):
            return f'untrack@{scope_id} {str(program.goal)}'
        if isinstance(program, CrowBehaviorApplicationExpression):
            return f'apply@{scope_id} {str(program)}'
        if isinstance(program, CrowBehaviorEffectApplicationExpression):
            return f'effect@{scope_id} {str(program)}'
        if isinstance(program, CrowBehaviorCommit):
            return f'commit'
        if isinstance(program, CrowBehaviorForeachLoopSuite):
            body_str = '\n'.join(format_behavior_statement(stmt, scope_id=scope_id) for stmt in program.statements)
            return f'foreach@{scope_id} {program.variable} {{\n{body_str}\n}}'
        if isinstance(program, CrowBehaviorWhileLoopSuite):
            body_str = '\n'.join(format_behavior_statement(stmt, scope_id=scope_id) for stmt in program.statements)
            return f'while@{scope_id} {str(program.condition)} {{\n{body_str}\n}}'
        if isinstance(program, CrowBehaviorConditionSuite):
            if program.else_statements is None:
                body_str = '\n'.join(format_behavior_statement(stmt, scope_id=scope_id) for stmt in program.statements)
                return f'if@{scope_id} {str(program.condition)} {{\n{body_str}\n}}'
            body_str = '\n'.join(format_behavior_statement(stmt, scope_id=scope_id) for stmt in program.statements)
            else_body_str = '\n'.join(format_behavior_statement(stmt, scope_id=scope_id) for stmt in program.else_statements)
            return f'if@{scope_id} {str(program.condition)} {{\n{body_str}\n}} else {{\n{else_body_str}\n}}'

        raise ValueError(f'Invalid program ({type(program)}): {program}')

    if scope_id is not None and scopes is not None or scope is not None:
        if scope is None:
            scope = scopes.get(scope_id, dict())
        formatted_scope = dict()
        for name, value in scope.items():
            if isinstance(name, Variable):
                formatted_name = name
            else:
                formatted_name = Variable(name, AutoType)
            if isinstance(value, Variable):
                formatted_value = VariableExpression(value)
            elif isinstance(value, ObjectConstant):
                formatted_value = ObjectConstantExpression(value)
            elif isinstance(value, StateObjectReference):
                formatted_value = ObjectConstantExpression(ObjectConstant(value.name, value.dtype))
            elif isinstance(value, StateObjectList):
                formatted_value = ObjectConstantExpression(ObjectConstant(value, value.element_type))
            elif isinstance(value, TensorValue):
                formatted_value = ConstantExpression(value)
            else:
                formatted_value = value
            formatted_scope[VariableExpression(formatted_name)] = formatted_value
        scope = formatted_scope

    if isinstance(program, Expression):
        if scope is None:
            return str(program)
        return str(crow_replace_expression_variables(program, scope))
    if isinstance(program, CrowBindExpression):
        return f'bind {", ".join(str(var) for var in program.variables)} <- {crow_replace_expression_variables(program.goal, scope)}'
    if isinstance(program, CrowMemQueryExpression):
        return f'mem_query {crow_replace_expression_variables(program.query, scope)}'
    if isinstance(program, CrowAssertExpression):
        return f'{program.op_str} {crow_replace_expression_variables(program.bool_expr, scope)}'
    if isinstance(program, CrowRuntimeAssignmentExpression):
        return f'assign {program.variable} <- {crow_replace_expression_variables(program.value, scope)}'
    if isinstance(program, CrowControllerApplicationExpression):
        return f'do {crow_replace_expression_variables_ext(program, scope)}'
    if isinstance(program, CrowFeatureAssignmentExpression):
        return f'assign_feature {crow_replace_expression_variables(program.feature, scope)} <- {crow_replace_expression_variables(program.value, scope)}'
    if isinstance(program, CrowAchieveExpression):
        return f'{program.op_str} {crow_replace_expression_variables(program.goal, scope)}'
    if isinstance(program, CrowUntrackExpression):
        return f'untrack {crow_replace_expression_variables(program.goal, scope)}'
    if isinstance(program, CrowBehaviorApplicationExpression):
        return f'apply {crow_replace_expression_variables_ext(program, scope)}'
    if isinstance(program, CrowBehaviorEffectApplicationExpression):
        return f'effect {crow_replace_expression_variables_ext(program, scope)}'
    if isinstance(program, CrowBehaviorCommit):
        return f'commit'
    if isinstance(program, CrowBehaviorForeachLoopSuite):
        body_str = indent_text('\n'.join(format_behavior_statement(stmt, scopes=scopes, scope_id=scope_id, scope=scope) for stmt in program.statements))
        return f'foreach {program.variable} {{\n{body_str}\n}}'
    if isinstance(program, CrowBehaviorConditionSuite):
        if program.else_statements is None:
            body_str = indent_text('\n'.join(format_behavior_statement(stmt, scopes=scopes, scope_id=scope_id, scope=scope) for stmt in program.statements))
            return f'if {crow_replace_expression_variables(program.condition, scope)} {{\n{body_str}\n}}'
        body_str = indent_text('\n'.join(format_behavior_statement(stmt, scopes=scopes, scope_id=scope_id, scope=scope) for stmt in program.statements))
        else_body_str = indent_text('\n'.join(format_behavior_statement(stmt, scopes=scopes, scope_id=scope_id, scope=scope) for stmt in program.else_statements))
        return f'if {crow_replace_expression_variables(program.condition, scope)} {{\n{body_str}\n}} else {{\n{else_body_str}\n}}'

    raise ValueError(f'Invalid program ({type(program)}): {program}')


def format_behavior_program(program: CrowBehaviorOrderingSuite, scopes: Optional[dict], flatten=False) -> str:
    if flatten:
        unorderd_statements = list()
        for stmt, scope_id in program.iter_statements_with_scope():
            unorderd_statements.append(format_behavior_statement(stmt, scopes=scopes, scope_id=scope_id))
        unorderd_statements.sort()
        return 'unordered{\n' + indent_text('\n'.join(unorderd_statements)) + '\n}'

    fmt = f'{program.order.value}{{\n'
    for stmt in program.statements:
        if isinstance(stmt, CrowBehaviorOrderingSuite):
            fmt += indent_text(f'{format_behavior_program(stmt, scopes)}') + '\n'
        else:
            fmt += indent_text(format_behavior_statement(stmt, scopes=scopes, scope_id=program.variable_scope_identifier)) + '\n'
    fmt += '}'
    return fmt


def execute_effect_statements(
    executor: CrowExecutor, statements: Sequence[Union[CrowFeatureAssignmentExpression, CrowBehaviorForeachLoopSuite, CrowBehaviorConditionSuite]],
    state: CrowState, csp: Optional[ConstraintSatisfactionProblem] = None,
    scope: Optional[dict] = None, state_index: Optional[int] = None
):
    if scope is None:
        scope = dict()
    else:
        scope = {k: v for k, v in scope.items() if not (k.startswith('__') and k.endswith('__'))}
    for stmt in statements:
        if isinstance(stmt, CrowFeatureAssignmentExpression):
            with executor.update_effect_mode(stmt.evaluation_mode, state_index=state_index):
                executor.execute(AssignExpression(stmt.feature, stmt.value), state=state, csp=csp, bounded_variables=scope)
        elif isinstance(stmt, CrowBehaviorForeachLoopSuite) and stmt.is_foreach_in_expression:
            var = stmt.variable
            values = executor.execute(stmt.loop_in_expression, state=state, csp=csp, bounded_variables=scope).values
            for value in values:
                new_scope = scope.copy()
                new_scope[var.name] = value
                execute_effect_statements(executor, stmt.statements, state, csp=csp, scope=new_scope)
        elif isinstance(stmt, CrowBehaviorForeachLoopSuite):
            var = stmt.variable
            new_scope = scope.copy()
            for i, name in enumerate(state.object_type2name[var.typename]):
                new_scope[var.name] = ObjectConstant(name, var.dtype)
                execute_effect_statements(executor, stmt.statements, state, csp=csp, scope=new_scope)
        elif isinstance(stmt, CrowBehaviorConditionSuite):
            rv = executor.execute(stmt.condition, state=state, bounded_variables=scope).item()
            if isinstance(rv, OptimisticValue):
                raise NotImplementedError('OptimisticValue is not supported in the current implementation.')
            if bool(rv):
                execute_effect_statements(executor, stmt.statements, state, csp=csp, scope=scope)
            else:
                if stmt.else_statements is not None:
                    execute_effect_statements(executor, stmt.else_statements, state, csp=csp, scope=scope)
        else:
            raise ValueError(f'Unsupported statement type: {type(stmt)}')


def execute_behavior_effect_body(executor: CrowExecutor, behavior: Union[CrowBehavior, CrowController], state: CrowState, scope: dict, csp: Optional[ConstraintSatisfactionProblem] = None, state_index: Optional[int] = None) -> CrowState:
    """Execute a behavior effect with for-loops and conditional statements."""
    new_state = state.clone()
    execute_effect_statements(executor, behavior.effect_body.statements, new_state, csp=csp, scope=scope, state_index=state_index)
    return new_state


def execute_effect_applier(executor: CrowExecutor, applier: CrowEffectApplier, state: CrowState, csp: Optional[ConstraintSatisfactionProblem] = None, state_index: Optional[int] = None) -> CrowState:
    return execute_effect_statements(executor, applier.statements, state, csp=csp, scope=applier.bounded_variables, state_index=state_index)


def execute_object_bind(executor: CrowExecutor, stmt: CrowBindExpression, state: CrowState, scope: dict) -> Iterator[Dict[str, Union[ObjectConstant, Variable]]]:
    """Execute a bind statement and yield the possible scopes.

    Args:
        executor: the executor to execute the statement.
        stmt: the bind statement.
        state: the current state.
        scope: the current scope.

    Returns:
        an iterator of possible scopes with the bind statement applied.
    """

    if stmt.goal.is_null_expression:
        rv = TensorValue.TRUE
    else:
        eval_scope = {k: v for k, v in scope.items() if not (k.startswith('__') and k.endswith('__'))}
        for var in stmt.variables:
            # NB(Jiayuan Mao @ 2024/11/17): This is a hack to avoid the case where the variable is already in the scope.
            # The reason is that, for everywhere else in the code, the scope is a mapping from name string => value.
            # However, here, since we need to specify QINDEX, we need to use the variable object as the key.
            # Then this will cause two variables with the same name but different objects to be treated as different.
            if var.name in eval_scope:
                del eval_scope[var.name]
            eval_scope[var] = QINDEX
        rv = executor.execute(stmt.goal, state=state, bounded_variables=eval_scope)

    typeonly_indices_variables = list()
    typeonly_indices_values = list()
    for v in stmt.variables:
        if v.name not in rv.batch_variables:
            typeonly_indices_variables.append(v.name)
            typeonly_indices_values.append(range(len(state.object_type2name[v.dtype.typename])))
    for indices in rv.tensor.nonzero():
        for typeonly_indices in itertools.product(*typeonly_indices_values):
            new_scope_variables = scope.copy()
            for var in stmt.variables:
                if var.name in rv.batch_variables:
                    new_scope_variables[var.name] = ObjectConstant(state.object_type2name[var.dtype.typename][indices[rv.batch_variables.index(var.name)]], var.dtype)
                else:
                    new_scope_variables[var.name] = ObjectConstant(state.object_type2name[var.dtype.typename][typeonly_indices[typeonly_indices_variables.index(var.name)]], var.dtype)
            yield new_scope_variables


class CrowAdditiveHeuristicProgramExecutor(object):
    def __init__(self, executor: CrowExecutor, state: CrowState, minimize: Optional[ValueOutputExpression] = None, is_unit_cost: bool = False):
        self.executor = executor
        self.state = state.clone()
        self.minimize = minimize
        self.initial_cost = self.get_cost()
        self.is_unit_cost = is_unit_cost

        if not self.is_unit_cost:
            assert self.minimize is not None, 'minimize should be provided when is_unit_cost is False.'

    def get_cost(self) -> float:
        if self.minimize is None:
            return 0

        return self.executor.execute(self.minimize, state=self.state).item()

    def run(self, stmt: Union[CrowAchieveExpression, CrowBehaviorApplicationExpression, CrowControllerApplicationExpression], scope: dict) -> float:
        scope = {k: v for k, v in scope.items() if not (k.startswith('__') and k.endswith('__'))}
        if isinstance(stmt, CrowControllerApplicationExpression):
            if self.minimize is None:
                return 1.0

            if stmt.controller.effect_body is None:
                return 0.0

            current_cost = self.get_cost()
            bounded_variables = {k.name: self.evaluate(v, state=self.state, csp=None, bounded_variables=scope, clone_csp=False, force_tensor_value=True)[0] for k, v in zip(stmt.controller.arguments, stmt.arguments)}
            if any(isinstance(v, Variable) for v in bounded_variables.values()):
                return 0.0
            execute_effect_statements(self.executor, stmt.controller.effect_body.statements, self.state, scope=bounded_variables)
            return self.get_cost() - current_cost

        if isinstance(stmt, CrowAchieveExpression):
            # Case 1: The variable is not fully grounded.
            for var in find_free_variables(stmt.goal):
                if var.name not in scope:
                    return 0.0

            # Case 2: The goal is already achieved.
            rv, _ = self.evaluate(stmt.goal, self.state, csp=None, bounded_variables=scope, clone_csp=False, force_tensor_value=False)
            if isinstance(rv, OptimisticValue) or bool(rv):
                return 0.0

        # Case 3: Now we want to find a way to achieve the goal.
        if isinstance(stmt, CrowAchieveExpression):
            behavior_matches = match_applicable_behaviors(self.executor.domain, self.state, stmt.goal, scope)
        elif isinstance(stmt, CrowBehaviorApplicationExpression):
            # TODO(Jiayuan Mao @ 2024/08/11): handle CSP correctly.
            behavior_matches = [ApplicableBehaviorItem(stmt.behavior, {
                k.name: self.evaluate(v, state=self.state, csp=None, bounded_variables=scope, clone_csp=False, force_tensor_value=True)[0]
                for k, v in zip(stmt.behavior.arguments, stmt.arguments)
            })]
        else:
            raise ValueError(f'Unsupported statement type: {type(stmt)}')

        if len(behavior_matches) == 0:
            return 0.0

        need_backtrack = (len(behavior_matches) > 1)
        best_behavior_cost = None
        best_behavior_cost_state = None

        current_cost = self.get_cost()
        state_backup = None
        for behavior_match in behavior_matches:
            if need_backtrack:
                state_backup = self.state.clone()

            # Check if the behavior can be grounded.
            skip_evaluation = False
            if any(isinstance(v, Variable) for v in behavior_match.bounded_variables.values()):
                skip_evaluation = True
            if skip_evaluation:
                best_behavior_cost = 0.0
                continue

            # Not really grounded.
            if behavior_match.behavior.heuristic is None:
                # If there is a behavior that can achieve the goal but it does not have an associated heuristic value, we do a "auto" evaluation.
                if self.minimize is not None:
                    for stmt in behavior_match.behavior.body.statements:
                        if isinstance(stmt, CrowAchieveExpression):
                            self.run(stmt, behavior_match.bounded_variables)
                        elif isinstance(stmt, CrowBehaviorApplicationExpression):
                            self.run(stmt, behavior_match.bounded_variables)
                        elif isinstance(stmt, CrowControllerApplicationExpression):
                            self.run(stmt, behavior_match.bounded_variables)
                    behavior_cost = self.get_cost() - current_cost
                else:
                    behavior_cost = 0
                    for stmt in behavior_match.behavior.body.statements:
                        if isinstance(stmt, CrowAchieveExpression):
                            behavior_cost += self.run(stmt, behavior_match.bounded_variables)
                        elif isinstance(stmt, CrowBehaviorApplicationExpression):
                            behavior_cost += self.run(stmt, behavior_match.bounded_variables)
                        elif isinstance(stmt, CrowControllerApplicationExpression):
                            behavior_cost += 1
            else:
                # Run the heuristic program.
                for stmt in behavior_match.behavior.heuristic.statements:
                    if isinstance(stmt, CrowAchieveExpression):
                        self.run(stmt, behavior_match.bounded_variables)
                    elif isinstance(stmt, CrowBehaviorApplicationExpression):
                        self.run(stmt, behavior_match.bounded_variables)
                    elif isinstance(stmt, CrowFeatureAssignmentExpression):
                        execute_effect_statements(self.executor, [stmt], self.state, scope=behavior_match.bounded_variables)
                    else:
                        raise ValueError(f'Unsupported statement type: {type(stmt)}')

                behavior_cost = self.get_cost() - current_cost

            if best_behavior_cost is None or behavior_cost < best_behavior_cost:
                best_behavior_cost = behavior_cost
                if need_backtrack:
                    best_behavior_cost_state = self.state.clone()

            if need_backtrack:
                self.state = state_backup

        if need_backtrack:
            self.state = best_behavior_cost_state
        return best_behavior_cost

    def evaluate(
        self, expression: Union[ObjectOrValueOutputExpression, VariableAssignmentExpression], state: CrowState, csp: Optional[ConstraintSatisfactionProblem] = None,
        bounded_variables: Optional[Dict[str, Union[TensorValue, ObjectConstant]]] = None,
        clone_csp: bool = True,
        force_tensor_value: bool = False
    ) -> Tuple[Union[None, StateObjectReference, StateObjectList, TensorValue, OptimisticValue], Optional[ConstraintSatisfactionProblem]]:
        """Evaluate an expression and return the result.

        Args:
            expression: the expression to evaluate.
            state: the current state.
            csp: the current CSP.
            bounded_variables: the bounded variables.
            clone_csp: whether to clone the CSP.
            force_tensor_value: whether to force the result to be a tensor value.

        Returns:
            the evaluation result and the updated CSP.
        """
        if clone_csp:
            csp = csp.clone() if csp is not None else None
        if bounded_variables is not None:
            bounded_variables = {k: v for k, v in bounded_variables.items() if not (k.startswith('__') and k.endswith('__'))}

        if isinstance(expression, VariableAssignmentExpression):
            self.executor.execute(expression, state=state, csp=csp, bounded_variables=bounded_variables)
            return None, csp

        rv = self.executor.execute(expression, state=state, csp=csp, bounded_variables=bounded_variables)
        if isinstance(rv, TensorValue):
            if force_tensor_value:
                return rv, csp
            if rv.is_scalar:
                return rv.item(), csp
            return rv, csp

        if isinstance(rv, ListValue) and len(rv.values) > 0:
            if isinstance(rv.values[0], StateObjectReference):
                rv = StateObjectList(rv.dtype, rv.values)

        assert isinstance(rv, StateObjectReference) or isinstance(rv, StateObjectList) or isinstance(rv, ListValue)
        return rv, csp


def execute_additive_heuristic_program(
    executor: CrowExecutor, stmt: Union[CrowAchieveExpression, CrowBehaviorApplicationExpression, CrowControllerApplicationExpression], state: CrowState, scope: dict, minimize: Optional[ValueOutputExpression] = None,
    is_unit_cost: bool = False
) -> float:
    program_executor = CrowAdditiveHeuristicProgramExecutor(executor, state, minimize=minimize, is_unit_cost=is_unit_cost)
    return program_executor.run(stmt, scope)

