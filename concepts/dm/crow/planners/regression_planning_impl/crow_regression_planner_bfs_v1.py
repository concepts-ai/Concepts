#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : crow_regression_planner_bfs_v1.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/21/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import queue
import jacinle

from typing import Optional, Union, Iterator, Sequence, Tuple, List, Dict, NamedTuple
from concepts.dsl.constraint import ConstraintSatisfactionProblem
from concepts.dsl.dsl_types import Variable
from concepts.dsl.expression import ValueOutputExpression
from concepts.dsl.tensor_state import StateObjectReference

from concepts.dm.crow.controller import CrowControllerApplier, CrowControllerApplicationExpression
from concepts.dm.crow.behavior import CrowBehavior, CrowAchieveExpression, CrowBindExpression, CrowRuntimeAssignmentExpression, CrowAssertExpression
from concepts.dm.crow.behavior import CrowBehaviorStatementOrdering, CrowBehaviorOrderingSuite
from concepts.dm.crow.behavior_utils import match_applicable_behaviors, execute_object_bind
from concepts.dm.crow.executors.crow_executor import CrowExecutor

from concepts.dm.crow.planners.regression_planning import CrowPlanningResult, CrowRegressionPlanner, ScopedCrowExpression
from concepts.dm.crow.planners.regression_utils import canonize_bounded_variables, split_simple_sequential
from concepts.dm.crow.planners.regression_planning_impl.crow_regression_planner_dfs_v1_utils import execute_behavior_effect

__all__ = ['CrowRegressionPlannerBFSv1']


class _BFSNode(NamedTuple):
    program: Optional[CrowBehaviorOrderingSuite]
    scopes: Dict[int, dict]
    latest_scope: int
    right_statements: List[ScopedCrowExpression]
    depth: int = 0

    def __lt__(self, other):
        return len(self.right_statements) < len(other.right_statements)


LOG_BFS_GRAPH = False


class CrowRegressionPlannerBFSv1(CrowRegressionPlanner):
    def _post_init(self, enable_reordering: bool = True):
        self.enable_reordering = enable_reordering

    def main_entry(self, program: CrowBehaviorOrderingSuite, minimize: Optional[ValueOutputExpression] = None) -> List[Tuple[CrowControllerApplier, ...]]:
        state = _BFSNode(program, scopes={0: dict()}, right_statements=[], latest_scope=0)
        candidate_plans = self.bfs(state)
        candidate_plans = [result.controller_actions for result in candidate_plans]
        return candidate_plans

    def bfs(self, state: _BFSNode) -> Sequence[CrowPlanningResult]:
        q = queue.PriorityQueue()
        q.put(state)

        graph = None
        if LOG_BFS_GRAPH:
            graph = dict(nodes=dict(), edges=list())
            graph['nodes'][id(state)] = state

        while not q.empty():
            self._search_stat['nr_expanded_nodes'] += 1
            if self._search_stat['nr_expanded_nodes'] > 1000000:
                import ipdb; ipdb.set_trace()
                raise RuntimeError('Too many nodes expanded.')
            if self._search_stat['nr_expanded_nodes'] % 1000 == 0:
                jacinle.log_function.print('Expanded nodes:', self._search_stat['nr_expanded_nodes'])

            current_state = q.get()
            # print('Current state:', current_state.right_statements)
            for left, stmt, scope_id in current_state.program.pop_right_statement():
                for i, new_state in enumerate(self._bfs_expand(current_state, left, stmt, scope_id)):
                    if LOG_BFS_GRAPH:
                        graph['nodes'][id(new_state)] = new_state
                        graph['edges'].append((id(current_state), id(new_state), f'{stmt}@{i}'))
                    if new_state.program is None:
                        if (result := self._bfs_verify(new_state)) is not None:
                            return [result]
                    else:
                        # jacinle.log_function.print('  New state:', new_state.right_statements)
                        q.put(new_state)
        return list()

    def _bfs_expand(self, state: _BFSNode, left: CrowBehaviorOrderingSuite, stmt: Union[CrowAchieveExpression, CrowControllerApplicationExpression], scope_id: int) -> Iterator[_BFSNode]:
        if isinstance(stmt, CrowAchieveExpression):
            if left is None:
                yield _BFSNode(None, state.scopes, state.latest_scope, [ScopedCrowExpression(CrowAssertExpression(stmt.goal), scope_id)] + state.right_statements, state.depth + 1)
            else:
                yield _BFSNode(left, state.scopes, state.latest_scope, [ScopedCrowExpression(CrowAssertExpression(stmt.goal), scope_id)] + state.right_statements, state.depth + 1)

            all_matching = match_applicable_behaviors(self.executor.domain, self.state, stmt.goal, state.scopes[scope_id])
            all_matching = list(all_matching)
            for behavior_matching in all_matching:
                # print('Behavior matching:', behavior_matching.behavior.name, behavior_matching.bounded_variables)
                # if 'hand' in behavior_matching.behavior.name:
                #     import ipdb; ipdb.set_trace()
                #     pass
                bounded_variables = behavior_matching.bounded_variables
                for var, value in bounded_variables.items():
                    if isinstance(value, Variable):
                        bounded_variables[var] = value.clone_with_scope(scope_id)
                new_scope_id = state.latest_scope + 1
                new_scopes = state.scopes.copy()
                new_scopes[new_scope_id] = bounded_variables.copy()
                program = behavior_matching.behavior.assign_body_program_scope(new_scope_id)

                complex_part, simple_part = split_simple_sequential(program.get_flatten_body(), new_scope_id)
                if left is None:
                    if len(complex_part) == 0:
                        new_program = None
                    else:
                        new_program = CrowBehaviorOrderingSuite(CrowBehaviorStatementOrdering.SEQUENTIAL, complex_part, variable_scope_identifier=new_scope_id)
                else:
                    new_program = CrowBehaviorOrderingSuite(CrowBehaviorStatementOrdering.SEQUENTIAL, (left,) + program.get_flatten_body(), variable_scope_identifier=new_scope_id)
                yield _BFSNode(new_program, new_scopes, new_scope_id, simple_part + [ScopedCrowExpression(behavior_matching.behavior, new_scope_id)] + state.right_statements, state.depth + 1)

                if self.enable_reordering:
                    promotable, sequential_body = program.split_promotable()
                    complex_seq, simple_seq = split_simple_sequential(sequential_body, new_scope_id)
                    if left is not None and promotable is not None:
                        new_program = CrowBehaviorOrderingSuite(CrowBehaviorStatementOrdering.SEQUENTIAL, (
                            CrowBehaviorOrderingSuite(CrowBehaviorStatementOrdering.UNORDERED, (
                                left,
                                CrowBehaviorOrderingSuite(CrowBehaviorStatementOrdering.SEQUENTIAL, promotable, variable_scope_identifier=new_scope_id)
                            ), variable_scope_identifier=new_scope_id),
                            *complex_seq
                        ), variable_scope_identifier=new_scope_id)
                        yield _BFSNode(new_program, new_scopes, new_scope_id, simple_seq + [ScopedCrowExpression(behavior_matching.behavior, new_scope_id)] + state.right_statements, state.depth + 1)
        elif isinstance(stmt, (CrowBindExpression, CrowRuntimeAssignmentExpression, CrowAssertExpression, CrowControllerApplicationExpression)):
            if left is None:
                yield _BFSNode(None, state.scopes, state.latest_scope, [ScopedCrowExpression(stmt, scope_id)] + state.right_statements, state.depth + 1)
            else:
                yield _BFSNode(left, state.scopes, state.latest_scope, [ScopedCrowExpression(stmt, scope_id)] + state.right_statements, state.depth + 1)
        else:
            raise ValueError(f'Unknown statement type: {stmt}')

    def _bfs_verify(self, state: _BFSNode) -> Optional[CrowPlanningResult]:
        init_results = [CrowPlanningResult(self.state, ConstraintSatisfactionProblem() if self.enable_csp else None, tuple(), state.scopes)]
        results = execute_statements(self.executor, init_results, state.right_statements)

        if len(results) > 0:
            return results[0]
        else:
            return None


def execute_statements(executor: CrowExecutor, init_results: Sequence[CrowPlanningResult], statements: Sequence[ScopedCrowExpression]) -> List[CrowPlanningResult]:
    # statement_str = ' '.join([f'{str(x)}@{i}' for x, i in state.right_statements])
    # print(jacinle.colored(statement_str, 'yellow'), state.scopes)

    results = init_results
    for statement in statements:
        stmt, scope_id = statement.statement, statement.scope_id
        new_results = list()

        try:
            # Invalid ordering of find statement.
            bounded_variables = canonize_bounded_variables(results[0].scopes, scope_id)
        except KeyError:
            return list()

        if isinstance(stmt, CrowAssertExpression):
            for result in results:
                rv = executor.execute(stmt.bool_expr, state=result.state, bounded_variables=canonize_bounded_variables(result.scopes, scope_id))
                if bool(rv.item()):
                    new_results.append(result)
        elif isinstance(stmt, CrowControllerApplicationExpression):
            for result in results:
                argument_values = [executor.execute(x, state=result.state, bounded_variables=canonize_bounded_variables(result.scopes, scope_id)) for x in stmt.arguments]
                for i, argv in enumerate(argument_values):
                    if isinstance(argv, StateObjectReference):
                        argument_values[i] = argv.name
                new_results.append(CrowPlanningResult(result.state, result.csp, result.controller_actions + (CrowControllerApplier(stmt.controller, argument_values),), result.scopes))
        elif isinstance(stmt, CrowBindExpression):
            for result in results:
                for new_scope in execute_object_bind(executor, stmt, result.state, canonize_bounded_variables(result.scopes, scope_id)):
                    # print('!!!Object binding result:', new_scope)
                    new_scopes = result.scopes.copy()
                    new_scopes[scope_id] = new_scope
                    new_results.append(CrowPlanningResult(result.state, result.csp, result.controller_actions, new_scopes))
        elif isinstance(stmt, CrowBehavior):
            new_results = [execute_behavior_effect(executor, stmt, result.state, canonize_bounded_variables(result.scopes, scope_id)) for result in results]
        else:
            raise ValueError(f'Unknown statement type: {stmt}')
        results = new_results

        if len(results) == 0:
            break

    return results

