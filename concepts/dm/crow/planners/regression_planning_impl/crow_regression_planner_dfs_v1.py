#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : crow_regression_planner_dfs_v1.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/11/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import jacinle

from typing import Optional, Union, Sequence, Tuple, List, Dict, NamedTuple
from concepts.dsl.constraint import ConstraintSatisfactionProblem
from concepts.dsl.dsl_types import Variable
from concepts.dsl.expression import ValueOutputExpression
from concepts.dsl.tensor_state import StateObjectReference

from concepts.dm.crow.controller import CrowControllerApplier, CrowControllerApplicationExpression
from concepts.dm.crow.behavior import CrowAchieveExpression, CrowBindExpression, CrowRuntimeAssignmentExpression, CrowAssertExpression
from concepts.dm.crow.behavior import CrowBehaviorStatementOrdering, CrowBehaviorOrderingSuite, CrowBehaviorApplicationExpression
from concepts.dm.crow.behavior_utils import match_applicable_behaviors, execute_object_bind

from concepts.dm.crow.planners.regression_planning import CrowPlanningResult, CrowRegressionPlanner, ScopedCrowExpression
from concepts.dm.crow.planners.regression_utils import canonicalize_bounded_variables
from concepts.dm.crow.planners.regression_planning_impl.crow_regression_planner_dfs_v1_utils import execute_behavior_effect_batch, unique_results

__all__ = ['CrowRegressionPlannerDFSv1']


class _DFSNode(NamedTuple):
    program: Optional[CrowBehaviorOrderingSuite]
    scopes: Dict[int, dict]
    latest_scope: int
    right_statements: List[ScopedCrowExpression]
    depth: int = 0

    def __lt__(self, other):
        return len(self.right_statements) < len(other.right_statements)


class CrowRegressionPlannerDFSv1(CrowRegressionPlanner):
    def _post_init(self, max_search_depth: int = 25):
        self.max_search_depth = max_search_depth

    def main_entry(self, program: CrowBehaviorOrderingSuite, minimize: Optional[ValueOutputExpression] = None) -> List[Tuple[CrowControllerApplier, ...]]:
        state = _DFSNode(program, scopes={0: dict()}, right_statements=[], latest_scope=0)
        candidate_plans = self.bfs(state)
        candidate_plans = [result.controller_actions for result in candidate_plans]
        return candidate_plans

    @jacinle.log_function(verbose=False)
    def dfs(self, state: _DFSNode) -> Sequence[CrowPlanningResult]:
        jacinle.log_function.print('Program:', state.program, 'Depth:', state.depth)
        if state.depth >= self.max_search_depth:
            raise RuntimeError('Maximum search depth reached.')
        right_statements_str = ' '.join([f'{str(x)}@{i}' for x, i in state.right_statements])
        jacinle.log_function.print(jacinle.colored(right_statements_str, 'yellow'), state.scopes)
        new_results = list()
        for left, stmt, scope_id in state.program.pop_right_statement():
            new_results.extend(self._dfs_expand(state, left, stmt, scope_id))
        # jacinle.log_function.print('New results:', [x.controller_appliers for x in new_results])
        return unique_results(new_results)

    def _dfs_expand(self, state: _DFSNode, left: CrowBehaviorOrderingSuite, stmt: Union[CrowAchieveExpression, CrowControllerApplicationExpression], scope_id: int) -> Sequence[CrowPlanningResult]:
        """Expand a statement.

        Args:
            state: the current search state.
            left: the left part of the program.
            stmt: the statement to be expanded.
            scope_id: the current scope id.
        """
        # jacinle.log_function.print('Expanding:', stmt, 'with left:', left, 'and scope', state.scopes[scope_id])
        new_results = list()
        if isinstance(stmt, CrowAchieveExpression):
            jacinle.log_function.print('Expanding:', stmt, 'with left:', left, 'and scope_id:', scope_id)
            jacinle.log_function.print('Null goal expansion', stmt.goal)
            # Branch 1: the goal is directly satisfied at the after expanding the left part (null production).
            if left is None:
                last_results = [CrowPlanningResult(self.state, ConstraintSatisfactionProblem() if self.enable_csp else None, tuple(), state.scopes)]
            else:
                last_results = self.dfs(_DFSNode(left, state.scopes, state.latest_scope, [ScopedCrowExpression(CrowAssertExpression(stmt.goal), scope_id)] + state.right_statements, state.depth + 1))
            for last_result in last_results:
                rv = self.executor.execute(stmt.goal, state=last_result.state, bounded_variables=canonicalize_bounded_variables(last_result.scopes, scope_id))
                jacinle.log_function.print('Last result:', last_result.controller_actions, 'Null Eval=', bool(rv.item()))
                if bool(rv.item()):
                    new_results.append(last_result)
            if len(new_results) > 0:
                return new_results

            # Branch 2: the goal is not directly satisfied. We need to refine the goal.
            all_matching = match_applicable_behaviors(self.executor.domain, self.state, stmt.goal, state.scopes[scope_id])
            all_matching = list(all_matching)
            matchings_str = [(x.behavior.name, x.bounded_variables) for x in all_matching]
            jacinle.log_function.print(jacinle.tabulate(matchings_str, headers=['Behavior', 'Bounded Variables']))
            for behavior_matching in all_matching:
                bounded_variables = behavior_matching.bounded_variables
                for var, value in bounded_variables.items():
                    if isinstance(value, Variable):
                        bounded_variables[var] = value.clone_with_scope(scope_id)

                # Create a new scope for this subgoal refinement.
                new_scope_id = state.latest_scope + 1
                new_scopes = state.scopes.copy()
                new_scopes[new_scope_id] = bounded_variables.copy()
                program = behavior_matching.behavior.assign_body_program_scope(new_scope_id)

                # Branch 2.1: the goal will be refined by ignoring the promotable part.
                # Create a new program by sequencing the left part and the refinement of this behavior.
                if left is None:
                    new_program = CrowBehaviorOrderingSuite(CrowBehaviorStatementOrdering.SEQUENTIAL, program.get_flatten_body(), variable_scope_identifier=new_scope_id)
                else:
                    new_program = CrowBehaviorOrderingSuite(CrowBehaviorStatementOrdering.SEQUENTIAL, (left,) + program.get_flatten_body(), variable_scope_identifier=new_scope_id)

                # Now recursively calls the DFS to expand the new program.
                # import ipdb; ipdb.set_trace()
                this_new_results = self.dfs(_DFSNode(new_program, new_scopes, new_scope_id, [ScopedCrowExpression(behavior_matching.behavior, new_scope_id)] + state.right_statements, state.depth + 1))
                if len(this_new_results) > 0:
                    new_results.extend(execute_behavior_effect_batch(self.executor, this_new_results, behavior_matching.behavior, new_scope_id))
                    continue

                # Branch 2.2: If the sequential expansion fails, we try to expand the unordered part.
                if self.enable_reordering:
                    promotable, sequential_body = program.split_promotable()
                    if left is not None and promotable is not None:
                        new_program = CrowBehaviorOrderingSuite(
                            CrowBehaviorStatementOrdering.SEQUENTIAL,
                            (CrowBehaviorOrderingSuite(CrowBehaviorStatementOrdering.UNORDERED, (left,) + promotable, variable_scope_identifier=new_scope_id), sequential_body),
                            variable_scope_identifier=new_scope_id
                        )
                        this_new_results = self.dfs(_DFSNode(new_program, new_scopes, new_scope_id, [ScopedCrowExpression(behavior_matching.behavior, new_scope_id)] + state.right_statements, state.depth + 1))
                        if len(this_new_results) > 0:
                            new_results.extend(execute_behavior_effect_batch(self.executor, this_new_results, behavior_matching.behavior, new_scope_id))
            return new_results
        elif isinstance(stmt, CrowBehaviorApplicationExpression):
            pass
        elif isinstance(stmt, (CrowBindExpression, CrowRuntimeAssignmentExpression, CrowAssertExpression, CrowControllerApplicationExpression)):
            # These statements are "primitive" or "atomic". Therefore, all we need to do is to expand the left branch and then apply the primitive.
            if left is None:
                last_results = [CrowPlanningResult(self.state, ConstraintSatisfactionProblem() if self.enable_csp else None, tuple(), state.scopes)]
            else:
                last_results = self.dfs(_DFSNode(left, state.scopes, state.latest_scope, [ScopedCrowExpression(stmt, scope_id)] + state.right_statements, state.depth + 1))
            for last_result in last_results:
                new_results.extend(self._dfs_expand_primitive(last_result, stmt, scope_id))
            return new_results
        else:
            raise ValueError(f'Unknown statement type: {stmt}')

    def _dfs_expand_primitive(self, last_result: CrowPlanningResult, stmt: Union[CrowBindExpression, CrowRuntimeAssignmentExpression, CrowAssertExpression, CrowControllerApplicationExpression], scope_id: int) -> Sequence[CrowPlanningResult]:
        """Apply a primitive statement on top of a particular planning result of the left branch."""
        if isinstance(stmt, CrowControllerApplicationExpression):
            argument_values = [self.executor.execute(x, state=last_result.state, bounded_variables=canonicalize_bounded_variables(last_result.scopes, scope_id)) for x in stmt.arguments]
            for i, argv in enumerate(argument_values):
                if isinstance(argv, StateObjectReference):
                    argument_values[i] = argv.name
            return [CrowPlanningResult(last_result.state, last_result.csp, last_result.controller_actions + (CrowControllerApplier(stmt.controller, argument_values),), last_result.scopes)]
        elif isinstance(stmt, CrowBindExpression):
            if stmt.is_object_bind:
                new_results = list()
                for new_scope in execute_object_bind(self.executor, stmt, last_result.state, canonicalize_bounded_variables(last_result.scopes, scope_id)):
                    new_scopes = last_result.scopes.copy()
                    new_scopes[scope_id] = new_scope
                    new_results.append(CrowPlanningResult(last_result.state, last_result.csp, last_result.controller_actions, new_scopes))
                jacinle.log_function.print(f'Object binding results for {stmt} under scope {canonicalize_bounded_variables(last_result.scopes, scope_id)}:', len(new_results), 'states.')
                return new_results
            else:
                raise NotImplementedError()
        elif isinstance(stmt, CrowRuntimeAssignmentExpression):
            raise NotImplementedError()
        elif isinstance(stmt, CrowAssertExpression):
            # TODO(Jiayuan Mao @ 2024/03/19): implement the CSP tracking.
            rv = self.executor.execute(stmt.bool_expr, state=last_result.state, bounded_variables=canonicalize_bounded_variables(last_result.scopes, scope_id))
            jacinle.log_function.print(f'Assert {stmt.bool_expr} under scope {canonicalize_bounded_variables(last_result.scopes, scope_id)}:', bool(rv.item()))
            if bool(rv.item()):
                return [last_result]
            else:
                return []
        else:
            raise ValueError(f'Unknown statement type: {stmt}')
