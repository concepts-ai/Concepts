#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : crow_regression_planner_dfs_v2.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/21/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import jacinle

from typing import Optional, Union, Sequence, Tuple, List, Dict, NamedTuple
from concepts.dsl.constraint import ConstraintSatisfactionProblem
from concepts.dsl.dsl_types import Variable
from concepts.dsl.expression import ValueOutputExpression
from concepts.dsl.tensor_state import StateObjectReference

from concepts.dm.crow.crow_domain import CrowState
from concepts.dm.crow.controller import CrowControllerApplier, CrowControllerApplicationExpression
from concepts.dm.crow.behavior import CrowAchieveExpression, CrowBindExpression, CrowRuntimeAssignmentExpression, CrowAssertExpression
from concepts.dm.crow.behavior import CrowBehaviorOrderingSuite, CrowBehaviorApplicationExpression
from concepts.dm.crow.behavior_utils import match_applicable_behaviors, ApplicableBehaviorItem, execute_object_bind, format_behavior_program

from concepts.dm.crow.planners.regression_planning import CrowPlanningResult, CrowRegressionPlanner, ScopedCrowExpression
from concepts.dm.crow.planners.regression_utils import canonize_bounded_variables
from concepts.dm.crow.planners.regression_planning_impl.crow_regression_planner_dfs_v1_utils import execute_behavior_effect_batch, unique_results


class _DFSNode(NamedTuple):
    program: CrowBehaviorOrderingSuite
    state: CrowState
    csp: ConstraintSatisfactionProblem
    scopes: Dict[int, dict]
    latest_scope: int
    left_actions: Tuple[CrowControllerApplier, ...]
    right_statements: Optional[List[ScopedCrowExpression]] = None
    allow_promotable: bool = True
    depth: int = 0


class CrowRegressionPlannerDFSv2(CrowRegressionPlanner):
    def _post_init(self, max_search_depth: int = 25):
        self.max_search_depth = max_search_depth

    def main_entry(self, program: CrowBehaviorOrderingSuite, minimize: Optional[ValueOutputExpression] = None) -> List[Tuple[CrowControllerApplier, ...]]:
        state = _DFSNode(program, self.state, ConstraintSatisfactionProblem(), {0: {}}, 0, tuple(), list(), allow_promotable=True, depth=0)
        results = self.dfs(state)
        return [x.controller_actions for x in results]

    @jacinle.log_function(verbose=False)
    def dfs(self, state: _DFSNode) -> Sequence[CrowPlanningResult]:
        if state.depth >= self.max_search_depth:
            print('Max depth reached.')
            # import ipdb; ipdb.set_trace()
            return []

        # left_actions_str = ' '.join([str(x) for x in state.left_actions]) if len(state.left_actions) > 0 else '<empty left>'
        # right_statements_str = ' '.join([f'{str(x)}@{i}' for x, i in state.right_statements]) if len(state.right_statements) > 0 else '<empty right>'
        # jacinle.log_function.print(
        #     'LEFT:  ' + jacinle.colored(left_actions_str, 'green'),
        #     'RIGHT: ' + jacinle.colored(right_statements_str, 'yellow'),
        #     'SCOPE: ' + str(state.scopes),
        #     sep='\n'
        # )
        # jacinle.log_function.print('Program:', state.program)
        # jacinle.log_function.print('Scopes:', state.scopes)
        # jacinle.log_function.print('Depth', state.depth)

        # def _is_sequential_program(program: CrowActionOrderingSuite) -> bool:
        #     if program.order is not CrowActionOrderingSuite.ORDER.SEQUENTIAL:
        #         return False
        #     for stmt in program.statements:
        #         if isinstance(stmt, CrowActionOrderingSuite):
        #             if not _is_sequential_program(stmt):
        #                 return False
        #     return True
        #
        # if not _is_sequential_program(state.program):
        #     print('Non-sequential program.')
        #     import ipdb; ipdb.set_trace()
        #     pass

        new_results = list()
        for left, stmt, scope_id in state.program.pop_right_statement():
            new_results.extend(self._dfs_expand(state, left, stmt, scope_id))
        # jacinle.log_function.print('New results:', [x.actions for x in new_results])
        return unique_results(new_results)

    def _dfs_expand(self, state: _DFSNode, left: CrowBehaviorOrderingSuite, stmt: Union[CrowAchieveExpression, CrowControllerApplicationExpression], scope_id: int) -> Sequence[CrowPlanningResult]:
        """Expand a statement.

        Args:
            state: the current search state.
            left: the left part of the program.
            stmt: the statement to be expanded.
            scope_id: the current scope id.
        """

        new_results = list()
        if isinstance(stmt, CrowAchieveExpression):
            jacinle.log_function.print('Expanding:', stmt, 'with left:', format_behavior_program(left, state.scopes) if left is not None else None, 'and scope_id:', scope_id)
            # jacinle.log_function.print('Null goal expansion', stmt.goal)
            # Branch 1: the goal is directly satisfied at the after expanding the left part (null production).
            if left is None:
                last_results = [CrowPlanningResult(state.state, state.csp, state.left_actions, state.scopes)]
            else:
                last_results = self.dfs(_DFSNode(
                    left, state.state, state.csp, state.scopes, state.latest_scope,
                    state.left_actions, [ScopedCrowExpression(CrowAchieveExpression(stmt.goal, once=stmt.once), scope_id)] + state.right_statements,
                    allow_promotable=state.allow_promotable, depth=state.depth + 1
                ))
            for last_result in last_results:
                rv = self.executor.execute(stmt.goal, state=last_result.state, bounded_variables=canonize_bounded_variables(last_result.scopes, scope_id))
                # jacinle.log_function.print('Last result:', last_result.actions, 'Null Eval =', bool(rv.item()))
                if bool(rv.item()):
                    new_results.append(last_result)

            if len(new_results) > 0:
                return new_results

            # Branch 2: the goal is not directly satisfied. We need to refine the goal.
            all_matching = match_applicable_behaviors(self.executor.domain, self.state, stmt.goal, state.scopes[scope_id])
            all_matching = list(all_matching)
            # matchings_str = [(x.action.name, x.bounded_variables) for x in all_matching]
            # jacinle.log_function.print(jacinle.tabulate(matchings_str, headers=['Action', 'Bounded Variables']))
            for action_matching in all_matching:
                # jacinle.log_function.print('Trying action:', action_matching.action, 'with bounded variables:', action_matching.bounded_variables)
                # if action_matching.action.name == 'r_clear_from_holding':
                #     import ipdb; ipdb.set_trace()
                new_results.extend(self._dfs_expand_action(state, last_results, left, action_matching, scope_id))
        elif isinstance(stmt, CrowBehaviorApplicationExpression):
            if left is None:
                last_results = [CrowPlanningResult(state.state, state.csp, state.left_actions, state.scopes)]
            else:
                last_results = self.dfs(_DFSNode(
                    left, state.state, state.csp, state.scopes, state.latest_scope,
                    state.left_actions, [ScopedCrowExpression(stmt, scope_id)] + state.right_statements,
                    allow_promotable=state.allow_promotable, depth=state.depth
                ))

            for last_result in last_results:
                argument_values = [self.executor.execute(x, state=last_result.state, bounded_variables=canonize_bounded_variables(last_result.scopes, scope_id)) for x in stmt.arguments]
                action_matching = ApplicableBehaviorItem(stmt.behavior, {k.name: v for k, v in zip(stmt.behavior.arguments, argument_values)})
                new_results.extend(self._dfs_expand_action(state, [last_result], left, action_matching, scope_id))
        elif isinstance(stmt, (CrowBindExpression, CrowRuntimeAssignmentExpression, CrowAssertExpression, CrowControllerApplicationExpression)):
            # These statements are "primitive" or "atomic". Therefore, all we need to do is to expand the left branch and then apply the primitive.
            if left is None:
                last_results = [CrowPlanningResult(state.state, state.csp, state.left_actions, state.scopes)]
            else:
                last_results = self.dfs(_DFSNode(
                    left, state.state, state.csp, state.scopes, state.latest_scope,
                    state.left_actions, [ScopedCrowExpression(stmt, scope_id)] + state.right_statements,
                    allow_promotable=state.allow_promotable, depth=state.depth
                ))
            for last_result in last_results:
                new_results.extend(self._dfs_expand_primitive(last_result, stmt, scope_id))
        else:
            raise ValueError(f'Unknown statement type: {stmt}')
        return new_results

    def _dfs_expand_action(self, state: _DFSNode, last_results: Sequence[CrowPlanningResult], left: Optional[CrowBehaviorOrderingSuite], action_matching: ApplicableBehaviorItem, scope_id: int) -> Sequence[CrowPlanningResult]:
        new_results = list()
        bounded_variables = action_matching.bounded_variables
        for var, value in bounded_variables.items():
            if isinstance(value, Variable):
                bounded_variables[var] = value.clone_with_scope(scope_id)

        # Create a new scope for this subgoal refinement.
        new_scope_id = state.latest_scope + 1
        program = action_matching.behavior.assign_body_program_scope(new_scope_id)
        preamble, promotable, rest = program.split_preamble_and_promotable()

        new_scopes = state.scopes.copy()
        new_scopes[new_scope_id] = bounded_variables.copy()
        if preamble is None:
            last_results = [CrowPlanningResult(state.state, state.csp, state.left_actions, new_scopes)]
        else:
            last_results = self.dfs(_DFSNode(
                CrowBehaviorOrderingSuite.make_sequential(preamble, variable_scope_identifier=new_scope_id),
                state.state, state.csp, new_scopes, new_scope_id, state.left_actions, [], allow_promotable=False, depth=state.depth + 1
            ))

        if len(last_results) == 0:
            return []

        new_promotable_results = list()
        if promotable is None:
            if left is None:
                new_promotable_results = last_results
            else:
                for last_result in last_results:
                    new_promotable_results.extend(self.dfs(_DFSNode(
                        left, last_result.state, last_result.csp, last_result.scopes, new_scope_id, last_result.controller_actions, [ScopedCrowExpression(rest, scope_id)] + state.right_statements,
                        allow_promotable=True, depth=state.depth + 1
                    )))
        else:
            if left is None:
                program = CrowBehaviorOrderingSuite.make_sequential(promotable, variable_scope_identifier=new_scope_id)
            else:
                program = CrowBehaviorOrderingSuite.make_unordered(left, CrowBehaviorOrderingSuite.make_sequential(promotable, variable_scope_identifier=new_scope_id))
                jacinle.log_function.print('Making unordered program!')
                jacinle.log_function.print(format_behavior_program(program, last_results[0].scopes))
                # import ipdb; ipdb.set_trace()
                # pass
            right_program = [ScopedCrowExpression(rest, scope_id), ScopedCrowExpression(action_matching.behavior, scope_id)] + state.right_statements
            for last_result in last_results:
                results = self.dfs(_DFSNode(
                    program, last_result.state, last_result.csp, last_result.scopes, new_scope_id, last_result.controller_actions, right_program,
                    allow_promotable=True, depth=state.depth + 1
                ))
                new_promotable_results.extend(results)

        if len(new_promotable_results) == 0:
            return []

        right_program = [ScopedCrowExpression(action_matching.behavior, scope_id)] + state.right_statements
        for last_result in new_promotable_results:
            results = self.dfs(_DFSNode(
                CrowBehaviorOrderingSuite.make_sequential(rest, variable_scope_identifier=new_scope_id),
                last_result.state, last_result.csp, last_result.scopes, new_scope_id, last_result.controller_actions, right_program,
                allow_promotable=True, depth=state.depth + 1
            ))
            new_results.extend(execute_behavior_effect_batch(self.executor, results, action_matching.behavior, new_scope_id))

        return new_results

    def _dfs_expand_primitive(self, last_result: CrowPlanningResult, stmt: Union[CrowBindExpression, CrowRuntimeAssignmentExpression, CrowAssertExpression, CrowControllerApplicationExpression], scope_id: int) -> Sequence[CrowPlanningResult]:
        """Apply a primitive statement on top of a particular planning result of the left branch."""
        # jacinle.log_function.print('Expanding primitive:', stmt, 'with scope_id:', scope_id)
        if isinstance(stmt, CrowControllerApplicationExpression):
            argument_values = [self.executor.execute(x, state=last_result.state, bounded_variables=canonize_bounded_variables(last_result.scopes, scope_id)) for x in stmt.arguments]
            for i, argv in enumerate(argument_values):
                if isinstance(argv, StateObjectReference):
                    argument_values[i] = argv.name
            return [CrowPlanningResult(last_result.state, last_result.csp, last_result.controller_actions + (CrowControllerApplier(stmt.controller, argument_values),), last_result.scopes)]
        elif isinstance(stmt, CrowBindExpression):
            if stmt.is_object_bind:
                new_results = list()
                for new_scope in execute_object_bind(self.executor, stmt, last_result.state, canonize_bounded_variables(last_result.scopes, scope_id)):
                    new_scopes = last_result.scopes.copy()
                    new_scopes[scope_id] = new_scope
                    new_results.append(CrowPlanningResult(last_result.state, last_result.csp, last_result.controller_actions, new_scopes))
                # jacinle.log_function.print(f'Object binding results for {stmt} under scope {canonize_bounded_variables(last_result.scopes, scope_id)}:', len(new_results), 'states.')
                return new_results
            else:
                raise NotImplementedError()
        elif isinstance(stmt, CrowRuntimeAssignmentExpression):
            rv = self.executor.execute(stmt.value, state=last_result.state, bounded_variables=canonize_bounded_variables(last_result.scopes, scope_id))
            new_scopes = last_result.scopes.copy()
            new_scopes[scope_id] = last_result.scopes[scope_id].copy()
            new_scopes[scope_id][stmt.variable.name] = rv.item()
            return [CrowPlanningResult(last_result.state, last_result.csp, last_result.controller_actions, new_scopes)]
        elif isinstance(stmt, CrowAssertExpression):
            # TODO(Jiayuan Mao @ 2024/03/19): implement the CSP tracking.
            rv = self.executor.execute(stmt.bool_expr, state=last_result.state, bounded_variables=canonize_bounded_variables(last_result.scopes, scope_id))
            # jacinle.log_function.print(f'Assert {stmt.bool_expr} under scope {canonize_bounded_variables(last_result.scopes, scope_id)}:', bool(rv.item()))
            if bool(rv.item()):
                return [last_result]
            else:
                return []
        else:
            raise ValueError(f'Unknown statement type: {stmt}')
