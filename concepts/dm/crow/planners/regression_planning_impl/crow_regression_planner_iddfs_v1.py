#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : crow_regression_planner_iddfs_v1.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/30/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import itertools
import warnings
from typing import Any, Optional, Union, Iterator, Tuple, NamedTuple, List, Dict
from types import GeneratorType

import jacinle
from jacinle.utils.meta import UNSET

from concepts.dsl.constraint import ConstraintSatisfactionProblem, OptimisticValue
from concepts.dsl.dsl_types import Variable
from concepts.dsl.expression import VariableExpression, ValueOutputExpression, ObjectConstantExpression, ConstantExpression, NotExpression
from concepts.dsl.value import ListValue
from concepts.dsl.tensor_value import TensorValue
from concepts.dsl.tensor_state import StateObjectReference

from concepts.dm.crow.controller import CrowControllerApplier, CrowControllerApplicationExpression
from concepts.dm.crow.behavior import CrowBehaviorOrderingSuite, CrowBehaviorStatementOrdering, CrowBehaviorCommit
from concepts.dm.crow.behavior import CrowBindExpression, CrowAssertExpression, CrowRuntimeAssignmentExpression, CrowAchieveExpression, CrowUntrackExpression, CrowBehaviorApplicationExpression, CrowBehaviorEffectApplicationExpression
from concepts.dm.crow.behavior import CrowBehaviorForeachLoopSuite, CrowBehaviorWhileLoopSuite, CrowBehaviorConditionSuite, CrowEffectApplier
from concepts.dm.crow.behavior_utils import match_applicable_behaviors, match_policy_applicable_behaviors, ApplicableBehaviorItem
from concepts.dm.crow.behavior_utils import format_behavior_statement, format_behavior_program, execute_behavior_effect_body, execute_object_bind

from concepts.dm.crow.planners.regression_planning import CrowPlanningResult3, CrowRegressionPlanner, ScopedCrowExpression, SupportedCrowExpressionType
from concepts.dm.crow.planners.regression_dependency import RegressionTraceStatement
from concepts.dm.crow.planners.regression_utils import replace_variable_with_value, format_regression_statement
from concepts.dm.crow.csp_solver.dpll_sampling import dpll_solve
from concepts.dm.crow.csp_solver.csp_utils import csp_ground_action_list

__all__ = ['CrowRegressionPlannerIDDFSv1']


class _IDDFSNode(NamedTuple):
    result: CrowPlanningResult3
    """The current planning result."""

    program: Optional[CrowBehaviorOrderingSuite]
    """The current program that is being expanded (middle)."""

    right_statements: Tuple[ScopedCrowExpression, ...]
    """The statements that are waiting to be expanded."""

    commit_execution: bool = False
    """Whether to commit the behavior execution."""

    @classmethod
    def make_empty(cls, state, program, commit_execution=True):
        return cls(
            CrowPlanningResult3.make_empty(state),
            program, tuple(), commit_execution
        )

    def clone(self, result=UNSET, program=UNSET, right_statements=UNSET, commit_execution=UNSET) -> '_IDDFSNode':
        return _IDDFSNode(
            result=result if result is not UNSET else self.result,
            program=program if program is not UNSET else self.program,
            right_statements=right_statements if right_statements is not UNSET else self.right_statements,
            commit_execution=commit_execution if commit_execution is not UNSET else self.commit_execution
        )

    def print(self):
        left_str = ' '.join([str(x) for x in self.result.controller_actions])
        program_str = str(self.program) if self.program is not None else '<empty program>'

        def stringify_stmt(stmt: ScopedCrowExpression) -> str:
            return str(stmt.statement).replace('\n', '') + f'@{stmt.scope_id}'

        right_statements_str = '\n'.join([stringify_stmt(s) for s in self.right_statements]) if len(self.right_statements) > 0 else '<empty right>'
        print('HASH:  ', hash(str(self)), f'!exe={self.commit_execution}')
        print(
            'LEFT:  ' + jacinle.colored(left_str, 'green'),
            'PROG:  ' + jacinle.colored(program_str, 'blue'),
            'RIGHT: \n' + jacinle.indent_text(jacinle.colored(right_statements_str, 'yellow')),
            'SCOPE: ' + str(self.result.latest_scope),
            sep='\n'
        )

    def iter_program_statements(self) -> Iterator[SupportedCrowExpressionType]:
        if self.program is not None:
            yield from self.program.iter_statements()

    def iter_from_right(self) -> Iterator[Tuple['_IDDFSNode', SupportedCrowExpressionType, Optional[CrowBehaviorOrderingSuite], int]]:
        """This is a helper function to iterate over the right-most statement of the program.

        - When `self.right_statements` is empty, this function iterates over the right-most statement of the middle program.
        - When `self.right_statements` is not empty, this function simply pops the right-most statement and returns the new planning state.
        """

        if len(self.right_statements) == 0:
            for middle, stmt, scope_id in self.program.pop_right_statement():
                new_state = self.clone(program=middle)
                yield new_state, stmt, middle, scope_id
        else:
            if isinstance(self.right_statements[-1].statement, CrowBehaviorOrderingSuite):
                for middle, stmt, scope_id in self.right_statements[-1].statement.pop_right_statement():
                    if middle is None:
                        yield self.clone(right_statements=self.right_statements[:-1]), stmt, None, scope_id
                    else:
                        new_state = self.clone(right_statements=self.right_statements[:-1] + (ScopedCrowExpression(middle, scope_id),))
                        yield new_state, stmt , None, scope_id
            else:
                stmt, right_stmt = self.right_statements[-1], self.right_statements[:-1]
                new_state = self.clone(right_statements=right_stmt)
                yield new_state, stmt.statement, None, stmt.scope_id


class CrowRegressionPlannerIDDFSv1(CrowRegressionPlanner):
    def _post_init(self, min_search_depth:int = 5, max_search_depth:int = 20, always_commit_skeleton:bool = True, commit_skeleton_everything: bool = True, enable_state_hash:bool = True):
        """Post initialization of the planner.

        Args:
            min_search_depth (int): The minimum search depth.
            max_search_depth (int): The maximum search depth.
            always_commit_skeleton (bool): Whether to always commit the skeleton of the program.
            commit_skeleton_everything (bool): Whether to commit the skeleton of the program for every type of statements. This is a very aggressive search strategy. For example, it will
                also commit skeleton for `BindExpression` (which will bind the variable to the first value that satisfies the constraint). This is not recommended for most cases.
                However, this flag is set True for backward compatibility reasons. In previous releases, the implementation for `always_commit_skeleton` is actually `commit_skeleton_everything`.
                Setting this flag to False will make the planner only commit the skeleton for `AchieveExpression` and `BehaviorApplicationExpression` when `always_commit_skeleton` is True.
            enable_state_hash (bool): Whether to enable the state hash. When there are not a lot of duplicated states during search, turning this off can improve the performance.
        """
        self.always_commit_skeleton = always_commit_skeleton
        self.commit_skeleton_everything = commit_skeleton_everything
        self.min_search_depth = min_search_depth
        self.max_search_depth = max_search_depth
        self.enable_state_hash = enable_state_hash

        self._enable_human_control_interface = False
        self._enable_visited_states_count = False
        self._visited_states = dict()
        self._visited_states_count = dict()

        if self.always_commit_skeleton is False:
            warnings.warn('`always_commit_skeleton` is set to False. This feature is still experimental and may not work as expected.', RuntimeWarning)

    always_commit_skeleton: bool
    """Whether to always commit the skeleton of the program."""

    min_search_depth: int
    """The minimum search depth."""

    include_effect_appliers: bool
    """Whether to include the effect appliers in the search result. The effect appliers are of type :class:`~concepts.dm.crow.behavior.CrowEffectApplier`."""

    enable_state_hash: bool
    """Whether to enable the state hash. When there are not a lot of duplicated states during search, turning this off can improve the performance."""

    _enable_human_control_interface: bool
    """Whether to enable humans to control the expansion of the search tree."""

    _enable_visited_states_count: bool
    """Whether to enable the visited states count."""

    _visited_states: Dict[Tuple[str, ...], Tuple[Iterator, int]]
    """The visited states. The value is a tuple of (generator, depth)."""

    def set_enable_human_control_interface(self, human_control_interface: bool = True):
        self._enable_human_control_interface = human_control_interface

    def set_enable_visited_states_count(self, enable_visited_states_count: bool = True):
        self._enable_visited_states_count = enable_visited_states_count

    def print_visited_states_count(self):
        if not self._enable_visited_states_count:
            raise RuntimeError('Visited states count is not enabled.')

        table = list()
        for k, count in self._visited_states_count.items():
            table.append((k[0], k[1], k[2], k[3], k[4], count))
        table.sort(key=lambda x: x, reverse=True)
        print(jacinle.tabulate(table, headers=['Program', 'Right Statements', 'Actions', 'Scopes', 'Scope Constraints', 'Count']))

    def main_entry(self, program: CrowBehaviorOrderingSuite, minimize: Optional[ValueOutputExpression] = None) -> List[Tuple[CrowControllerApplier, ...]]:
        state = _IDDFSNode.make_empty(self.state, program, commit_execution=True)
        self._visited_states.clear()
        self._visited_states_count.clear()

        for depth in range(self.min_search_depth, self.max_search_depth + 1):
            results = list(self.dfs(state, depth))

            # NB(Jiayuan Mao @ 2024/08/10): we always apply a CSP solving at the end...
            if len(results) > 0 and self.enable_csp:
                new_results = list()
                for result in results:
                    new_results.extend(self._solve_csp(result))
                results = new_results

            self.set_results(results)
            if len(results) > 0:
                return [x.controller_actions for x in results]

        return []

    def hash_state_human(self, planning_state: _IDDFSNode) -> Tuple[str, ...]:
        """Hash the planning state for human-readable output. Note that this function is not toally consistent with `hash_state`.
        This function is a "stronger" version of `hash_state`. For example, the following two states with different `hash_state` corresponds to the same
        human-readable hash.

        State1::

            unordered{ achieve@4 clear(V::x) }
            Scopes:
                0: {}
                1: {}
                2: {'x': ObjectConstant<A: block>, 'y': ObjectConstant<B: block>}
                3: {'x': ObjectConstant<B: block>, 'y': ObjectConstant<C: block>}
                4: {'x': ObjectConstant<B: block>}

        State2::

            unordered{ achieve@2 clear(V::y) }
            Scopes:
                0: {}
                1: {}
                2: {'x': ObjectConstant<A: block>, 'y': ObjectConstant<B: block>}
                3: {'x': ObjectConstant<B: block>, 'y': ObjectConstant<C: block>}
                4: {'x': ObjectConstant<B: block>}

        Args:
            planning_state: the planning state to hash.

        Returns:
            the hash of the planning state. It is a tuple of strings.
        """
        hash_program = format_behavior_program(planning_state.program, planning_state.result.scopes, flatten=True) if planning_state.program is not None else '<None>'
        hash_statements = tuple(format_regression_statement(x, planning_state.result.scopes) for x in planning_state.right_statements)
        hash_actions = tuple(str(x) for x in planning_state.result.controller_actions)
        hash_scopes = '\n'.join(f'{k}: {v}' for k, v in planning_state.result.scopes.items())
        hash_scope_constraints_parts = list()
        for scope_id, constraints in planning_state.result.scope_constraints.items():
            constraints = '; '.join(format_behavior_statement(x, scopes=planning_state.result.scopes, scope_id=scope_id) for x in constraints)
            hash_scope_constraints_parts.append(f'{scope_id}: {constraints}')
        hash_scope_constraints = '\n'.join(hash_scope_constraints_parts)
        hash_latest_scope_index = planning_state.result.latest_scope

        planning_state_hash = (hash_program, hash_statements, hash_actions, hash_scopes, hash_scope_constraints, hash_latest_scope_index)
        return planning_state_hash

    def hash_state(self, planning_state: _IDDFSNode) -> Tuple[str, ...]:
        hash_program = format_behavior_program(planning_state.program, None, flatten=True) if planning_state.program is not None else '<None>'
        hash_statements = tuple(format_regression_statement(x, None) for x in planning_state.right_statements)
        hash_actions = tuple(str(x) for x in planning_state.result.controller_actions)
        hash_scopes = '\n'.join(f'{k}: {v}' for k, v in planning_state.result.scopes.items())
        hash_scope_constraints = '\n'.join(f'{k}: {v}' for k, v in planning_state.result.scope_constraints.items())
        hash_latest_scope_index = planning_state.result.latest_scope
        planning_state_hash = (hash_program, hash_statements, hash_actions, hash_scopes, hash_scope_constraints, hash_latest_scope_index)
        return planning_state_hash

    def dfs(self, planning_state: _IDDFSNode, depth: int) -> Iterator[CrowPlanningResult3]:
        """The main entry of the depth-first search. This function is a generator of possible planning results.
        Internally, it calls :meth:`dfs_impl` to perform the actual search but wraps around it to handle visited states (so that we can avoid redundant search).

        Args:
            planning_state: the current planning state.
            depth: the current search depth.
        """
        if depth < 0:
            return

        if not self.enable_state_hash:
            yield from self.dfs_impl(planning_state, depth)
            return

        planning_state_hash = self.hash_state(planning_state)

        if planning_state_hash not in self._visited_states or (depth > self._visited_states[planning_state_hash][1]):
            self._visited_states[planning_state_hash] = (self.dfs_impl(planning_state, depth), depth)

        generator, generator_depth = self._visited_states[planning_state_hash]
        if isinstance(generator, GeneratorType) or '_tee' in str(type(generator)):
            g1, g2 = itertools.tee(generator)
            self._visited_states[planning_state_hash] = (g1, generator_depth)
            yield from g2
        else:
            raise RuntimeError('Invalid generator type.')

    def dfs_impl(self, planning_state: _IDDFSNode, depth: int) -> Iterator[CrowPlanningResult3]:
        """The actual implementation of the depth-first search. This function is a generator of possible planning results.

        Args:
            planning_state: the current planning state.
            depth: the current search depth.
        """
        if depth < 0:
            return

        if self._enable_visited_states_count:
            planning_state_hash = self.hash_state_human(planning_state)
            if planning_state_hash in self._visited_states_count:
                self._visited_states_count[planning_state_hash] += 1
            else:
                self._visited_states_count[planning_state_hash] = 1

        if planning_state.program is None and len(planning_state.right_statements) == 0:
            yield planning_state.result
            return

        if self.verbose:
            jacinle.lf_indent_print('Current program:', planning_state.program, 'with', planning_state.right_statements)
        for new_state, stmt, middle_program, scope_id in planning_state.iter_from_right():
            # jacinle.lf_indent_print('-' * 80)
            # jacinle.lf_indent_print('Processing:', stmt, 'constraints', planning_state.result.all_scope_constraints())
            if isinstance(stmt, CrowBehaviorCommit):
                assert middle_program is None, 'Cannot commit a behavior within a promotable program.'
                # if stmt.execution:
                #     raise NotImplementedError('Not implemented yet.')
                if stmt.sketch:
                    # Select only the first solution.
                    results_iterator = itertools.islice(self.dfs(new_state, depth), 1)
                else:
                    results_iterator = self.dfs(new_state, depth)
                if stmt.csp:
                    for result in results_iterator:
                        yield from self._solve_csp(result)
                else:
                    yield from results_iterator
            else:
                nr_yielded = 0
                for x in self.dfs_inner(new_state, stmt, middle_program, scope_id, depth):
                    # jacinle.lf_indent_print('Yielding:', x)
                    yield x
                    nr_yielded += 1
                    if self.always_commit_skeleton:
                        # If this backward-compatible flag is set, we will commit the skeleton for every type of statements.
                        if self.commit_skeleton_everything:
                            break
                        # The "correct" way is to only commit the skeleton for AchieveExpression and BehaviorApplicationExpression.
                        if isinstance(stmt, (CrowAchieveExpression, CrowBehaviorApplicationExpression)):
                            break
                if self.verbose:
                    jacinle.lf_indent_print(f'Yielded {nr_yielded} results for {new_state.program} with {new_state.right_statements}.')

    def dfs_inner(self, planning_state: _IDDFSNode, stmt: SupportedCrowExpressionType, middle_program: Optional[CrowBehaviorOrderingSuite], scope_id: int, depth: int) -> Iterator[CrowPlanningResult3]:
        # Assert stmt is not CrowBehaviorCommit.
        if isinstance(stmt, CrowBehaviorCommit):
            raise RuntimeError('Cannot handle CrowBehaviorCommit in dfs_inner.')
        elif isinstance(stmt, CrowUntrackExpression):
            for result in self.dfs(planning_state, depth):
                yield self._maybe_annotate_dependency(
                    result, result.clone_with_removed_constraint(scope_id, stmt.goal),
                    stmt, scope_id
                )
        elif isinstance(stmt, CrowAchieveExpression):
            if self._enable_human_control_interface:
                jacinle.lf_indent_print('Current stmt:', stmt)
                cmd = input('Skip or continue? [s/C]: ')
                if cmd == 's':
                    return

            cached_results = None
            found_result = False

            enable_direct_achieve = True
            if self._enable_human_control_interface:
                print('Processing achieve stmt:', stmt.goal, 'with', planning_state.result.all_scope_constraints())
                cmd = input('For the direct achieve, skip or continue? [s/C]: ')
                if cmd == 's':
                    enable_direct_achieve = False

            if enable_direct_achieve:
                cached_results = list()
                for result in self.dfs(planning_state, depth):
                    cached_results.append(result)
                    rv, csp = self.evaluate(stmt.goal, result.state, result.csp, bounded_variables=result.scopes[scope_id])
                    if self.verbose:
                        jacinle.lf_indent_print('Achieve stmt direct evaluation::', replace_variable_with_value(stmt.goal, result.scopes[scope_id]), '=>', rv)
                    # jacinle.lf_indent_print('Achieve stmt::', _replace_variable_with_value(stmt.goal, result.scopes[scope_id]), '=>', rv, 'with', result.scopes[scope_id])
                    if isinstance(rv, OptimisticValue):
                        if not stmt.once and self.verbose:
                            jacinle.lf_indent_print(jacinle.colored('  Adding constraint:', 'yellow'), replace_variable_with_value(stmt.goal, result.scopes[scope_id]), 'to', scope_id)
                        yield result.clone(
                            csp=csp.add_equal_constraint(rv, True),
                            dependency_trace=result.dependency_trace + (RegressionTraceStatement(CrowAssertExpression(stmt.goal, once=stmt.once), scope_id, scope=result.scopes[scope_id], additional_info='skip', derived_from=stmt), ) if self.include_dependency_trace else tuple()
                        ).clone_with_new_constraint(scope_id, stmt.goal, True, do=not stmt.once)
                    else:
                        if bool(rv):
                            found_result = True
                            if not stmt.once and self.verbose:
                                jacinle.lf_indent_print(jacinle.colored('  Adding constraint:', 'yellow'), replace_variable_with_value(stmt.goal, result.scopes[scope_id]), 'to', scope_id)
                            yield result.clone_with_new_constraint(scope_id, stmt.goal, True, do=not stmt.once).clone(
                                dependency_trace=result.dependency_trace + (RegressionTraceStatement(CrowAssertExpression(stmt.goal, once=stmt.once), scope_id, scope=result.scopes[scope_id], additional_info='skip', derived_from=stmt), ) if self.include_dependency_trace else tuple()
                            )
            if not found_result or not self.always_commit_skeleton:
                if self.verbose:
                    jacinle.lf_indent_print('=' * 80)
                    jacinle.lf_indent_print('always-commit-skeleton=False or No result found for achieve stmt:', stmt.goal, 'by "direct" achieve after', planning_state.program, planning_state.right_statements, '. Now trying to apply the behavior.')
                    jacinle.lf_indent_print('middle program:', middle_program)
                yield from self._handle_achieve_statement(planning_state, stmt, scope_id, middle_program, depth, cached_results)
        elif isinstance(stmt, CrowBehaviorApplicationExpression):
            yield from self._handle_achieve_statement(planning_state, stmt, scope_id, middle_program, depth)
        elif isinstance(stmt, CrowBehaviorOrderingSuite):
            if stmt.order is CrowBehaviorStatementOrdering.CRITICAL:
                stmt = stmt.unwrap_critical()
                for result in self.dfs(planning_state, depth):
                    yield from self.dfs(_IDDFSNode(result, None, tuple([ScopedCrowExpression(stmt, scope_id)])), depth)
            elif stmt.order is CrowBehaviorStatementOrdering.ALTERNATIVE:
                for alternative in stmt.unwrap_alternative():
                    for result in self.dfs(planning_state, depth):
                        yield from self.dfs(_IDDFSNode(result, None, tuple([ScopedCrowExpression(alternative, scope_id)])), depth)
            else:
                raise RuntimeError('Invalid behavior ordering suite. This branch should not be reached. Report a bug.')
        elif isinstance(stmt, CrowBehaviorForeachLoopSuite):
            if middle_program is not None:
                raise NotImplementedError('Does not support foreach loop in the promotable sections.')

            for result in self.dfs(planning_state, depth):
                yield from self._handle_foreach_loop(result, stmt, scope_id, depth)
        elif isinstance(stmt, CrowBehaviorWhileLoopSuite):
            if middle_program is not None:
                raise NotImplementedError('Does not support while loop in the promotable sections.')

            for result in self.dfs(planning_state, depth):
                yield from self._handle_while_loop(result, stmt, scope_id, depth)
        elif isinstance(stmt, CrowBehaviorConditionSuite):
            if middle_program is not None:
                raise NotImplementedError('Does not support condition in the promotable sections.')
            for result in self.dfs(planning_state, depth):
                yield from self._handle_condition(result, stmt, scope_id, depth)
        elif isinstance(stmt, (CrowBindExpression, CrowAssertExpression, CrowRuntimeAssignmentExpression, CrowControllerApplicationExpression)):
            for result in self.dfs(planning_state, depth):
                for inner_result in self._handle_primitive_statement(result, stmt, scope_id):
                    yield self._maybe_annotate_dependency(result, inner_result, stmt, scope_id)
        elif isinstance(stmt, CrowBehaviorEffectApplicationExpression):
            for result in self.dfs(planning_state, depth):
                for inner_result in self._handle_behavior_effect_application(result, stmt, scope_id):
                    yield self._maybe_annotate_dependency(result, inner_result, stmt, scope_id)
        else:
            raise ValueError(f'Invalid statement type for dfs_inner: {stmt}.')

    def _maybe_annotate_dependency(self, previous_result, result, stmt, scope_id):
        if self.include_dependency_trace:
            if isinstance(stmt, CrowBindExpression):
                if len(stmt.variables) == 1:
                    equiv_stmt = CrowRuntimeAssignmentExpression(stmt.variables[0], VariableExpression(stmt.variables[0]))
                else:
                    equiv_stmt = CrowBehaviorOrderingSuite.make_sequential(*[
                        CrowRuntimeAssignmentExpression(v, VariableExpression(v)) for v in stmt.variables
                    ], variable_scope_identifier=scope_id)
                result = result.clone(dependency_trace=result.dependency_trace + (RegressionTraceStatement(
                    equiv_stmt,
                    scope_id, scope=previous_result.scopes[scope_id], new_scope=result.scopes[scope_id], derived_from=stmt
                ), ))
            else:
                result = result.clone(dependency_trace=result.dependency_trace + (RegressionTraceStatement(stmt, scope_id, scope=previous_result.scopes[scope_id]), ))
        return result

    @jacinle.log_function(verbose=False, is_generator=True)
    def _handle_achieve_statement(
        self, planning_state: _IDDFSNode,
        statement: Union[CrowAchieveExpression, CrowBehaviorApplicationExpression], scope_id: int,
        middle_program: Optional[CrowBehaviorOrderingSuite],
        depth: int, cached_results: Optional[List[CrowPlanningResult3]] = None
    ) -> Iterator[CrowPlanningResult3]:
        # NB(Jiayuan Mao @ 2024/07/13): AchieveExpression statements and BehaviorApplicationExpression statements are handled in a different way.
        # 1. For AchieveExpression statements, the matching is done at a pure "syntax" level. We are trying to exactly match the surface form of the goal expression with other behaviors.
        # For example, if the goal is `pose(x) == pose(y) + [1, 0, 0]`, we are trying to find a behavior that has the same surface form, such as `pose(x) == pose(y) + z`.
        # 2. For BehaviorApplicationExpression statements, the matching is done at a "semantic" level. We are going to evaluate the arguments of the expression.
        # Therefore, this semantic evaluation must be done later, after we have obtained the `result` of the other subgoals.
        if isinstance(statement, CrowAchieveExpression) and not statement.is_policy_achieve:
            behavior_matches = match_applicable_behaviors(self.executor.domain, planning_state.result.state, statement.goal, planning_state.result.scopes[scope_id])
        elif isinstance(statement, CrowAchieveExpression) and statement.is_policy_achieve:
            behavior_matches = match_policy_applicable_behaviors(self.executor.domain, planning_state.result.state, statement.goal, planning_state.result.scopes[scope_id], pachieve_kwargs=statement.flags)
        elif isinstance(statement, CrowBehaviorApplicationExpression):
            behavior_matches = [ApplicableBehaviorItem(statement.behavior, {k.name: v for k, v in zip(statement.behavior.arguments, statement.arguments)}, defered_execution=True)]
        else:
            raise ValueError(f'Invalid statement type for handle_achieve_statement: {statement}.')

        if self.verbose:
            behavior_matches_table = list()
            for behavior_match in behavior_matches:
                behavior_matches_table.append((str(behavior_match.behavior), str(behavior_match.bounded_variables)))
            jacinle.lf_indent_print('Behavior matching for achieve stmt:', statement)
            jacinle.lf_indent_print('Behavior matches:', jacinle.tabulate([(i, b.behavior, b.bounded_variables) for i, b in enumerate(behavior_matches)], headers=['Index', 'Behavior', 'Bounded Variables']), sep='\n')

        if self._enable_human_control_interface:
            cmd = input('Select the behavior to apply: ')
            if len(cmd.strip()) != 0:
                if cmd.isdigit():
                    behavior_matches = [behavior_matches[int(cmd)]]
                else:
                    if cmd == 'ipdb':
                        import ipdb; ipdb.set_trace()

        # Step 1: Try to apply the behavior after other subgoals have been achieved.
        if middle_program is None:
            if cached_results is not None:
                other_subgoal_results = cached_results
            else:
                other_subgoal_results = self.dfs(planning_state, depth)

            for result in other_subgoal_results:
                for behavior_match in behavior_matches:
                    if behavior_match.defered_execution:
                        # TODO(Jiayuan Mao @ 2024/07/18): handle csp cloning correctly.
                        argument_values = {
                            k: self.evaluate(v, state=result.state, csp=result.csp, bounded_variables=result.scopes[scope_id], clone_csp=False, force_tensor_value=True)[0]
                            for k, v in behavior_match.bounded_variables.items()
                        }
                        behavior_match = ApplicableBehaviorItem(behavior_match.behavior, argument_values)
                    if self.verbose:
                        jacinle.lf_indent_print('Trying to apply behavior(1):', behavior_match.behavior, 'with', behavior_match.bounded_variables)
                    self._search_stat['nr_expanded_nodes'] += 1
                    for inner_result in self._handle_achieve_statement_inner1(planning_state, result, behavior_match, statement, scope_id, depth - 1):
                        if isinstance(statement, CrowAchieveExpression):
                            if not statement.once:
                                if self.verbose:
                                    jacinle.lf_indent_print(jacinle.colored('  Adding constraint:', 'yellow'), statement.goal, 'to', scope_id)
                            inner_result = inner_result.clone_with_new_constraint(scope_id, statement.goal, True, do=not statement.once)
                        yield inner_result

        # NB(Jiayuan Mao @ 2024/06/1): If this is turned on, when we can "serialize" the execution of a subgoal, we will not try to apply the behavior together with the "middle program".
        # This will cause the planner to be less expressive, but it can be more efficient.
        # For example, this actually can't solve the Sussman Anomaly problem.
        # if found_result:
        #     return

        # Step 2: Try to apply the behavior together with the "middle program"
        if middle_program is not None:
            for behavior_match in behavior_matches:
                # if not behavior_match.behavior.is_sequential_only():  # If it is sequential only, it should have been already handled in the previous step.
                if True:
                    if behavior_match.defered_execution:
                        # TODO(Jiayuan Mao @ 2024/07/18): handle csp cloning correctly.
                        argument_values = {
                            k: self.evaluate(v, state=planning_state.result.state, csp=planning_state.result.csp, bounded_variables=planning_state.result.scopes[scope_id], clone_csp=False, force_tensor_value=True)[0]
                            for k, v in behavior_match.bounded_variables.items()
                        }
                        behavior_match = ApplicableBehaviorItem(behavior_match.behavior, argument_values)
                    if self.verbose:
                        jacinle.lf_indent_print('Trying to apply behavior(2):', behavior_match.behavior, 'with', behavior_match.bounded_variables)
                    self._search_stat['nr_expanded_nodes'] += 1
                    for inner_result in self._handle_achieve_statement_inner2(planning_state, middle_program, behavior_match, statement, scope_id, depth - 1):
                        if isinstance(statement, CrowAchieveExpression):
                            if not statement.once:
                                if self.verbose:
                                    jacinle.lf_indent_print(jacinle.colored('  Adding constraint:', 'yellow'), replace_variable_with_value(statement.goal, inner_result.scopes[scope_id]), 'to', scope_id)
                            inner_result = inner_result.clone_with_new_constraint(scope_id, statement.goal, True, do=not statement.once)
                            if self.verbose:
                                jacinle.lf_indent_print('Yield result for state', planning_state, 'and statement', statement)
                                jacinle.lf_indent_print('Solution is', inner_result.controller_actions)
                                jacinle.lf_indent_print('Constraints are', inner_result.all_scope_constraints())
                            yield inner_result
                        else:
                            yield inner_result

    def _handle_achieve_statement_inner1(
        self, planning_state: _IDDFSNode, result: CrowPlanningResult3,
        behavior_match: ApplicableBehaviorItem, derived_from_stmt: Union[CrowAchieveExpression, CrowBehaviorApplicationExpression], scope_id: int, depth: int
    ) -> Iterator[CrowPlanningResult3]:
        # NB(Jiayuan Mao @ 2024/05/31): The input `scope_id` is the scope where the statement (the matched behavior) is being applied.
        new_scope_id = result.latest_scope + 1
        new_scopes = result.scopes.copy()
        new_scopes[new_scope_id] = _resolve_bounded_variables(behavior_match.bounded_variables, result.scopes[scope_id])
        program = behavior_match.behavior.assign_body_program_scope(new_scope_id)
        preamble, promotable, rest = program.split_preamble_and_promotable()

        behavior_application_stmt = None
        if self.include_dependency_trace:
            if isinstance(derived_from_stmt, CrowAchieveExpression):
                behavior_application_stmt = CrowBehaviorApplicationExpression(behavior_match.behavior, tuple([VariableExpression(v) for v in behavior_match.behavior.arguments]))
            else:
                behavior_application_stmt, derived_from_stmt = derived_from_stmt, None
        result = result.clone(
            scopes=new_scopes, latest_scope=new_scope_id,
            dependency_trace=result.dependency_trace + (RegressionTraceStatement(behavior_application_stmt, scope_id, new_scope_id=new_scope_id, scope=result.scopes[scope_id], new_scope=new_scopes[new_scope_id], additional_info='sequential achieve', derived_from=derived_from_stmt), ) if self.include_dependency_trace else tuple()
        )
        if preamble is not None:
            preamble = tuple(ScopedCrowExpression(x, new_scope_id) for x in preamble)
            left_results = self.dfs(_IDDFSNode(result, None, preamble), depth)
        else:
            left_results = [result]

        for left_result in left_results:
            new_promotable = CrowBehaviorOrderingSuite.make_sequential(promotable, variable_scope_identifier=new_scope_id) if promotable is not None else None
            new_rest = tuple(ScopedCrowExpression(x, new_scope_id) for x in rest) + (ScopedCrowExpression(CrowBehaviorEffectApplicationExpression(behavior_match.behavior), new_scope_id),)
            yield from self.dfs(_IDDFSNode(left_result, new_promotable, new_rest), depth)

    def _handle_achieve_statement_inner2(
        self, planning_state: _IDDFSNode, middle_program: CrowBehaviorOrderingSuite,
        behavior_match: ApplicableBehaviorItem, derived_from_stmt: Union[CrowAchieveExpression, CrowBehaviorApplicationExpression], scope_id: int, depth: int
    ) -> Iterator[CrowPlanningResult3]:
        for value in behavior_match.bounded_variables.values():
            assert not isinstance(value, Variable)

        new_scope_id = planning_state.result.latest_scope + 1
        new_scopes = planning_state.result.scopes.copy()
        new_scopes[new_scope_id] = _resolve_bounded_variables(behavior_match.bounded_variables, planning_state.result.scopes[scope_id])

        program = behavior_match.behavior.assign_body_program_scope(new_scope_id)
        preamble, promotable, rest = program.split_preamble_and_promotable()

        behavior_application_stmt = None
        if self.include_dependency_trace:
            if isinstance(derived_from_stmt, CrowAchieveExpression):
                behavior_application_stmt = CrowBehaviorApplicationExpression(behavior_match.behavior, tuple([VariableExpression(v) for v in behavior_match.behavior.arguments]))
            else:
                behavior_application_stmt, derived_from_stmt = derived_from_stmt, None
        result = planning_state.result.clone(
            scopes=new_scopes, latest_scope=new_scope_id,
            dependency_trace=planning_state.result.dependency_trace + (RegressionTraceStatement(behavior_application_stmt, scope_id, new_scope_id=new_scope_id, scope=new_scopes[scope_id], new_scope=new_scopes[new_scope_id], additional_info='promotable achieve', derived_from=derived_from_stmt), ) if self.include_dependency_trace else tuple()
        )
        if preamble is not None:
            preamble = tuple(ScopedCrowExpression(x, new_scope_id) for x in preamble)
            left_results = self.dfs(_IDDFSNode(result, None, preamble), depth)
        else:
            left_results = [result]

        for left_result in left_results:
            if promotable is not None:
                if middle_program is not None:
                    new_promotable = CrowBehaviorOrderingSuite.make_unordered(middle_program, CrowBehaviorOrderingSuite.make_sequential(promotable, variable_scope_identifier=new_scope_id))
                else:
                    new_promotable = CrowBehaviorOrderingSuite.make_sequential(promotable, variable_scope_identifier=new_scope_id)
            else:
                new_promotable = middle_program
            new_rest = tuple(ScopedCrowExpression(x, new_scope_id) for x in rest) + (ScopedCrowExpression(CrowBehaviorEffectApplicationExpression(behavior_match.behavior), new_scope_id),)
            yield from self.dfs(_IDDFSNode(left_result, new_promotable, new_rest), depth)

    def _handle_foreach_loop(
        self, result: CrowPlanningResult3, stmt: CrowBehaviorForeachLoopSuite, scope_id: int,
        depth: int,
    ) -> Iterator[CrowPlanningResult3]:
        if self.verbose:
            jacinle.lf_indent_print('Processing foreach loop:', stmt)

        variable = stmt.variable
        if stmt.is_foreach_in_expression:
            objects, new_csp = self.evaluate(stmt.loop_in_expression, state=result.state, csp=result.csp, bounded_variables=result.scopes[scope_id], clone_csp=True)
            result = result.clone(csp=new_csp)
            if isinstance(objects, ListValue):
                objects = objects.values
            elif isinstance(objects, TensorValue) and objects.dtype.is_batched_list_type:
                objects = tuple(objects.iter_batched_indexing())
        else:
            objects = result.state.object_type2name[variable.dtype.typename]

        if self.verbose:
            jacinle.lf_indent_print('Foreach loop objects:', objects)

        dependency_traces = list()
        statements = list()
        new_scopes = result.scopes.copy()
        new_scope_id = result.latest_scope
        for index in range(len(objects)):
            new_scope_id = new_scope_id + 1
            new_scopes[new_scope_id] = new_scopes[scope_id].copy()

            is_object = False
            if isinstance(objects[index], StateObjectReference):
                is_object = True
                new_scopes[new_scope_id][variable.name] = objects[index]
            elif isinstance(objects[index], str):
                is_object = True
                new_scopes[new_scope_id][variable.name] = StateObjectReference(objects[index], index, variable.dtype)
            elif isinstance(objects[index], TensorValue):
                new_scopes[new_scope_id][variable.name] = objects[index]
            else:
                raise ValueError(f'Invalid object type: {objects[index]}.')

            if self.include_dependency_trace:
                constant_expression = ObjectConstantExpression(new_scopes[new_scope_id][variable.name]) if is_object else ConstantExpression(objects[index])
                dependency_traces.append(RegressionTraceStatement(
                    CrowRuntimeAssignmentExpression(variable, constant_expression),
                    scope_id, new_scope_id=new_scope_id,
                    scope=new_scopes[scope_id], new_scope=new_scopes[new_scope_id],
                    additional_info=f'loop variable {variable.name}={new_scopes[new_scope_id][variable.name].name}' if is_object else f'loop variable {variable.name}={objects[index]}',
                    derived_from=stmt
                ))

            statements.extend([ScopedCrowExpression(x, new_scope_id) for x in stmt.statements])

        new_r = result.clone(
            scopes=new_scopes, latest_scope=new_scope_id,
            dependency_trace=result.dependency_trace + tuple(dependency_traces) if self.include_dependency_trace else tuple()
        )
        state = _IDDFSNode(new_r, None, tuple(statements))
        yield from self.dfs(state, depth)

    def _handle_while_loop(
        self, result: CrowPlanningResult3, stmt: CrowBehaviorWhileLoopSuite, scope_id: int,
        depth: int
    ) -> Iterator[CrowPlanningResult3]:
        if self.verbose:
            jacinle.lf_indent_print('Processing while loop:', stmt)

        def while_dfs(r: CrowPlanningResult3, while_depth: int):
            if while_depth == 0:
                return

            rv, new_csp = self.evaluate(stmt.condition, state=r.state, csp=r.csp, bounded_variables=r.scopes[scope_id])

            if self.verbose:
                jacinle.lf_indent_print('While loop condition:', stmt.condition, '=>', rv, '@', while_depth)

            if isinstance(rv, OptimisticValue):
                yield r.clone(
                    csp=new_csp.add_equal_constraint(rv, False),
                    dependency_trace=r.dependency_trace + (RegressionTraceStatement(CrowAssertExpression(NotExpression(stmt.condition)), scope_id, scope=r.scopes[scope_id], additional_info='while end', derived_from=stmt), ) if self.include_dependency_trace else tuple()
                )
                state = _IDDFSNode(r.clone(
                    csp=new_csp.add_equal_constraint(rv, True),
                    dependency_trace=r.dependency_trace + (RegressionTraceStatement(CrowAssertExpression(stmt.condition), scope_id, scope=r.scopes[scope_id], additional_info=' while continue', derived_from=stmt), ) if self.include_dependency_trace else tuple()
                ), None, tuple(ScopedCrowExpression(x, scope_id) for x in stmt.statements))
                for inner_result in self.dfs(state, depth - 1):
                    yield from while_dfs(inner_result, while_depth - 1)
            else:
                if bool(rv):
                    if self.include_dependency_trace:
                        r = r.clone(dependency_trace=r.dependency_trace + (RegressionTraceStatement(CrowAssertExpression(stmt.condition), scope_id, scope=r.scopes[scope_id], additional_info='while continue', derived_from=stmt), ))
                    state = _IDDFSNode(r, None, tuple(ScopedCrowExpression(x, scope_id) for x in stmt.statements))
                    for inner_result in self.dfs(state, depth - 1):
                        yield from while_dfs(inner_result, while_depth - 1)
                else:
                    if self.include_dependency_trace:
                        r = r.clone(dependency_trace=r.dependency_trace + (RegressionTraceStatement(CrowAssertExpression(NotExpression(stmt.condition)), scope_id, scope=r.scopes[scope_id], additional_info='while end', derived_from=stmt), ))
                    yield r

        yield from while_dfs(result, stmt.max_depth)

    def _handle_condition(
        self, result: CrowPlanningResult3, stmt: CrowBehaviorConditionSuite, scope_id: int,
        depth: int
    ) -> Iterator[CrowPlanningResult3]:
        if self.verbose:
            jacinle.lf_indent_print('Processing condition suite:', stmt)

        rv, new_csp = self.evaluate(stmt.condition, state=result.state, csp=result.csp, bounded_variables=result.scopes[scope_id])
        if self.verbose:
            jacinle.lf_indent_print('Condition suite:', stmt.condition, '=>', rv)

        if isinstance(rv, OptimisticValue):
            yield from self.dfs(_IDDFSNode(result.clone(
                csp=new_csp.add_equal_constraint(rv, True),
                dependency_trace=result.dependency_trace + (RegressionTraceStatement(CrowAssertExpression(stmt.condition), scope_id, scope=result.scopes[scope_id], additional_info='condition=True branch', derived_from=stmt), ) if self.include_dependency_trace else tuple()
            ), None, tuple(ScopedCrowExpression(x, scope_id) for x in stmt.statements)), depth)
            if stmt.else_statements is not None:
                yield from self.dfs(_IDDFSNode(result.clone(
                    csp=new_csp.add_equal_constraint(rv, False),
                    dependency_trace=result.dependency_trace + (RegressionTraceStatement(CrowAssertExpression(NotExpression(stmt.condition)), scope_id, scope=result.scopes[scope_id], additional_info='condition=False branch', derived_from=stmt), ) if self.include_dependency_trace else tuple()
                ), None, tuple(ScopedCrowExpression(x, scope_id) for x in stmt.else_statements)), depth)
            else:
                yield result.clone(csp=new_csp.add_equal_constraint(rv, False))
        else:
            if bool(rv):
                if self.include_dependency_trace:
                    result = result.clone(dependency_trace=result.dependency_trace + (RegressionTraceStatement(CrowAssertExpression(stmt.condition), scope_id, scope=result.scopes[scope_id], additional_info='condition=True branch', derived_from=stmt), ))
                yield from self.dfs(_IDDFSNode(result, None, tuple(ScopedCrowExpression(x, scope_id) for x in stmt.statements)), depth)
            else:
                if stmt.else_statements is not None:
                    if self.include_dependency_trace:
                        result = result.clone(dependency_trace=result.dependency_trace + (RegressionTraceStatement(CrowAssertExpression(NotExpression(stmt.condition)), scope_id, scope=result.scopes[scope_id], additional_info='condition=False branch', derived_from=stmt), ))
                    yield from self.dfs(_IDDFSNode(result, None, tuple(ScopedCrowExpression(x, scope_id) for x in stmt.else_statements)), depth)
                else:
                    if self.include_dependency_trace:
                        result = result.clone(dependency_trace=result.dependency_trace + (RegressionTraceStatement(CrowAssertExpression(NotExpression(stmt.condition)), scope_id, scope=result.scopes[scope_id], additional_info='condition=False branch', derived_from=stmt), ))
                    yield result

    def _handle_primitive_statement(
        self, result: CrowPlanningResult3,
        stmt: Union[CrowBindExpression, CrowAssertExpression, CrowRuntimeAssignmentExpression, CrowControllerApplicationExpression], scope_id: int
    ) -> Iterator[CrowPlanningResult3]:
        if isinstance(stmt, CrowBindExpression):
            if self.verbose:
                jacinle.lf_indent_print('Processing bind stmt:', stmt)
            if stmt.is_object_bind:
                for i, new_scope in enumerate(execute_object_bind(self.executor, stmt, result.state, result.scopes[scope_id])):
                    new_scopes = result.scopes.copy()
                    new_scopes[scope_id] = new_scope
                    if self.verbose:
                        jacinle.lf_indent_print('New scope:', new_scope)
                    yield result.clone(scopes=new_scopes)
            else:
                new_csp = result.csp.clone()
                new_scopes = result.scopes.copy()
                new_scopes[scope_id] = result.scopes[scope_id].copy()
                for var in stmt.variables:
                    new_scopes[scope_id][var.name] = TensorValue.from_optimistic_value(new_csp.new_var(var.dtype, wrap=True))
                if not stmt.goal.is_null_expression:
                    rv, new_csp = self.evaluate(stmt.goal, state=result.state, csp=new_csp, bounded_variables=new_scopes[scope_id], force_tensor_value=True, clone_csp=False)
                    yield result.clone(csp=new_csp.add_equal_constraint(rv, True), scopes=new_scopes)
                else:
                    yield result.clone(csp=new_csp, scopes=new_scopes)
        elif isinstance(stmt, CrowAssertExpression):
            rv, new_csp = self.evaluate(stmt.bool_expr, state=result.state, csp=result.csp, bounded_variables=result.scopes[scope_id])
            if self.verbose:
                jacinle.lf_indent_print('Processing assert stmt:', stmt.bool_expr, '=>', rv)
            if isinstance(rv, OptimisticValue):
                yield result.clone(csp=new_csp.add_equal_constraint(rv, True)).clone_with_new_constraint(scope_id, stmt.bool_expr, True, do=not stmt.once)
            elif bool(rv):
                yield result.clone_with_new_constraint(scope_id, stmt.bool_expr, True, do=not stmt.once)
            else:
                pass  # Return nothing.
        elif isinstance(stmt, CrowRuntimeAssignmentExpression):
            rv, new_csp = self.evaluate(stmt.value, state=result.state, csp=result.csp, bounded_variables=result.scopes[scope_id], force_tensor_value=True)
            if self.verbose:
                jacinle.lf_indent_print('Processing runtime assignment stmt:', stmt.variable, '<-', stmt.value, '. Value:', rv)
            new_scopes = result.scopes.copy()
            new_scopes[scope_id] = result.scopes[scope_id].copy()
            new_scopes[scope_id][stmt.variable.name] = rv
            yield result.clone(csp=new_csp, scopes=new_scopes)
        elif isinstance(stmt, CrowControllerApplicationExpression):
            new_csp = result.csp.clone()
            argument_values = [self.evaluate(x, state=result.state, csp=new_csp, bounded_variables=result.scopes[scope_id], clone_csp=False, force_tensor_value=True)[0] for x in stmt.arguments]
            if self.verbose:
                jacinle.lf_indent_print('Processing controller application stmt:', CrowControllerApplier(stmt.controller, argument_values))
            result = result.clone(controller_actions=result.controller_actions + (CrowControllerApplier(stmt.controller, argument_values),))
            if stmt.controller.effect_body is not None:
                yield from self._handle_behavior_effect_application(result, stmt, None, argument_values)
            else:
                yield result
        else:
            raise ValueError(f'Invalid statement type for handle_primitive_statement: {stmt}.')

    def _handle_behavior_effect_application(
        self, result: CrowPlanningResult3, effect: Union[CrowBehaviorEffectApplicationExpression, CrowControllerApplicationExpression], scope_id: Optional[int],
        argument_values: Optional[List[Any]] = None
    ) -> Iterator[CrowPlanningResult3]:
        if self.verbose:
            if isinstance(effect, CrowBehaviorEffectApplicationExpression):
                jacinle.lf_indent_print('Processing behavior effect:', effect.behavior)
            elif isinstance(effect, CrowControllerApplicationExpression):
                jacinle.lf_indent_print('Processing controller effect:', effect.controller)
            else:
                raise ValueError(f'Invalid effect type: {effect}.')
            jacinle.lf_indent_print('Constraints:', result.all_scope_constraints())

        new_csp = result.csp.clone()

        if isinstance(effect, CrowBehaviorEffectApplicationExpression):
            new_state = execute_behavior_effect_body(self.executor, effect.behavior, state=result.state, csp=new_csp, scope=result.scopes[scope_id], action_index=len(result.controller_actions) - 1)
            effect_applier = CrowEffectApplier(effect.behavior.effect_body.statements, result.scopes[scope_id])
        elif isinstance(effect, CrowControllerApplicationExpression):
            scope = {x.name: y for x, y in zip(effect.controller.arguments, argument_values)}
            new_state = execute_behavior_effect_body(self.executor, effect.controller, state=result.state, csp=new_csp, scope=scope, action_index=len(result.controller_actions) - 1)
            effect_applier = CrowEffectApplier(effect.controller.effect_body.statements, scope)
        else:
            raise ValueError(f'Invalid effect type: {effect}.')

        if scope_id is not None:
            new_scopes = result.scopes.copy()
            # del new_scopes[scope_id]
            new_scope_constraints = result.scope_constraints.copy()
            if scope_id in new_scope_constraints:
                del new_scope_constraints[scope_id]
                if self.verbose:
                    jacinle.lf_indent_print('Removing constraints for scope', scope_id)
        else:
            new_scopes = result.scopes
            new_scope_constraints = result.scope_constraints
        new_scope_constraint_evaluations = dict()

        if self.verbose:
            jacinle.lf_indent_print('Previous Actions', result.controller_actions)
        for c_scope_id, constraints in new_scope_constraints.items():
            if self.verbose:
                jacinle.lf_indent_print('Constraints for scope', c_scope_id, ':')
            new_scope_constraint_evaluations[c_scope_id] = list()
            for constraint in constraints:
                c_rv, _ = self.evaluate(constraint, state=new_state, csp=new_csp, bounded_variables=new_scopes[c_scope_id], clone_csp=False)
                if self.verbose:
                    if bool(c_rv) is False:
                        jacinle.lf_indent_print(jacinle.colored('  Constraint:', 'red'), replace_variable_with_value(constraint, new_scopes[c_scope_id]), '=>', c_rv)
                    else:
                        jacinle.lf_indent_print(jacinle.colored('  Constraint:', 'green'), replace_variable_with_value(constraint, new_scopes[c_scope_id]), '=>', c_rv)

                if isinstance(c_rv, OptimisticValue):
                    new_csp.add_equal_constraint(c_rv, True)
                    new_scope_constraint_evaluations[c_scope_id].append(True)
                else:
                    if bool(c_rv):
                        new_scope_constraint_evaluations[c_scope_id].append(True)
                    else:
                        if self.verbose:
                            jacinle.lf_indent_print(jacinle.colored('  Constraint violated:', 'red'), replace_variable_with_value(constraint, new_scopes[c_scope_id]), '=>', c_rv)
                        # This constraint has been violated.
                        return

        new_controller_actions = result.controller_actions
        if self.include_effect_appliers:
            new_controller_actions += (effect_applier,)
        yield result.clone(
            state=new_state, csp=new_csp, controller_actions=new_controller_actions,
            scopes=new_scopes, scope_constraints=new_scope_constraints, scope_constraint_evaluations=new_scope_constraint_evaluations
        )

    def _solve_csp(self, result: CrowPlanningResult3) -> Iterator[CrowPlanningResult3]:
        if not self.enable_csp:
            yield result
            return

        if result.csp is None or result.csp.empty():
            yield result
            return

        solution = dpll_solve(self.executor, result.csp, simulation_interface=self.simulation_interface, generator_manager=self.generator_manager, actions=result.controller_actions, verbose=self.verbose)

        if solution is not None:
            actions = csp_ground_action_list(self.executor, result.controller_actions, solution)
            # We have cleared the CSP.
            # TODO(Jiayuan Mao @ 2024/06/8): We also need to update the scope variables --- they can now be directly assigned. The same applies to the state.
            # For now this is okay, because we are only solving the CSP problem at the very end.
            yield result.clone(csp=ConstraintSatisfactionProblem(), controller_actions=actions)


def _resolve_bounded_variables(bounded_variables: Dict[str, Any], scope: Dict[str, Any]) -> Dict[str, Any]:
    bounded_variables = bounded_variables.copy()
    for name, value in bounded_variables.items():
        if isinstance(value, Variable):
            if value.name not in scope:
                raise ValueError(f'Variable {value.name} not found in the scope.')
            bounded_variables[name] = scope[value.name]
    return bounded_variables

