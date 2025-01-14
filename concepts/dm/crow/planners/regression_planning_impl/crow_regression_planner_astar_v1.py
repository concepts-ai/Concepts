#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : crow_regression_planner_astar_v1.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/21/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import queue
from typing import Optional, Union, Iterator, Sequence, Tuple, List, Dict, NamedTuple

import jacinle
from jacinle.utils.meta import UNSET

from concepts.dsl.constraint import ConstraintSatisfactionProblem, OptimisticValue
from concepts.dsl.dsl_types import Variable
from concepts.dsl.expression import ValueOutputExpression
from concepts.dsl.tensor_value import TensorValue
from concepts.dsl.tensor_state import StateObjectReference

from concepts.dm.crow.crow_domain import CrowState
from concepts.dm.crow.controller import CrowControllerApplier, CrowControllerApplicationExpression
from concepts.dm.crow.behavior import CrowBehavior, CrowAchieveExpression, CrowBindExpression, CrowRuntimeAssignmentExpression, CrowAssertExpression, CrowBehaviorApplicationExpression
from concepts.dm.crow.behavior import CrowBehaviorOrderingSuite
from concepts.dm.crow.behavior_utils import match_applicable_behaviors, ApplicableBehaviorItem, execute_object_bind

from concepts.dm.crow.planners.regression_planning import CrowPlanningResult, CrowRegressionPlanner
from concepts.dm.crow.planners.regression_utils import canonicalize_bounded_variables
from concepts.dm.crow.planners.regression_planning_impl.crow_regression_planner_dfs_v1_utils import execute_behavior_effect
from concepts.dm.crow.csp_solver.csp_utils import csp_ground_action_list

__all__ = ['CrowRegressionPlannerAStarv1']


class _CrowBehaviorEffectApplication(NamedTuple):
    """Apply the effect of an action."""
    behavior: CrowBehavior


class _ScopedStatement2(NamedTuple):
    """A statement in the right stack of the planning state."""
    statement: Union[CrowBehaviorOrderingSuite, CrowBindExpression, CrowAssertExpression, CrowControllerApplicationExpression, _CrowBehaviorEffectApplication, CrowAchieveExpression, CrowBehaviorApplicationExpression]
    """The statement."""

    scope_id: int
    """The scope id of the statement."""

    expanded_state: Optional['_AStarNode'] = None
    """The state where this statement was serialized to the right stack."""


class _AStarNode(NamedTuple):
    """The planning state for the A* search algorithm."""

    program: Optional[CrowBehaviorOrderingSuite]
    """The current program that is being expanded (middle)."""

    state: CrowState
    """The current state of the planning."""

    csp: ConstraintSatisfactionProblem
    """The current constraint satisfaction problem."""

    scopes: Dict[int, dict]
    """The current scopes."""

    latest_scope: int
    """The latest scope id."""

    left_statements: Tuple[CrowControllerApplier, ...]
    """The (left) statements that have been executed."""

    right_statements: Tuple[_ScopedStatement2, ...]
    """The statements that are waiting to be expanded."""

    statements_evaluations: Dict[int, bool]
    """Whether a statement has been evaluated."""

    commit_sketch: bool = False
    """Whether to commit the sketch."""

    commit_csp: bool = False
    """Whether to commit the CSP."""

    commit_execution: bool = False
    """Whether to commit the behavior execution."""

    def clone(
        self, program=UNSET, state=UNSET, csp=UNSET, scopes=UNSET, latest_scope=UNSET, left_statements=UNSET, right_statements=UNSET, statements_evaluations=UNSET,
        commit_sketch=UNSET, commit_csp=UNSET, commit_execution=UNSET
    ):
        return _AStarNode(
            program if program is not UNSET else self.program,
            state if state is not UNSET else self.state,
            csp if csp is not UNSET else self.csp,
            scopes if scopes is not UNSET else self.scopes,
            latest_scope if latest_scope is not UNSET else self.latest_scope,
            left_statements if left_statements is not UNSET else self.left_statements,
            right_statements if right_statements is not UNSET else self.right_statements,
            statements_evaluations if statements_evaluations is not UNSET else (dict() if program is not UNSET else self.statements_evaluations),
            commit_sketch if commit_sketch is not UNSET else self.commit_sketch,
            commit_csp if commit_csp is not UNSET else self.commit_csp,
            commit_execution if commit_execution is not UNSET else self.commit_execution
        )

    @classmethod
    def make_empty(cls, program, state, csp, commit_execution=False, commit_sketch=False, commit_csp=False):
        return cls(
            program, state, csp, scopes={0: {}}, latest_scope=0, left_statements=tuple(), right_statements=tuple(), statements_evaluations=dict(),
            commit_execution=commit_execution, commit_sketch=commit_sketch, commit_csp=commit_csp
        )

    def print(self):
        left_str = ' '.join([str(x) for x in self.left_statements]) if len(self.left_statements) > 0 else '<empty left>'
        program_str = str(self.program) if self.program is not None else '<empty program>'

        def stringify_stmt(stmt: _ScopedStatement2) -> str:
            return str(stmt.statement).replace('\n', '') + f'@{stmt.scope_id}'

        right_statements_str = '\n'.join([stringify_stmt(s) for s in self.right_statements]) if len(self.right_statements) > 0 else '<empty right>'
        print('HASH:  ', hash(str(self)), f'!sketch={self.commit_sketch}', f'!csp={self.commit_csp}', f'!exe={self.commit_execution}')
        print(
            'LEFT:  ' + jacinle.colored(left_str, 'green'),
            'PROG:  ' + jacinle.colored(program_str, 'blue'),
            'RIGHT: \n' + jacinle.indent_text(jacinle.colored(right_statements_str, 'yellow')),
            'SCOPE: ' + str(self.scopes),
            sep='\n'
        )

    def iter_program_statements(self) -> Iterator[Union[CrowAchieveExpression, CrowBindExpression, CrowRuntimeAssignmentExpression, CrowAssertExpression, CrowControllerApplicationExpression, CrowBehaviorApplicationExpression]]:
        if self.program is not None:
            yield from self.program.iter_statements()


class _AStarNodeWithHeuristic(NamedTuple):
    state: _AStarNode

    g: float
    h: float

    previous_state: Optional['_AStarNodeWithHeuristic'] = None

    def __lt__(self, other):
        return self.g + self.h < other.g + other.h

    def print_history(self):
        self.state.print()
        queue_state = self
        while queue_state.previous_state is not None:
            print('<-' * 30)
            queue_state = queue_state.previous_state
            queue_state.state.print()


class SolutionFound(Exception):
    def __init__(self, results: Sequence[CrowPlanningResult]):
        self.results = results


LOG_GRAPH_STATES = False
LOG_ENQUEUE_DEQUEUE = False


class CrowRegressionPlannerAStarv1(CrowRegressionPlanner):
    def main_entry(self, program: CrowBehaviorOrderingSuite, minimize: Optional[ValueOutputExpression] = None) -> List[Tuple[CrowControllerApplier, ...]]:
        state = _AStarNode.make_empty(program, self.state, ConstraintSatisfactionProblem())
        results = self.bfs(state)
        return [x.controller_actions for x in results]

    def _compute_heuristic(self, state: _AStarNode) -> float:
        """Compute the heuristic value for a planning state.

        The current implementation uses a simple (non-admissible) heuristic that counts the number of achieve statements in the program and the right statements.
        Roughly speaking, it is the number of "subgoals" that haven't been achieved yet.
        """
        h = 0
        if state.program is not None:
            for expression in state.program.iter_statements():
                if isinstance(expression, CrowAchieveExpression) or isinstance(expression, CrowControllerApplicationExpression):
                    h += 1
        for item in state.right_statements:
            if isinstance(item.statement, (CrowAchieveExpression, CrowBehaviorApplicationExpression)):
                h += 1
            elif isinstance(item.statement, CrowBehaviorOrderingSuite):
                for stmt in item.statement.iter_statements():
                    if isinstance(stmt, (CrowAchieveExpression, CrowBehaviorApplicationExpression)):
                        h += 1
        return h

    queue: queue.PriorityQueue

    _graph: dict
    _current_queue_state: Optional[_AStarNodeWithHeuristic] = None
    _expanded_right_first: Dict[int, _AStarNodeWithHeuristic]
    _expanded_queue_nodes: Dict[str, _AStarNodeWithHeuristic]
    _expanded_queue_node_to_children: Dict[int, List[_AStarNodeWithHeuristic]]

    def bfs(self, planning_state: _AStarNode) -> Sequence[CrowPlanningResult]:
        self.queue = queue.PriorityQueue()

        self._graph = {'nodes': {}, 'edges': list()}
        self._expanded_right_first = dict()
        self._expanded_queue_nodes = dict()
        self._expanded_queue_node_to_children = dict()

        self.bfs_add_queue(planning_state)

        try:
            while not self.queue.empty():
                self._search_stat['nr_expanded_nodes'] += 1
                if self._search_stat['nr_expanded_nodes'] % 1000 == 0:
                    print('Expanded nodes:', self._search_stat['nr_expanded_nodes'])
                if self._search_stat['nr_expanded_nodes'] > 1000000:
                    print('Too many expanded nodes.')
                    import ipdb; ipdb.set_trace()
                    break

                queue_state = self.queue.get()
                self._current_queue_state = queue_state
                self._expanded_queue_node_to_children[id(queue_state)] = list()
                self.bfs_expand(queue_state.state)
        except SolutionFound as e:
            return e.results

        return []

    def bfs_add_queue(self, planning_state: _AStarNode):
        if LOG_ENQUEUE_DEQUEUE:
            print(jacinle.colored('Enqueue' + '-' * 60, 'blue'))
            planning_state.print()
            # input('Press Enter to continue...')

        g = len(planning_state.left_statements)
        h = self._compute_heuristic(planning_state)

        if planning_state.program is None and len(planning_state.right_statements) == 0:
            if not planning_state.csp.empty():
                from concepts.dm.crow.csp_solver.dpll_sampling import dpll_solve
                solution = dpll_solve(self.executor, planning_state.csp, simulation_interface=self.simulation_interface, actions=planning_state.left_statements)

                if solution is None:
                    # If the CSP is unsatisfiable, we will prune this branch.
                    return
                else:
                    actions = csp_ground_action_list(self.executor, planning_state.left_statements, solution)
                    raise SolutionFound([CrowPlanningResult(planning_state.state, planning_state.csp, actions, planning_state.scopes)])

            raise SolutionFound([CrowPlanningResult(planning_state.state, planning_state.csp, planning_state.left_statements, planning_state.scopes)])

        queue_state = _AStarNodeWithHeuristic(planning_state, g, h, self._current_queue_state)

        if LOG_GRAPH_STATES:
            self._graph['nodes'][id(queue_state)] = queue_state
            if self._current_queue_state is not None:
                self._graph['edges'].append((id(self._current_queue_state), id(queue_state)))

        state_hash = str(queue_state.state)
        if state_hash in self._expanded_queue_nodes:
            if id(self._current_queue_state.state.state) == id(self._expanded_queue_nodes[state_hash].state.state):
                print('Already expanded queue node.', hash(state_hash))
                return
        self._expanded_queue_nodes[state_hash] = queue_state

        if self._current_queue_state is not None:
            self._expanded_queue_node_to_children[id(self._current_queue_state)].append(queue_state)
        self.queue.put(queue_state)

    def bfs_expand(self, planning_state: _AStarNode):
        """The BFS algorithm is simulating a hierarchical planning algorithm. The current state can be encoded as:

        .. code-block:: python

            left_actions = (a1, a2, a3, ...)
            middle_program = CrowActionOrderingSuite(...)
            right_statements = [RegressionStatement2(...), RegressionStatement2(...), ...]

        It corresponds to this figure:

        .. code-block:: python

            a1 -> a2 -> a3 -> ... -> {middle_program} -> [right_statements]

        Therefore,

        - state.left_actions: the actions that have been executed (a1, a2, a3, ...).
        - state.program: the current program that is being expanded ({middle_program}).
        - state.right_statements: the statements that are waiting to be expanded ([right_statements]).

        At each step,

        - If the program is empty, we will pop up the first right statement and expand it.
        - If the program is not empty, we will randomly pop a statement from the middle program, and prepend it to the right statements.
        """

        if LOG_ENQUEUE_DEQUEUE:
            print(jacinle.colored('Dequeue ' + '-' * 60, 'red'))
            planning_state.print()
            input('Press Enter to continue...')

        if planning_state.program is None:
            # The current main program body is empty. We will pop up the first right statement and expand it.
            stmt = planning_state.right_statements[0]
            right_stmts = planning_state.right_statements[1:]
            planning_state = planning_state.clone(program=None, right_statements=right_stmts)
            self._bfs_expand_inner(planning_state, None, stmt.statement, stmt.expanded_state, stmt.scope_id, stmt_id=id(stmt))
        else:
            all_satisfied = self._bfs_is_all_satisfied(planning_state)
            if all_satisfied:
                self.bfs_add_queue(planning_state.clone(program=None))
            else:
                # The current main program body is not empty. We will randomly pop a statement from the middle program, and prepend it to the right statements.
                # A special case is that after popping the statement, the middle program becomes empty. In this case, we will directly expand the right statements.
                for middle, stmt, scope_id in planning_state.program.pop_right_statement():
                    new_state = planning_state.clone(program=middle)
                    self._bfs_expand_inner(planning_state, middle, stmt, new_state, scope_id, stmt_id=None)

    def _bfs_is_all_satisfied(self, planning_state: _AStarNode) -> bool:
        for stmt, scope_id in planning_state.program.iter_statements_with_scope():
            if id(stmt) in planning_state.statements_evaluations:
                if planning_state.statements_evaluations[id(stmt)]:
                    continue
                else:
                    return False

            if isinstance(stmt, CrowAchieveExpression) or isinstance(stmt, CrowAssertExpression):
                expr = stmt.goal if isinstance(stmt, CrowAchieveExpression) else stmt.bool_expr
                rv = self.executor.execute(expr, state=planning_state.state, bounded_variables=canonicalize_bounded_variables(planning_state.scopes, scope_id), optimistic_execution=True)
                rv = rv.item()
                if isinstance(rv, OptimisticValue):
                    rv = False
                planning_state.statements_evaluations[id(stmt)] = bool(rv)
                if not bool(rv):
                    return False
            else:
                return False
        return True

    def _bfs_expand_inner(
        self, planning_state: _AStarNode, middle: Optional[CrowBehaviorOrderingSuite],
        stmt: Union[CrowAchieveExpression, CrowBehaviorApplicationExpression, CrowBindExpression, CrowRuntimeAssignmentExpression, CrowAssertExpression, CrowControllerApplicationExpression],
        expanded_planning_state: Optional[_AStarNode],
        scope_id: int, stmt_id: Optional[int] = None
    ):
        """Expand the tree by extracting the right-most statement and recursively expanding the left part or refine the current statement.

        Args:
            planning_state: the current planning state.
            middle: the rest of the middle programs that are being expanded.
            stmt: the statement to be expanded at this point.
            expanded_planning_state: if the stmt is a CrowAchieveExpression, this is the state where the achieve statement was serialized to the right stack.
            scope_id: the current scope id.
            stmt_id: the unique id of the statement. If it is None, it means that the statement is not a part of the main program body. This is used to prune some unnecessary expansions.
        """
        # print('Expanding inner:', stmt, 'with scope', scope_id, 'and left program', middle)
        # print('Expanded_state:', expanded_state)
        # import ipdb; ipdb.set_trace()

        if isinstance(stmt, CrowAchieveExpression):
            if middle is not None:
                new_state = planning_state.clone(program=middle)
                self.bfs_add_queue(planning_state.clone(program=middle, right_statements=(_ScopedStatement2(stmt, scope_id, expanded_state=new_state),) + planning_state.right_statements))
            else:
                rv, csp = self.evaluate(stmt.goal, state=planning_state.state, csp=planning_state.csp, bounded_variables=canonicalize_bounded_variables(planning_state.scopes, scope_id))
                if isinstance(rv, OptimisticValue):
                    # If the value is optimistic, we will add a constraint to the CSP and continue the search.
                    # But we also need to consider the case where the optimistic value is False, so we do not stop the branching here (no return).
                    self.bfs_add_queue(planning_state.clone(program=None, csp=csp.add_equal_constraint(rv, True)))
                else:
                    if bool(rv):
                        self.bfs_add_queue(planning_state.clone(program=None))
                        return
                first_time_expand = stmt_id not in self._expanded_right_first or stmt_id is None
                if stmt_id is not None:
                    self._expanded_right_first[stmt_id] = self._current_queue_state

                # if not first_time_expand:
                #     print('Already expanded. {{{', '-' * 60)
                for action_matching in match_applicable_behaviors(self.executor.domain, planning_state.state, stmt.goal, planning_state.scopes[scope_id]):
                    self._bfs_expand_inner_action(expanded_planning_state, expanded_planning_state.program, action_matching, scope_id, prefix_expanded_planning_state=planning_state, first_time_expand=first_time_expand)
                # if not first_time_expand:
                #     print('                  }}}', '-' * 60)

        elif isinstance(stmt, CrowBehaviorApplicationExpression):
            if middle is not None:
                new_state = planning_state.clone(program=middle)
                self.bfs_add_queue(planning_state.clone(program=middle, right_statements=(_ScopedStatement2(stmt, scope_id, expanded_state=new_state),) + planning_state.right_statements))
            else:
                first_time_expand = stmt_id not in self._expanded_right_first
                if stmt_id is not None:
                    self._expanded_right_first[stmt_id] = self._current_queue_state

                # if not first_time_expand:
                #     print('Already expanded. {{{', '-' * 60)

                # TODO(Jiayuan Mao @ 2024/03/27): think about which state should these actions be grounded on and how we should handle the CSP.
                argument_values = [self.executor.execute(x, state=planning_state.state, bounded_variables=canonicalize_bounded_variables(planning_state.scopes, scope_id)) for x in stmt.arguments]
                action_matching = ApplicableBehaviorItem(stmt.behavior, {k.name: v for k, v in zip(stmt.behavior.arguments, argument_values)})
                self._bfs_expand_inner_action(expanded_planning_state, expanded_planning_state.program, action_matching, scope_id, prefix_expanded_planning_state=planning_state, first_time_expand=first_time_expand)
                # if not first_time_expand:
                #     print('                  }}}', '-' * 60)
        elif isinstance(stmt, CrowBehaviorOrderingSuite):
            assert middle is None, 'The middle part should be empty for a program.'
            self.bfs_add_queue(planning_state.clone(program=stmt))
        elif isinstance(stmt, (CrowBindExpression, CrowRuntimeAssignmentExpression, CrowAssertExpression, CrowControllerApplicationExpression)):
            if middle is not None:
                self.bfs_add_queue(planning_state.clone(program=middle, right_statements=(_ScopedStatement2(stmt, scope_id),) + planning_state.right_statements))
            else:
                self._bfs_expand_inner_primitive(planning_state, stmt, scope_id)
        elif isinstance(stmt, _CrowBehaviorEffectApplication):
            if middle is not None:
                self.bfs_add_queue(planning_state.clone(program=middle, right_statements=(_ScopedStatement2(stmt, scope_id),) + planning_state.right_statements))
            else:
                self._bfs_expand_inner_action_effect(planning_state, stmt.behavior, scope_id)
        else:
            raise ValueError(f'Unknown statement type: {stmt}')

    def _bfs_expand_inner_action(self, planning_state: _AStarNode, middle: Optional[CrowBehaviorOrderingSuite], action_matching: ApplicableBehaviorItem, scope_id: int, prefix_expanded_planning_state: _AStarNode, first_time_expand: bool):
        """Expand the tree by refining a particular action.

        Args:
            planning_state: the current planning state.
            middle: the rest of the middle programs that are being expanded.
            action_matching: the action to be refined, including the action and the bounded variables.
            scope_id: the current scope id.
            prefix_expanded_planning_state: the planning search state assuming everything in the middle program has been refined separately without considering the current action.
            first_time_expand: whether this is the first time to expand the action. If it is not the first time, we will skip the expansion.
        """
        bounded_variables = action_matching.bounded_variables
        for var, value in bounded_variables.items():
            if isinstance(value, Variable):
                bounded_variables[var] = value.clone_with_scope(scope_id)

        if action_matching.behavior.is_sequential_only():
            new_scope_id = prefix_expanded_planning_state.latest_scope + 1
            program = action_matching.behavior.assign_body_program_scope(new_scope_id)
            preamble, promotable, rest = program.split_preamble_and_promotable()
            new_scopes = prefix_expanded_planning_state.scopes.copy()
            new_scopes[new_scope_id] = bounded_variables.copy()
            program = CrowBehaviorOrderingSuite.make_sequential(rest, variable_scope_identifier=new_scope_id)
            self.bfs_add_queue(prefix_expanded_planning_state.clone(
                program=program, scopes=new_scopes, latest_scope=new_scope_id,
                right_statements=(_ScopedStatement2(_CrowBehaviorEffectApplication(action_matching.behavior), new_scope_id),) + prefix_expanded_planning_state.right_statements
            ))
            return

        if not first_time_expand:
            return

        new_scope_id = planning_state.latest_scope + 1
        program = action_matching.behavior.assign_body_program_scope(new_scope_id)
        preamble, promotable, rest = program.split_preamble_and_promotable()
        new_scopes = planning_state.scopes.copy()
        new_scopes[new_scope_id] = bounded_variables.copy()

        if preamble is not None:
            new_left_program = CrowBehaviorOrderingSuite.make_sequential(preamble, variable_scope_identifier=new_scope_id)
            if middle is not None:
                new_middle_program = CrowBehaviorOrderingSuite.make_unordered(middle, CrowBehaviorOrderingSuite.make_sequential(promotable, variable_scope_identifier=new_scope_id)) if promotable is not None else middle
            else:
                new_middle_program = CrowBehaviorOrderingSuite.make_sequential(promotable, variable_scope_identifier=new_scope_id) if promotable is not None else None
            new_right_program = CrowBehaviorOrderingSuite.make_sequential(rest, variable_scope_identifier=new_scope_id)
            new_right_statements = [_ScopedStatement2(new_right_program, new_scope_id), _ScopedStatement2(_CrowBehaviorEffectApplication(action_matching.behavior), new_scope_id)]
            if new_middle_program is not None:
                new_right_statements.insert(0, _ScopedStatement2(new_middle_program, new_scope_id))
            self.bfs_add_queue(planning_state.clone(program=new_left_program, scopes=new_scopes, latest_scope=new_scope_id, right_statements=tuple(new_right_statements) + planning_state.right_statements))
        else:
            if promotable is not None:
                if middle is not None:
                    new_left_program = CrowBehaviorOrderingSuite.make_unordered(middle, CrowBehaviorOrderingSuite.make_sequential(promotable, variable_scope_identifier=new_scope_id))
                else:
                    new_left_program = CrowBehaviorOrderingSuite.make_sequential(promotable, variable_scope_identifier=new_scope_id)
                new_right_program = CrowBehaviorOrderingSuite.make_sequential(rest, variable_scope_identifier=new_scope_id)
                new_right_statements = (_ScopedStatement2(new_right_program, new_scope_id), _ScopedStatement2(_CrowBehaviorEffectApplication(action_matching.behavior), new_scope_id)) + planning_state.right_statements
                self.bfs_add_queue(planning_state.clone(program=new_left_program, scopes=new_scopes, latest_scope=new_scope_id, right_statements=new_right_statements))
            else:
                raise RuntimeError('Should not reach here. This case should have been already handled by the action.is_sequential_only() check.')

    def _bfs_expand_inner_primitive(self, state: _AStarNode, stmt: Union[CrowBindExpression, CrowRuntimeAssignmentExpression, CrowAssertExpression, CrowControllerApplicationExpression], scope_id: int):
        """Expand the tree by refining a particular primitive statement."""
        if isinstance(stmt, CrowControllerApplicationExpression):
            new_csp = state.csp.clone()
            argument_values = [self.evaluate(x, state=state.state, csp=new_csp, bounded_variables=canonicalize_bounded_variables(state.scopes, scope_id), clone_csp=False)[0] for x in stmt.arguments]
            for i, argv in enumerate(argument_values):
                if isinstance(argv, StateObjectReference):
                    argument_values[i] = argv.name
            self.bfs_add_queue(state.clone(program=None, left_statements=state.left_statements + (CrowControllerApplier(stmt.controller, argument_values),), csp=new_csp))
        elif isinstance(stmt, CrowBindExpression):
            if stmt.is_object_bind:
                for new_scope in execute_object_bind(self.executor, stmt, state.state, canonicalize_bounded_variables(state.scopes, scope_id)):
                    new_scopes = state.scopes.copy()
                    new_scopes[scope_id] = new_scope
                    self.bfs_add_queue(state.clone(program=None, scopes=new_scopes))
            else:
                new_csp = state.csp.clone()
                new_scopes = state.scopes.copy()
                for var in stmt.variables:
                    new_scopes[scope_id][var] = TensorValue.from_optimistic_value(new_csp.new_var(var.dtype, wrap=True))
                rv, new_csp = self.evaluate(stmt.goal, state=state.state, csp=new_csp, bounded_variables=canonicalize_bounded_variables(new_scopes, scope_id))
                self.bfs_add_queue(state.clone(program=None, scopes=new_scopes, csp=new_csp.add_equal_constraint(rv, True)))
        elif isinstance(stmt, CrowRuntimeAssignmentExpression):
            rv, new_csp = self.evaluate(stmt.value, state=state.state, csp=state.csp, bounded_variables=canonicalize_bounded_variables(state.scopes, scope_id))
            new_scopes = state.scopes.copy()
            new_scopes[scope_id] = state.scopes[scope_id].copy()
            new_scopes[scope_id][stmt.variable.name] = rv
            self.bfs_add_queue(state.clone(program=None, scopes=new_scopes, csp=new_csp))
        elif isinstance(stmt, CrowAssertExpression):
            rv, new_csp = self.evaluate(stmt.bool_expr, state=state.state, csp=state.csp, bounded_variables=canonicalize_bounded_variables(state.scopes, scope_id))
            if isinstance(rv, OptimisticValue):
                self.bfs_add_queue(state.clone(program=None, csp=new_csp.add_equal_constraint(rv, True)))
            else:
                if bool(rv):
                    self.bfs_add_queue(state.clone(program=None))
        else:
            raise ValueError(f'Unknown statement type: {stmt}')

    def _bfs_expand_inner_action_effect(self, planning_state: _AStarNode, stmt: CrowBehavior, scope_id: int):
        new_csp = planning_state.csp.clone() if planning_state.csp is not None else None
        new_state = execute_behavior_effect(
            self.executor, stmt, planning_state.state, canonicalize_bounded_variables(planning_state.scopes, scope_id), csp=new_csp,
            state_index=len(planning_state.left_statements)
        )
        self.bfs_add_queue(planning_state.clone(program=None, state=new_state, csp=new_csp))
