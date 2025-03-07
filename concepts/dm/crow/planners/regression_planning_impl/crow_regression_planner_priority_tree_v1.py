#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : crow_regression_planner_priority_tree_v1.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/11/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import re
import weakref
from typing import Any, Optional, Union, Iterator, Tuple, NamedTuple, List, Dict, TYPE_CHECKING
from dataclasses import dataclass, field

import jacinle
from jacinle.storage.unsafe_queue import PriorityQueue
from jacinle.utils.meta import UNSET

from concepts.dsl.constraint import ConstraintSatisfactionProblem, OptimisticValue
from concepts.dsl.dsl_types import Variable
from concepts.dsl.expression import VariableExpression, ObjectConstantExpression, ConstantExpression, NotExpression, ValueOutputExpression
from concepts.dsl.value import ListValue
from concepts.dsl.tensor_value import TensorValue
from concepts.dsl.tensor_state import StateObjectReference

from concepts.dm.crow.controller import CrowControllerApplier, CrowControllerApplicationExpression
from concepts.dm.crow.behavior import CrowBehaviorOrderingSuite, CrowBehaviorCommit
from concepts.dm.crow.behavior import CrowBindExpression, CrowMemQueryExpression, CrowAssertExpression, CrowRuntimeAssignmentExpression, CrowAchieveExpression, CrowUntrackExpression, CrowBehaviorApplicationExpression, CrowBehaviorEffectApplicationExpression
from concepts.dm.crow.behavior import CrowBehaviorForeachLoopSuite, CrowBehaviorWhileLoopSuite, CrowBehaviorConditionSuite, CrowEffectApplier
from concepts.dm.crow.behavior_utils import match_applicable_behaviors, match_policy_applicable_behaviors, ApplicableBehaviorItem
from concepts.dm.crow.behavior_utils import format_behavior_statement, format_behavior_program, execute_behavior_effect_body, execute_object_bind, execute_additive_heuristic_program

from concepts.dm.crow.planners.regression_planning import SupportedCrowExpressionType, ScopedCrowExpression, CrowPlanningResult3, CrowRegressionPlanner
from concepts.dm.crow.planners.regression_dependency import RegressionTraceStatement
from concepts.dm.crow.planners.regression_utils import replace_variable_with_value, format_regression_statement
from concepts.dm.crow.csp_solver.dpll_sampling import dpll_solve
from concepts.dm.crow.csp_solver.csp_utils import csp_ground_action_list, csp_ground_state

if TYPE_CHECKING:
    from concepts.dm.crow.planners.priority_impl.priority_tree_priority_fns import PriorityFunctionBase

"""TODOs for 2024/08/14

- Adding a small heuristic to "promote" let/bind statements to the L section.
"""


class MRProgram(NamedTuple):
    middle: Optional[CrowBehaviorOrderingSuite]
    right: Tuple[ScopedCrowExpression, ...]


class PriorityTreeNodeData(NamedTuple):
    result: CrowPlanningResult3

    left: Tuple[ScopedCrowExpression, ...]
    middle: Optional[CrowBehaviorOrderingSuite]
    right: Tuple[ScopedCrowExpression, ...]
    minimize: Optional[ValueOutputExpression]

    @classmethod
    def make_empty(cls, state, middle, minimize: Optional[ValueOutputExpression]) -> 'PriorityTreeNodeData':
        return cls(CrowPlanningResult3.make_empty(state), tuple(), middle, tuple(), minimize)

    def clone(self, result=UNSET, left=UNSET, middle=UNSET, right=UNSET, minimize=UNSET) -> 'PriorityTreeNodeData':
        return PriorityTreeNodeData(
            result=self.result if result is UNSET else result,
            left=self.left if left is UNSET else left,
            middle=self.middle if middle is UNSET else middle,
            right=self.right if right is UNSET else right,
            minimize=self.minimize if minimize is UNSET else minimize
        )

    def is_empty(self) -> bool:
        return len(self.left) == 0 and self.middle is None and len(self.right) == 0

    def iter_from_right(self) -> Iterator[Tuple[
        'PriorityTreeNodeData', Union[SupportedCrowExpressionType, MRProgram], Optional[CrowBehaviorOrderingSuite], int]]:
        """This is a helper function to iterate over the right-most statement of the program.

        - When `self.left` is not empty, this function moves the left program to the right section.

            (L, M, R) <- ([], [], L) || ([], M, R)

        - When `self.right` is not empty, this function simply pops the right-most statement and returns the new planning state.

            ([], M, [R' | tail]) <- tail || ([], M, R')

        - When `self.right` is empty, this function iterates over the right-most statement of the middle program.

            ([], (M' | tail), []) <- tail || ([], M', [])

        """

        if len(self.left) != 0:
            yield self.clone(left=tuple(), middle=None, right=self.left), MRProgram(self.middle, self.right), None, 0
            return

        if len(self.right) != 0:
            if isinstance(self.right[-1].statement, CrowBehaviorOrderingSuite):
                for middle, stmt, scope_id in self.right[-1].statement.pop_right_statement():
                    if middle is None:
                        yield self.clone(right=self.right[:-1]), stmt, None, scope_id
                    else:
                        new_data = self.clone(right=self.right[:-1] + (ScopedCrowExpression(middle, scope_id),))
                        yield new_data, stmt , None, scope_id
            else:
                stmt, right_stmt = self.right[-1], self.right[:-1]
                new_data = self.clone(right=right_stmt)
                yield new_data, stmt.statement, None, stmt.scope_id
            return

        if self.middle is not None:
            for middle, stmt, scope_id in self.middle.pop_right_statement():
                new_data = self.clone(middle=middle)
                yield new_data, stmt, middle, scope_id
            return

    def short_str(self):
        action_string = '[' + '; '.join(str(x) for x in self.result.controller_actions) + ']'
        left_string = '[' + '; '.join(re.sub(r'\n\s*', ' ', format_regression_statement(x, self.result.scopes)) for x in self.left) + ']'
        middle_string = str(self.middle).replace('\n', ' ') if self.middle is not None else '<None>'
        right_string = '[' + '; '.join(re.sub(r'\n\s*', ' ', format_regression_statement(x, self.result.scopes)) for x in self.right) + ']'
        return f'@{hex(id(self))} A={action_string} L={left_string}, M={middle_string}, R={right_string}'


@dataclass
class PriorityTreeNode(object):
    data: PriorityTreeNodeData
    parents: List[Tuple[Optional[Union[ScopedCrowExpression, MRProgram]], 'PriorityTreeNode']] = field(default_factory=list)
    children: List['PriorityTreeNode'] = field(default_factory=list)
    results: List[CrowPlanningResult3] = field(default_factory=list)

    # Other statistics related to the node.

    depth: int = 0
    """The depth of the node in the tree."""

    g: Optional[float] = None
    """The G value of the node = g(node.data.result)"""
    node_h: Optional[float] = None
    """The H value of the node = h(node.data)"""
    accumulated_edge_h: Optional[float] = None
    """The accumulated H value of the edges = h(parent_edges)"""
    accumulated_edge_h_parent = None
    """The (stmt, parent) pair that contributes to the min accumulated edge H."""

    def add_child(self, child: 'PriorityTreeNode', stmt: Optional[Union[ScopedCrowExpression, MRProgram]]):
        self.children.append(weakref.ref(child))
        child.parents.append((stmt, self))
        child.depth = max(self.depth + 1, child.depth)

    def iter_parent_links(self) -> Iterator[Tuple[Optional[Union[SupportedCrowExpressionType, MRProgram]], int, 'PriorityTreeNode']]:
        for expr, parent_node in self.parents:
            if expr is None:
                yield None, -1, parent_node
            elif isinstance(expr, MRProgram):
                yield expr, -1, parent_node
            else:
                yield expr.statement, expr.scope_id, parent_node


@dataclass
class _QueueNode(object):
    node: PriorityTreeNode
    result: Optional[CrowPlanningResult3] = None
    priority: float = 0

    def compute_priority(self, fn: 'PriorityFunctionBase') -> '_QueueNode':
        self.priority = fn.get_priority(self.node)
        return self


def hash_state_human(planning_state: PriorityTreeNodeData) -> Tuple[Any, ...]:
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
    hash_l_statements = tuple(format_regression_statement(x, planning_state.result.scopes) for x in planning_state.left)
    hash_program = format_behavior_program(planning_state.middle, planning_state.result.scopes, flatten=True) if planning_state.middle is not None else '<None>'
    hash_r_statements = tuple(format_regression_statement(x, planning_state.result.scopes) for x in planning_state.right)
    hash_actions = tuple(str(x) for x in planning_state.result.controller_actions)
    hash_scopes = '\n'.join(f'{k}: {v}' for k, v in planning_state.result.scopes.items())
    hash_scope_constraints_parts = list()
    for scope_id, constraints in planning_state.result.scope_constraints.items():
        constraints = '; '.join(format_behavior_statement(x, scopes=planning_state.result.scopes, scope_id=scope_id) for x in constraints)
        hash_scope_constraints_parts.append(f'{scope_id}: {constraints}')
    hash_scope_constraints = '\n'.join(hash_scope_constraints_parts)
    hash_latest_scope_index = planning_state.result.latest_scope

    planning_state_hash = (hash_l_statements, hash_program, hash_r_statements, hash_actions, hash_scopes, hash_scope_constraints, hash_latest_scope_index)
    return planning_state_hash


def hash_state(planning_state: PriorityTreeNodeData) -> Tuple[Any, ...]:
    hash_l_statements = tuple(format_regression_statement(x, None) for x in planning_state.left)
    hash_program = format_behavior_program(planning_state.middle, None, flatten=True) if planning_state.middle is not None else '<None>'
    hash_r_statements = tuple(format_regression_statement(x, None) for x in planning_state.right)
    hash_actions = tuple(str(x) for x in planning_state.result.controller_actions)
    hash_scopes = '\n'.join(f'{k}: {v}' for k, v in planning_state.result.scopes.items())
    hash_scope_constraints = '\n'.join(f'{k}: {v}' for k, v in planning_state.result.scope_constraints.items())
    hash_latest_scope_index = planning_state.result.latest_scope
    planning_state_hash = (hash_l_statements, hash_program, hash_r_statements, hash_actions, hash_scopes, hash_scope_constraints, hash_latest_scope_index)
    return planning_state_hash


class _SolutionFound(Exception):
    pass


class CrowRegressionPlannerPriorityTreev1(CrowRegressionPlanner):
    def _post_init(self, always_commit_skeleton: bool = False, enable_state_hash: bool = False, priority_fn: str = 'fifo') -> None:
        # Ignroe always_commit_skeleton. It's for backward compatibility with iddfs_v1.
        self.enable_state_hash = enable_state_hash
        self.priority_fn = priority_fn
        self.priority_fn_impl = self._get_priority_fn_impl(self.priority_fn)

        self._root = None
        self._results_best_cost = None
        self._queue = PriorityQueue()
        self._visited_node_data = dict()

    _queue: PriorityQueue

    """Whether to always commit the skeleton of the program."""

    enable_state_hash: bool
    """Whether to enable the state hash."""

    priority_fn: str
    """The priority function to use."""

    _enable_visited_states_count: bool
    """Whether to enable the visited states count."""

    _visited_node_data: Dict[Tuple[str, ...], PriorityTreeNode]
    """The visited states. The value is a tuple of (generator, depth)."""

    def _get_priority_fn_impl(self, priority_fn: str) -> 'PriorityFunctionBase':
        import concepts.dm.crow.planners.priority_impl.priority_tree_priority_fns as priority_fns
        if priority_fn == 'fifo':
            return priority_fns.FIFOPriorityFunction(self.executor)
        elif priority_fn == 'lifo':
            return priority_fns.LIFOPriorityFunction(self.executor)
        elif priority_fn == 'unit_cost_astar':
            return priority_fns.UnitCostPriorityFunction(self.executor, priority_fns.WeightedAStarMixFunction())
        elif priority_fn == 'unit_cost_best_first':
            return priority_fns.UnitCostPriorityFunction(self.executor, priority_fns.BestFirstMixFunction())
        elif priority_fn == 'unit_cost_longest_first':
            return priority_fns.UnitCostPriorityFunction(self.executor, priority_fns.GFirstMixFunction())
        elif priority_fn == 'simple_unit_cost_astar':
            return priority_fns.SimpleUnitCostPriorityFunction(self.executor, priority_fns.WeightedAStarMixFunction())
        elif priority_fn == 'simple_unit_cost_best_first':
            return priority_fns.SimpleUnitCostPriorityFunction(self.executor, priority_fns.BestFirstMixFunction())
        elif priority_fn == 'simple_additive_astar':
            return priority_fns.SimpleAdditivePriorityFunction(self.executor, priority_fns.WeightedAStarMixFunction())
        elif priority_fn == 'simple_additive_best_first':
            return priority_fns.SimpleAdditivePriorityFunction(self.executor, priority_fns.BestFirstMixFunction())
        else:
            raise ValueError(f'Invalid priority function: {priority_fn}.')

    def main_entry(self, program: CrowBehaviorOrderingSuite, minimize: Optional[ValueOutputExpression] = None) -> List[Tuple[CrowControllerApplier, ...]]:
        self._queue = PriorityQueue()
        self._visited_node_data.clear()
        self._results = None
        self._results_best_cost = None

        root_data = PriorityTreeNodeData.make_empty(self.state, program, minimize)
        _init_scope_from(root_data.result.scopes[0], -1, {}, overwrite_commit=True)

        self._root = self._push_node(None, root_data, None)

        try:
            while not self._queue.empty():
                node, result = self._pop_node()
                self._search_stat['nr_expanded_nodes'] += 1

                if result is not None:
                    self._expand_result(node, result)
                else:
                    self._expand_node(node)
        except _SolutionFound:
            pass

        if self._results is not None:
            return [x.controller_actions for x in self._results]

    def main_continue(self, root_node: PriorityTreeNode, current_node: PriorityTreeNode, current_result: CrowPlanningResult3) -> List[Tuple[CrowControllerApplier, ...]]:
        self._queue = PriorityQueue()
        self._visited_node_data.clear()
        self._results = None
        self._results_best_cost = None

        self._root = root_node
        self._push_node_result(current_node, current_result)

        try:
            while not self._queue.empty():
                node, result = self._pop_node()
                self._search_stat['nr_expanded_nodes'] += 1

                if result is not None:
                    self._expand_result(node, result)
                else:
                    self._expand_node(node)
        except _SolutionFound:
            pass

        if self._results is not None:
            return [x.controller_actions for x in self._results]

    def set_results(self, results: List[CrowPlanningResult3]) -> None:
        if self.enable_csp:
            new_results = list()
            for result in results:
                new_results.extend(self._solve_csp(result))
            results = new_results

        if len(results) == 0:
            return

        if self._results is None:
            self._results = results
        else:
            self._results.extend(results)

        from concepts.dm.crow.planners.priority_impl.priority_tree_priority_fns import AdditivePriorityFunctionBase
        if isinstance(self.priority_fn_impl, AdditivePriorityFunctionBase):
            for r in results:
                r.cost = self.priority_fn_impl.get_result_g(r, self._root.data.minimize)
                if self._results_best_cost is None or r.cost < self._results_best_cost:
                    self._results_best_cost = r.cost
            if not self._queue.empty():
                top_node_cost = self._queue.peek().priority
            else:
                top_node_cost = float('inf')

            if self._results_best_cost is not None and self._results_best_cost <= top_node_cost:
                # print(f'Early stopping: best_cost={self._results_best_cost} top_node_cost={top_node_cost}')
                # Sort the results by the cost.
                self._results.sort(key=lambda x: x.cost)
                raise _SolutionFound()

            # else:
            #     print(f'Continue searching: best_cost={self._results_best_cost} top_node_cost={top_node_cost}')
        else:
            raise _SolutionFound()

    def _pop_node(self) -> Tuple[PriorityTreeNode, Optional[CrowPlanningResult3]]:
        node = self._queue.get()
        return node.node, node.result

    def _push_node(self, parent_node: Optional[PriorityTreeNode], node_data: PriorityTreeNodeData, stmt: Optional[Union[ScopedCrowExpression, MRProgram]], enqueue: bool = True):
        # For now, just create a new node.

        if self.enable_state_hash:
            data_hash = hash_state(node_data)
            is_new_node = False
            if data_hash in self._visited_node_data:
                child_node = self._visited_node_data[data_hash]
            else:
                is_new_node = True
                child_node = PriorityTreeNode(node_data)
                self._visited_node_data[data_hash] = child_node
        else:
            is_new_node = True
            child_node = PriorityTreeNode(node_data)

        if parent_node is not None:
            parent_node.add_child(child_node, stmt)

        pushed = False
        if is_new_node and enqueue:
            pushed = True
            queue_node = _QueueNode(child_node).compute_priority(self.priority_fn_impl)
            self._queue.put(queue_node, priority=queue_node.priority)

        if not is_new_node:
            if stmt is None:
                stmt, scope_id = None, -1
            elif isinstance(stmt, MRProgram):
                stmt, scope_id = stmt, -1
            else:
                assert isinstance(stmt, ScopedCrowExpression)
                stmt, scope_id = stmt.statement, stmt.scope_id

            for result in child_node.results:
                self._expand_result_with_stmt(result, stmt, scope_id, parent_node)

        if self.verbose:
            stmt_string = str(stmt).replace('\n', ' ') if stmt is not None else '<None>'
            print(f'  Pushing node: {child_node.data.short_str()} || stmt={stmt_string} (priority={queue_node.priority if pushed else None})')
            if parent_node is not None:
                print(f'    Parent: {parent_node.data.short_str()}')
            if not pushed and enqueue:
                print('    Node ignored.')

        return child_node

    def _push_node_result(self, result_node: PriorityTreeNode, result: CrowPlanningResult3):
        if self.verbose:
            print(f'  Pushing result: {result_node.data.short_str()}')
            print(f'    Result: A={result.controller_actions}')

        result_node.results.append(result)
        queue_node = _QueueNode(result_node, result).compute_priority(self.priority_fn_impl)
        self._queue.put(queue_node, priority=queue_node.priority)

    def _expand_result(self, child_node: PriorityTreeNode, child_result: CrowPlanningResult3):
        if self.verbose:
            print(f'Expanding result: {child_node.data.short_str()}')
            print(f'  Result: A={child_result.controller_actions}')

        if child_node is self._root:
            child_result.planner_root_node = self._root
            child_result.planner_current_node = child_node
            self.set_results([child_result])

        # result || stmt -> parent_node.program
        for stmt, scope_id, parent_node in child_node.iter_parent_links():
            if self.verbose:
                print('  > Parent Link:', stmt, scope_id)
                print(f'  > Parent: {parent_node.data.short_str()}')

            self._expand_result_with_stmt(child_result, stmt, scope_id, parent_node)

    def _expand_result_with_stmt(self, child_result: CrowPlanningResult3, stmt: Union[SupportedCrowExpressionType, MRProgram, None], scope_id: int, parent_node: PriorityTreeNode):
        if stmt is None:
            self._push_node_result(parent_node, child_result)
        elif isinstance(stmt, MRProgram):
            self._push_node(parent_node, PriorityTreeNodeData(child_result, tuple(), stmt.middle, stmt.right, parent_node.data.minimize), None)
        elif isinstance(stmt, CrowBehaviorCommit):
            for child_result in self._handle_commit_statement(child_result, stmt, scope_id):
                if stmt.execution:
                    new_scopes = child_result.scopes.copy()
                    new_scopes[scope_id]['__commit_execution__'] = TensorValue.from_scalar(True)
                    new_result = child_result.clone(scopes=new_scopes)
                    self._maybe_commit_result(new_result, scope_id, parent_node)
                    self._push_node_result(parent_node, new_result)
                else:
                    self._push_node_result(parent_node, child_result)
        elif isinstance(stmt, CrowAchieveExpression):
            self._handle_result_achieve_statement(parent_node, child_result, stmt, scope_id)
        elif isinstance(stmt, CrowBehaviorApplicationExpression):
            self._handle_result_achieve_statement(parent_node, child_result, stmt, scope_id)
        elif isinstance(stmt, CrowBehaviorOrderingSuite):
            raise NotImplementedError()
        elif isinstance(stmt, CrowBehaviorForeachLoopSuite):
            self._handle_result_foreach_statement(parent_node, child_result, stmt, scope_id)
        elif isinstance(stmt, CrowBehaviorWhileLoopSuite):
            raise NotImplementedError()
        elif isinstance(stmt, CrowBehaviorConditionSuite):
            self._handle_result_condition_statement(parent_node, child_result, stmt, scope_id)
        elif isinstance(stmt, (CrowBindExpression, CrowMemQueryExpression, CrowAssertExpression, CrowUntrackExpression, CrowRuntimeAssignmentExpression, CrowControllerApplicationExpression)):
            for applied_result in self._handle_primitive_statement(child_result, stmt, scope_id):
                self._push_node_result(parent_node, self._maybe_annotate_dependency(child_result, applied_result, stmt, scope_id))
        elif isinstance(stmt, CrowBehaviorEffectApplicationExpression):
            for applied_result in self._handle_behavior_effect_application(child_result, stmt, scope_id):
                self._push_node_result(parent_node, self._maybe_annotate_dependency(child_result, applied_result, stmt, scope_id))
        else:
            raise ValueError(f'Invalid statement type for dfs_inner: {stmt}.')

    def _handle_commit_statement(self, result: CrowPlanningResult3, stmt: CrowBehaviorCommit, scope_id: int) -> List[CrowPlanningResult3]:
        if stmt.csp:
            yield from self._solve_csp(result)
        else:
            yield result

    def _maybe_commit_result(self, result: CrowPlanningResult3, scope_id: int, parent_node: PriorityTreeNode) -> None:
        if len(result.controller_actions) == 0:
            return

        all_passed = True
        while True:
            if scope_id < 0:
                break

            if not result.scopes[scope_id]['__commit_execution__'].item():
                all_passed = False
                break

            scope_id = result.scopes[scope_id]['__parent__'].item()

        if all_passed:
            result.planner_root_node = self._root
            result.planner_current_node = parent_node
            self.set_results([result])

    def _expand_node(self, node: PriorityTreeNode):
        data = node.data

        if self.verbose:
            print(f'Expanding node: {data.short_str()}')
            # print(f'  g={node.get_g(self.executor)} parent_g={node.get_accumulated_edge_g(self.executor)}')
            # x = node
            # while x.g_parent is not None:
            #     stmt, parent = x.g_parent
            #     if stmt is not None:
            #         print(f'  | g={_compute_simple_g(stmt)} || {stmt} -> {parent.data.short_str()}')
            #     x = parent

        if data.is_empty():
            self._push_node_result(node, data.result)  # Boundary condition. The program is now already empty (hitting a leaf node).
        else:
            for new_data, stmt, middle_program, scope_id in data.iter_from_right():
                if new_data.is_empty():
                    self._expand_result_with_stmt(data.result, stmt, scope_id, node)
                    continue

                if isinstance(stmt, MRProgram):
                    self._push_node(node, new_data, stmt)
                    continue

                if middle_program is not None and isinstance(stmt, (CrowAchieveExpression, CrowBehaviorApplicationExpression)):
                    self._handle_node_achieve_statement(node, new_data, stmt, middle_program, scope_id)
                else:
                    # TODO(Jiayuan Mao @ 2024/08/11): do we need to do this "naive" expansion when middle program is present?
                    self._push_node(node, new_data, ScopedCrowExpression(stmt, scope_id))

    def _handle_node_achieve_statement(self, parent_node: PriorityTreeNode, new_data: PriorityTreeNodeData, stmt: Union[CrowAchieveExpression, CrowBehaviorApplicationExpression], middle_program: CrowBehaviorOrderingSuite, scope_id: int):
        assert len(new_data.left) == 0
        if isinstance(stmt, CrowAchieveExpression):
            self._push_node(parent_node, new_data, ScopedCrowExpression(CrowAssertExpression(stmt.goal, once=stmt.once), scope_id))
        self._handle_achieve_statement(
            parent_node, None,
            stmt, scope_id, middle_program
        )

    def _handle_result_achieve_statement(self, parent_node: PriorityTreeNode, child_result: CrowPlanningResult3, stmt: Union[CrowAchieveExpression, CrowBehaviorApplicationExpression], scope_id):
        # TODO(Jiayuan Mao @ 2024/08/12): handle this assert later!
        # print('  Handling result achieve statement:', format_behavior_statement(stmt, scopes=child_result.scopes, scope_id=scope_id))
        # for node in self._queue.queue:
        #     print(f'   | Node:', node.node.data.short_str())
        self._handle_achieve_statement(
            parent_node, child_result,
            stmt, scope_id, None
        )

    def _handle_achieve_statement(
        self, parent_node: PriorityTreeNode, child_result: Optional[CrowPlanningResult3],
        stmt: Union[CrowAchieveExpression, CrowBehaviorApplicationExpression], scope_id: int,
        middle_program: Optional[CrowBehaviorOrderingSuite]
    ) -> None:
        # NB(Jiayuan Mao @ 2024/07/13): AchieveExpression statements and BehaviorApplicationExpression statements are handled in a different way.
        # 1. For AchieveExpression statements, the matching is done at a pure "syntax" level. We are trying to exactly match the surface form of the goal expression with other behaviors.
        # For example, if the goal is `pose(x) == pose(y) + [1, 0, 0]`, we are trying to find a behavior that has the same surface form, such as `pose(x) == pose(y) + z`.
        # 2. For BehaviorApplicationExpression statements, the matching is done at a "semantic" level. We are going to evaluate the arguments of the expression.
        # Therefore, this semantic evaluation must be done later, after we have obtained the `result` of the other subgoals.

        result = child_result if child_result is not None else parent_node.data.result

        if middle_program is None and isinstance(stmt, CrowAchieveExpression):
            # Evaluate the goal directly.
            rv, new_csp = self.evaluate(stmt.goal, state=result.state, csp=result.csp, bounded_variables=result.scopes[scope_id], clone_csp=True, state_index=result.get_state_index())

            if isinstance(rv, OptimisticValue):
                new_result = result.clone(
                    csp=new_csp.add_equal_constraint(rv, True),
                    dependency_trace=result.dependency_trace + (RegressionTraceStatement(CrowAssertExpression(stmt.goal, once=stmt.once), scope_id, scope=result.scopes[scope_id], additional_info='skip', derived_from=stmt), ) if self.include_dependency_trace else tuple()
                ).clone_with_new_constraint(scope_id, stmt.goal, True, do=not stmt.once)
                self._push_node_result(parent_node, new_result)
            else:
                if bool(rv):
                    new_result = result.clone_with_new_constraint(scope_id, stmt.goal, True, do=not stmt.once).clone(
                        dependency_trace=result.dependency_trace + (RegressionTraceStatement(CrowAssertExpression(stmt.goal, once=stmt.once), scope_id, scope=result.scopes[scope_id], additional_info='skip', derived_from=stmt), ) if self.include_dependency_trace else tuple()
                    )
                    self._push_node_result(parent_node, new_result)
                else:
                    pass

        if isinstance(stmt, CrowAchieveExpression) and not stmt.is_policy_achieve:
            behavior_matches = match_applicable_behaviors(self.executor.domain, result.state, stmt.goal, result.scopes[scope_id])
        elif isinstance(stmt, CrowAchieveExpression) and stmt.is_policy_achieve:
            behavior_matches = match_policy_applicable_behaviors(self.executor.domain, result.state, stmt.goal, result.scopes[scope_id], pachieve_kwargs=stmt.flags)
        elif isinstance(stmt, CrowBehaviorApplicationExpression):
            # TODO(Jiayuan Mao @ 2024/08/11): handle CSP correctly.
            behavior_matches = [ApplicableBehaviorItem(stmt.behavior, {
                k.name: self.evaluate(v, state=result.state, csp=result.csp, bounded_variables=result.scopes[scope_id], clone_csp=False, force_tensor_value=True)[0]
                for k, v in zip(stmt.behavior.arguments, stmt.arguments)
            })]
        else:
            raise ValueError(f'Invalid statement type for handle_achieve_statement: {stmt}.')

        if self.verbose:
            behavior_matches_table = list()
            for behavior_match in behavior_matches:
                behavior_matches_table.append((str(behavior_match.behavior), str(behavior_match.bounded_variables)))
            print('    Behavior matching for achieve stmt:', stmt)
            string = '\n'.join(['    Behavior matches:', jacinle.tabulate([(i, b.behavior, b.bounded_variables) for i, b in enumerate(behavior_matches)], headers=['Index', 'Behavior', 'Bounded Variables'])])
            print(jacinle.indent_text(string, indent_format='    '))

        # Case 1: Try to apply the behavior after other subgoals have been achieved.
        if middle_program is None:
            for behavior_match in behavior_matches:
                self._handle_achieve_statement_inner1(parent_node, result, behavior_match, stmt, scope_id)

        # Case 2: Try to apply the behavior together with the "middle program"
        if middle_program is not None:
            for behavior_match in behavior_matches:
                # if not behavior_match.behavior.is_sequential_only():  # If it is sequential only, it should have been already handled in the previous step.
                self._handle_achieve_statement_inner2(parent_node, middle_program, behavior_match, stmt, scope_id)

    def _handle_achieve_statement_inner1(
        self, parent_node: PriorityTreeNode, result: CrowPlanningResult3,
        behavior_match: ApplicableBehaviorItem, derived_from_stmt: Union[CrowAchieveExpression, CrowBehaviorApplicationExpression], scope_id: int
    ) -> None:
        # NB(Jiayuan Mao @ 2024/05/31): The input `scope_id` is the scope where the statement (the matched behavior) is being applied.
        new_scope_id = result.latest_scope + 1
        new_scopes = result.scopes.copy()
        new_scopes[new_scope_id] = _init_scope_from(_resolve_bounded_variables(behavior_match.bounded_variables, result.scopes[scope_id]), scope_id, result.scopes[scope_id])
        program = behavior_match.behavior.assign_body_program_scope(new_scope_id)
        L_prime, M_prime, R_prime = program.split_preamble_and_promotable()

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

        L_prime = tuple(ScopedCrowExpression(x, new_scope_id) for x in L_prime) if L_prime is not None else tuple()
        if behavior_match.behavior.always:
            L_prime = (ScopedCrowExpression(CrowBehaviorCommit.execution_only(), new_scope_id), ) + L_prime
        M_prime = CrowBehaviorOrderingSuite.make_sequential(M_prime, variable_scope_identifier=new_scope_id) if M_prime is not None else None
        R_prime = tuple(ScopedCrowExpression(x, new_scope_id) for x in R_prime)

        if not behavior_match.behavior.effect_body.is_empty():
            R_prime += (ScopedCrowExpression(CrowBehaviorEffectApplicationExpression(behavior_match.behavior), new_scope_id),)

        # Add assert to L_prime that the goal is achieved.
        if isinstance(derived_from_stmt, CrowAchieveExpression):
            L_prime = (ScopedCrowExpression(CrowAssertExpression(NotExpression(derived_from_stmt.goal), once=True), scope_id),) + L_prime
        if isinstance(derived_from_stmt, CrowAchieveExpression) and derived_from_stmt.once == False:
            R_prime += (ScopedCrowExpression(CrowAssertExpression(derived_from_stmt.goal, once=False), scope_id),)

        new_minimize = behavior_match.behavior.minimize if behavior_match.behavior.minimize is not None else parent_node.data.minimize
        self._push_node(parent_node, PriorityTreeNodeData(result, L_prime, M_prime, R_prime, new_minimize), None)

    def _handle_achieve_statement_inner2(
        self, parent_node: PriorityTreeNode, middle_program: CrowBehaviorOrderingSuite,
        behavior_match: ApplicableBehaviorItem, derived_from_stmt: Union[CrowAchieveExpression, CrowBehaviorApplicationExpression], scope_id: int
    ) -> None:
        data = parent_node.data
        new_scope_id = data.result.latest_scope + 1
        new_scopes = data.result.scopes.copy()
        new_scopes[new_scope_id] = _init_scope_from(_resolve_bounded_variables(behavior_match.bounded_variables, data.result.scopes[scope_id]), scope_id, data.result.scopes[scope_id])
        new_scopes[new_scope_id]['__parent__'] = scope_id

        program = behavior_match.behavior.assign_body_program_scope(new_scope_id)
        L_prime, M_prime, R_prime = program.split_preamble_and_promotable()

        behavior_application_stmt = None
        if self.include_dependency_trace:
            if isinstance(derived_from_stmt, CrowAchieveExpression):
                behavior_application_stmt = CrowBehaviorApplicationExpression(behavior_match.behavior, tuple([VariableExpression(v) for v in behavior_match.behavior.arguments]))
            else:
                behavior_application_stmt, derived_from_stmt = derived_from_stmt, None

        result = data.result.clone(
            scopes=new_scopes, latest_scope=new_scope_id,
            dependency_trace=data.result.dependency_trace + (RegressionTraceStatement(behavior_application_stmt, scope_id, new_scope_id=new_scope_id, scope=new_scopes[scope_id], new_scope=new_scopes[new_scope_id], additional_info='promotable achieve', derived_from=derived_from_stmt), ) if self.include_dependency_trace else tuple()
        )

        L_prime = tuple(ScopedCrowExpression(x, new_scope_id) for x in L_prime) if L_prime is not None else tuple()
        if behavior_match.behavior.always:
            L_prime = (ScopedCrowExpression(CrowBehaviorCommit.execution_only(), new_scope_id), ) + L_prime
        if M_prime is not None:
            if middle_program is not None:
                M_prime = CrowBehaviorOrderingSuite.make_unordered(middle_program, CrowBehaviorOrderingSuite.make_sequential(M_prime, variable_scope_identifier=new_scope_id))
            else:
                M_prime = CrowBehaviorOrderingSuite.make_sequential(M_prime, variable_scope_identifier=new_scope_id)
        else:
            M_prime = middle_program
        R_prime = tuple(ScopedCrowExpression(x, new_scope_id) for x in R_prime)
        if not behavior_match.behavior.effect_body.is_empty():
            R_prime += (ScopedCrowExpression(CrowBehaviorEffectApplicationExpression(behavior_match.behavior), new_scope_id),)

        if isinstance(derived_from_stmt, CrowAchieveExpression) and derived_from_stmt.once == False:
            R_prime += (ScopedCrowExpression(CrowAssertExpression(derived_from_stmt.goal, once=False), scope_id),)

        new_minimize = behavior_match.behavior.minimize if behavior_match.behavior.minimize is not None else data.minimize
        self._push_node(parent_node, PriorityTreeNodeData(result, L_prime, M_prime, R_prime, new_minimize), None)

    def _handle_result_foreach_statement(
        self, parent_node: PriorityTreeNode, child_result: CrowPlanningResult3,
        stmt: CrowBehaviorForeachLoopSuite, scope_id: int
    ):
        variable = stmt.variable
        new_csp = child_result.csp

        if stmt.is_foreach_in_expression:
            objects, new_csp = self.evaluate(stmt.loop_in_expression, state=child_result.state, csp=child_result.csp, bounded_variables=child_result.scopes[scope_id], clone_csp=True, state_index=child_result.get_state_index())
            if isinstance(objects, ListValue):
                objects = objects.values
            elif isinstance(objects, TensorValue) and objects.dtype.is_batched_list_type:
                objects = tuple(objects.iter_batched_indexing())
        else:
            objects = child_result.state.object_type2name[variable.dtype.typename]

        if len(objects) == 0:
            self._push_node_result(parent_node, child_result)
            return

        new_scope_id = child_result.latest_scope
        new_scopes = child_result.scopes.copy()
        new_additional_dependency_traces = list()
        accumulated_programs = list()
        for index in range(len(objects)):
            new_scope_id = new_scope_id + 1
            new_scopes[new_scope_id] = _init_scope_from(new_scopes[scope_id].copy(), scope_id, new_scopes[scope_id], copy_commit=True)

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

            constant_expression = ObjectConstantExpression(new_scopes[new_scope_id][variable.name]) if is_object else ConstantExpression(objects[index])
            if self.include_dependency_trace:
                new_additional_dependency_traces.append(RegressionTraceStatement(
                    CrowRuntimeAssignmentExpression(variable, constant_expression),
                    scope_id, new_scope_id=new_scope_id,
                    scope=new_scopes[scope_id], new_scope=new_scopes[new_scope_id],
                    additional_info=f'loop variable {variable.name}={new_scopes[new_scope_id][variable.name].name}' if is_object else f'loop variable {variable.name}={objects[index]}',
                    derived_from=stmt
                ))

            this_program = [ScopedCrowExpression(x, new_scope_id) for x in stmt.statements]
            accumulated_programs = accumulated_programs + this_program

        new_result = child_result.clone(
            csp=new_csp,
            scopes=new_scopes, latest_scope=new_scope_id,
            dependency_trace=child_result.dependency_trace + tuple(new_additional_dependency_traces) if self.include_dependency_trace else tuple()
        )

        self._push_node(parent_node, PriorityTreeNodeData(new_result, tuple(), None, tuple(accumulated_programs), parent_node.data.minimize), None)

    def _handle_result_condition_statement(
        self, parent_node: PriorityTreeNode, child_result: CrowPlanningResult3,
        stmt: CrowBehaviorConditionSuite, scope_id: int
    ) -> None:
        rv, new_csp = self.evaluate(stmt.condition, state=child_result.state, csp=child_result.csp, bounded_variables=child_result.scopes[scope_id], clone_csp=True, state_index=child_result.get_state_index())
        if self.verbose:
            print('    Condition suite:', stmt.condition, '=>', rv)

        if isinstance(rv, OptimisticValue):
            self._push_node(parent_node, PriorityTreeNodeData(child_result.clone(
                csp=new_csp.add_equal_constraint(rv, True),
                dependency_trace=child_result.dependency_trace + (RegressionTraceStatement(CrowAssertExpression(stmt.condition), scope_id, scope=child_result.scopes[scope_id], additional_info='condition=True branch', derived_from=stmt), ) if self.include_dependency_trace else tuple()
            ), tuple(), None, tuple(ScopedCrowExpression(x, scope_id) for x in stmt.statements), parent_node.data.minimize), None)
            if stmt.else_statements is not None:
                self._push_node(parent_node, PriorityTreeNodeData(child_result.clone(
                    csp=new_csp.add_equal_constraint(rv, False),
                    dependency_trace=child_result.dependency_trace + (RegressionTraceStatement(CrowAssertExpression(NotExpression(stmt.condition)), scope_id, scope=child_result.scopes[scope_id], additional_info='condition=False branch', derived_from=stmt), ) if self.include_dependency_trace else tuple()
                ), tuple(), None, tuple(ScopedCrowExpression(x, scope_id) for x in stmt.else_statements), parent_node.data.minimize), None)
            else:
                result = child_result.clone(csp=new_csp.add_equal_constraint(rv, False))
                if self.include_dependency_trace:
                    result = child_result.clone(dependency_trace=child_result.dependency_trace + (RegressionTraceStatement(CrowAssertExpression(NotExpression(stmt.condition)), scope_id, scope=child_result.scopes[scope_id], additional_info='condition=False branch', derived_from=stmt), ))
                self._push_node_result(parent_node, result)
        else:
            if bool(rv):
                result = child_result
                if self.include_dependency_trace:
                    result = child_result.clone(dependency_trace=child_result.dependency_trace + (RegressionTraceStatement(CrowAssertExpression(stmt.condition), scope_id, scope=child_result.scopes[scope_id], additional_info='condition=True branch', derived_from=stmt), ))
                self._push_node(parent_node, PriorityTreeNodeData(result, tuple(), None, tuple(ScopedCrowExpression(x, scope_id) for x in stmt.statements), parent_node.data.minimize), None)
            else:
                result = child_result
                if stmt.else_statements is not None:
                    if self.include_dependency_trace:
                        result = child_result.clone(dependency_trace=child_result.dependency_trace + (RegressionTraceStatement(CrowAssertExpression(NotExpression(stmt.condition)), scope_id, scope=child_result.scopes[scope_id], additional_info='condition=False branch', derived_from=stmt), ))
                    self._push_node(parent_node, PriorityTreeNodeData(result, tuple(), None, tuple(ScopedCrowExpression(x, scope_id) for x in stmt.else_statements), parent_node.data.minimize), None)
                else:
                    result = child_result
                    if self.include_dependency_trace:
                        result = child_result.clone(dependency_trace=child_result.dependency_trace + (RegressionTraceStatement(CrowAssertExpression(NotExpression(stmt.condition)), scope_id, scope=child_result.scopes[scope_id], additional_info='condition=False branch', derived_from=stmt), ))
                    self._push_node_result(parent_node, result)

    def _handle_primitive_statement(
        self, result: CrowPlanningResult3,
        stmt: Union[CrowBindExpression, CrowAssertExpression, CrowRuntimeAssignmentExpression, CrowControllerApplicationExpression], scope_id: int
    ) -> Iterator[CrowPlanningResult3]:
        if isinstance(stmt, CrowBindExpression):
            if self.verbose:
                print('    Processing bind stmt:', stmt)
            if stmt.is_object_bind:
                for i, new_scope in enumerate(execute_object_bind(self.executor, stmt, result.state, result.scopes[scope_id])):
                    new_scopes = result.scopes.copy()
                    new_scopes[scope_id] = new_scope
                    if self.verbose:
                        print('    New scope:', new_scope)
                    yield result.clone(scopes=new_scopes)
            else:
                new_csp = result.csp.clone()
                new_scopes = result.scopes.copy()
                new_scopes[scope_id] = result.scopes[scope_id].copy()
                for var in stmt.variables:
                    new_scopes[scope_id][var.name] = TensorValue.from_optimistic_value(new_csp.new_var(var.dtype, wrap=True))
                if not stmt.goal.is_null_expression:
                    rv, new_csp = self.evaluate(stmt.goal, state=result.state, csp=new_csp, bounded_variables=new_scopes[scope_id], force_tensor_value=True, clone_csp=False, state_index=result.get_state_index())
                    yield result.clone(csp=new_csp.add_equal_constraint(rv, True), scopes=new_scopes)
                else:
                    yield result.clone(csp=new_csp, scopes=new_scopes)
        elif isinstance(stmt, CrowMemQueryExpression):
            new_scopes = result.scopes.copy()
            new_state, new_csp, new_scope = self.mem_query(stmt.query, state=result.state, csp=result.csp, bounded_variables=result.scopes[scope_id], state_index=result.get_state_index())
            new_scopes[scope_id] = new_scope
            yield result.clone(state=new_state, csp=new_csp, scopes=new_scopes)
        elif isinstance(stmt, CrowAssertExpression):
            rv, new_csp = self.evaluate(stmt.bool_expr, state=result.state, csp=result.csp, bounded_variables=result.scopes[scope_id], clone_csp=True, state_index=result.get_state_index())
            if self.verbose:
                print('    Processing assert stmt:', stmt.bool_expr, '=>', rv)
            if isinstance(rv, OptimisticValue):
                yield result.clone(csp=new_csp.add_equal_constraint(rv, True)).clone_with_new_constraint(scope_id, stmt.bool_expr, True, do=not stmt.once)
            elif bool(rv):
                yield result.clone_with_new_constraint(scope_id, stmt.bool_expr, True, do=not stmt.once)
            else:
                pass  # Return nothing.
        elif isinstance(stmt, CrowUntrackExpression):
            yield result.clone_with_removed_constraint(scope_id, stmt.goal)
        elif isinstance(stmt, CrowRuntimeAssignmentExpression):
            rv, new_csp = self.evaluate(stmt.value, state=result.state, csp=result.csp, bounded_variables=result.scopes[scope_id], force_tensor_value=True, clone_csp=True, state_index=result.get_state_index())
            if self.verbose:
                print('    Processing runtime assignment stmt:', stmt.variable, '<-', stmt.value, '. Value:', rv)
            new_scopes = result.scopes.copy()
            new_scopes[scope_id] = result.scopes[scope_id].copy()
            new_scopes[scope_id][stmt.variable.name] = rv
            yield result.clone(csp=new_csp, scopes=new_scopes)
        elif isinstance(stmt, CrowControllerApplicationExpression):
            new_csp = result.csp.clone()
            argument_values = [self.evaluate(x, state=result.state, csp=new_csp, bounded_variables=result.scopes[scope_id], clone_csp=False, force_tensor_value=True, state_index=result.get_state_index())[0] for x in stmt.arguments]
            if self.verbose:
                print('    Processing controller application stmt:', CrowControllerApplier(stmt.controller, argument_values))
            new_csp.increment_state_timestamp()
            controller_applier = CrowControllerApplier(stmt.controller, argument_values)
            controller_applier.set_constraints(
                result.scope_constraints,
                result.scopes,
                scope_id
            )
            result = result.clone(csp=new_csp, controller_actions=result.controller_actions + (controller_applier, ))

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
                print('    Processing behavior effect:', effect.behavior)
            elif isinstance(effect, CrowControllerApplicationExpression):
                print('    Processing controller effect:', effect.controller)
            else:
                raise ValueError(f'Invalid effect type: {effect}.')
            print('    Constraints:', result.all_scope_constraints())

        new_csp = result.csp.clone()

        if isinstance(effect, CrowBehaviorEffectApplicationExpression):
            new_data = execute_behavior_effect_body(self.executor, effect.behavior, state=result.state, csp=new_csp, scope=result.scopes[scope_id], state_index=result.get_state_index())
            effect_applier = CrowEffectApplier(effect.behavior.effect_body.statements, result.scopes[scope_id])
        elif isinstance(effect, CrowControllerApplicationExpression):
            scope = {x.name: y for x, y in zip(effect.controller.arguments, argument_values)}
            new_data = execute_behavior_effect_body(self.executor, effect.controller, state=result.state, csp=new_csp, scope=scope, state_index=result.get_state_index())
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
                    print('    Removing constraints for scope', scope_id)
        else:
            new_scopes = result.scopes
            new_scope_constraints = result.scope_constraints
        new_scope_constraint_evaluations = dict()

        effect_applier.set_constraints(
            new_scope_constraints,
            new_scopes,
            scope_id
        )

        if self.verbose:
            print('    Previous Actions', result.controller_actions)
        for c_scope_id, constraints in new_scope_constraints.items():
            if self.verbose:
                print('    Constraints for scope', c_scope_id, ':')
            new_scope_constraint_evaluations[c_scope_id] = list()
            for constraint in constraints:
                c_rv, _ = self.evaluate(constraint, state=new_data, csp=new_csp, bounded_variables=new_scopes[c_scope_id], clone_csp=False, state_index=result.get_state_index())
                if self.verbose:
                    if bool(c_rv) is False:
                        print(jacinle.colored('  Constraint:', 'red'), replace_variable_with_value(constraint, new_scopes[c_scope_id]), '=>', c_rv)
                    else:
                        print(jacinle.colored('  Constraint:', 'green'), replace_variable_with_value(constraint, new_scopes[c_scope_id]), '=>', c_rv)

                if isinstance(c_rv, OptimisticValue):
                    new_csp.add_equal_constraint(c_rv, True)
                    new_scope_constraint_evaluations[c_scope_id].append(True)
                else:
                    if bool(c_rv):
                        new_scope_constraint_evaluations[c_scope_id].append(True)
                    else:
                        if self.verbose:
                            print(jacinle.colored('  Constraint violated:', 'red'), replace_variable_with_value(constraint, new_scopes[c_scope_id]), '=>', c_rv)
                        # This constraint has been violated.
                        return

        new_controller_actions = result.controller_actions
        if self.include_effect_appliers:
            new_controller_actions += (effect_applier,)
            new_csp.increment_state_timestamp()
        yield result.clone(
            state=new_data, csp=new_csp, controller_actions=new_controller_actions,
            scopes=new_scopes, scope_constraints=new_scope_constraints, scope_constraint_evaluations=new_scope_constraint_evaluations
        )

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

    def _solve_csp(self, result: CrowPlanningResult3) -> Iterator[CrowPlanningResult3]:
        if not self.enable_csp:
            yield result
            return

        if result.csp is None or result.csp.empty():
            yield result
            return

        solution = None
        for i in range(self.max_csp_trials):
            solution = dpll_solve(
                self.executor, result.csp, simulation_interface=self.simulation_interface, generator_manager=self.generator_manager,
                actions=result.controller_actions, verbose=self.verbose,
                max_generator_trials=self.max_csp_branching_factor,
                simulation_state=result.state.simulation_state, simulation_state_index=result.state.simulation_state_index
            )
            if solution is not None:
                break

        if solution is not None:
            new_state = csp_ground_state(self.executor, result.state, solution)
            new_actions = csp_ground_action_list(self.executor, result.controller_actions, solution)

            if self.simulation_interface is not None:
                with self.simulation_interface.restore_context():
                    for grounded_action in new_actions:
                        succ = self.simulation_interface.step_without_error(grounded_action)
                        if not succ:
                            if self.verbose:
                                jacinle.log_function.print(jacinle.colored(f'Action {grounded_action} failed.', 'red'))
                            raise ValueError(f'Action {grounded_action} failed.')
                    new_state.set_simulation_state(
                        state=self.simulation_interface.save_state(),
                        state_index=len(new_actions)
                    )

            # We have cleared the CSP.
            # TODO(Jiayuan Mao @ 2024/06/8): We also need to update the scope variables --- they can now be directly assigned. The same applies to the state.
            # For now this is okay, because we are only solving the CSP problem at the very end.

            yield result.clone(
                csp=ConstraintSatisfactionProblem(state_timestamp=len(new_actions)),
                state=new_state, controller_actions=new_actions
            )


def _resolve_bounded_variables(bounded_variables: Dict[str, Any], scope: Dict[str, Any]) -> Dict[str, Any]:
    bounded_variables = bounded_variables.copy()
    for name, value in bounded_variables.items():
        if isinstance(value, Variable):
            if value.name not in scope:
                raise ValueError(f'Variable {value.name} not found in the scope.')
            bounded_variables[name] = scope[value.name]
    return bounded_variables


def _init_scope_from(scope: Dict[str, Any], parent_scope_id: int, parent_scope: Dict[str, Any], copy_commit: bool = False, overwrite_commit: Optional[bool] = None) -> Dict[str, Any]:
    scope['__parent__'] = TensorValue.from_scalar(parent_scope_id)
    if overwrite_commit is not None:
        scope['__commit_execution__'] = TensorValue.from_scalar(overwrite_commit)
    else:
        scope['__commit_execution__'] = TensorValue.from_scalar(parent_scope['__commit_execution__'] if copy_commit else False)
    return scope

