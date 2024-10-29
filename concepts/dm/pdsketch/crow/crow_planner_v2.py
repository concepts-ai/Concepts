#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : crow_planner_v2.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/18/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import itertools
import jacinle

from typing import Any, Optional, Union, Sequence, Tuple, List, Dict, NamedTuple
from concepts.dsl.constraint import OptimisticValue, ConstraintSatisfactionProblem
from concepts.dsl.dsl_types import QINDEX, ObjectConstant
from concepts.dsl.expression import ValueOutputExpression, is_and_expr
from concepts.dsl.executors.tensor_value_executor import BoundedVariablesDictCompatible
from concepts.dm.pdsketch.executor import PDSketchExecutor, GeneratorManager
from concepts.dm.pdsketch.simulator_interface import PDSketchSimulatorInterface
from concepts.dm.pdsketch.domain import State
from concepts.dm.pdsketch.operator import OperatorApplier
from concepts.dm.pdsketch.regression_rule import RegressionRule, RegressionRuleApplier, AchieveExpression, BindExpression, RuntimeAssignExpression, RegressionCommitFlag
from concepts.dm.pdsketch.regression_rule import SubgoalCSPSerializability
from concepts.dm.pdsketch.planners.optimistic_search import ground_actions
from concepts.dm.pdsketch.planners.optimistic_search_with_simulation import csp_dpll_sampling_solve_with_simulation
from concepts.dm.pdsketch.csp_solvers.dpll_sampling import csp_dpll_sampling_solve
from concepts.dm.pdsketch.crow.crow_state import TotallyOrderedPlan, PartiallyOrderedPlan
from concepts.dm.pdsketch.regression_utils import has_optimistic_constant_expression, evaluate_bool_scalar_expression
from concepts.dm.pdsketch.regression_utils import gen_applicable_regression_rules, len_candidate_regression_rules, gen_grounded_subgoals_with_placeholders
from concepts.dm.pdsketch.regression_utils import map_csp_placeholder_goal, mark_constraint_group_solver, map_csp_placeholder_action, map_csp_placeholder_regression_rule_applier, map_csp_variable_state, map_csp_variable_mapping

__all__ = ['CROWPlanTreeNode', 'CROWSearchNode', 'CROWRecursiveSearcherV2', 'crow_recursive_v2']


class CROWPlanTreeNode(object):
    def __init__(self, goal: PartiallyOrderedPlan, always: bool):
        self.goal = goal
        self.always = always
        self.children = []
        self.parent = None

    def add_child(self, child: Union['CROWPlanTreeNode', OperatorApplier, RegressionRuleApplier]):
        self.children.append(child)
        child.parent = self

    def iter_actions(self):
        for child in self.children:
            if isinstance(child, CROWPlanTreeNode):
                yield from child.iter_actions()
            else:
                yield child

    def __str__(self):
        fmt = f'CROWPlanTreeNode({self.goal} always={self.always})\n'
        for child in self.children:
            fmt += f'- {jacinle.indent_text(str(child)).lstrip()}\n'
        return fmt


class CROWSearchNode(object):
    def __init__(
        self, state: State, goal_set: PartiallyOrderedPlan, constraints: Tuple[ValueOutputExpression, ...], csp: Optional[ConstraintSatisfactionProblem],
        plan_tree: Union[CROWPlanTreeNode, list],
        track_all_skeletons: bool,
        associated_regression_rule: Optional[RegressionRule] = None,
        is_top_level: bool = False,
        is_leftmost_branch: bool = False,
        is_all_always: bool = True,
        depth: int = 0,
    ):
        """Initialize search node for the CROW planner.

        Args:
            state: the current when the node is expanded.
            goal_set: the current goal (subgoal) to achieve.
            constraints: the constraints to be satisfied while achieving the goal.
            csp: the constraint satisfaction problem to be solved, associated with all the steps accumulated so far.
            plan_tree: the plan tree associated with the current node.
            track_all_skeletons: whether to track all the skeletons when we are refining the branch.
            associated_regression_rule: the regression rule associated with the current node.
            is_top_level: whether the current node is the top level node (directly instantiated from the goal).
            is_leftmost_branch: whether the current node is the leftmost branch of the search tree.
            is_all_always: whether all the subgoals from the root to the current node are [[always]].
            depth: the depth of the current node.
        """

        self.state = state
        self.goal = goal_set
        self.constraints = constraints
        self.csp = csp
        self.plan_tree = plan_tree

        self.track_all_skeletons = track_all_skeletons
        self.associated_regression_rule = associated_regression_rule
        self.is_top_level = is_top_level
        self.is_leftmost_branch = is_leftmost_branch
        self.is_all_always = is_all_always

        self.depth = depth

    @property
    def previous_actions(self):
        if isinstance(self.plan_tree, CROWPlanTreeNode):
            return list(self.plan_tree.iter_actions())
        return self.plan_tree


class PossibleSearchBranch(NamedTuple):
    state: State
    csp: Optional[ConstraintSatisfactionProblem]
    actions: List[OperatorApplier]
    csp_variable_mapping: Dict[int, Any]
    reg_variable_mapping: Dict[str, Any]


class SearchReturn(NamedTuple):
    state: State
    csp: Optional[ConstraintSatisfactionProblem]
    actions: List[OperatorApplier]


class CROWRecursiveSearcherV2(object):
    def __init__(
        self, executor: PDSketchExecutor, state: State, goal_expr: Union[str, ValueOutputExpression], *,
        enable_reordering: bool = False,
        max_search_depth: int = 10,
        max_beam_size: int = 20,
        # Group 1: goal serialization and refinements.
        is_goal_serializable: bool = True,
        is_goal_refinement_compressible: bool = True,
        # Group 2: CSP solver.
        enable_csp: bool = True,
        max_csp_trials: int = 10,
        max_global_csp_trials: int = 100,
        max_csp_branching_factor: int = 5,
        use_generator_manager: bool = False,
        store_generator_manager_history: bool = False,
        # Group 3: simulation.
        enable_simulation: bool = False,
        simulator: Optional[PDSketchSimulatorInterface] = None,
        # Group 4: dirty derived predicates.
        enable_dirty_derived_predicates: bool = False,
        enable_greedy_execution: bool = False,
        allow_empty_plan_for_optimistic_goal: bool = False,
        verbose: bool = True
    ):
        """Initialize the CROW planner.

        Args:
            executor: the executor used to execute the expressions.
            state: the current state.
            goal_expr: the goal expression to achieve.
            enable_reordering: whether to enable reordering in regression rule applications.
            max_search_depth: the maximum search depth.
            max_beam_size: the maximum beam size when trying different refinements of the same goal.
            is_goal_serializable: whether the goal has been serialized.
            is_goal_refinement_compressible: whether the serialized goals are refinement-compressible (i.e., for each goal item, do we need to track all the skeletons?)
            enable_csp: whether to enable CSP solving.
            max_csp_trials: the maximum number of CSP trials.
            max_global_csp_trials: the maximum number of CSP trials for the global CSP (i.e., the CSP associated with the root node of the search tree.)
            max_csp_branching_factor: the maximum branching factor for CSP.
            use_generator_manager: whether to use the generator manager.
            store_generator_manager_history: whether to store the history of the generator calls in the generator manager.
            enable_simulation: whether to enable simulation in CSP solving.
            simulator: the simulator used for simulation.
            enable_dirty_derived_predicates: whether to enable dirty derived predicates.
            enable_greedy_execution: whether to enable greedy execution.
            allow_empty_plan_for_optimistic_goal: whether to allow empty plan for optimistic goal.
            verbose: whether to print verbose information.
        """
        self.executor = executor
        self.state = state
        self.goal_expr = goal_expr

        if isinstance(self.goal_expr, str):
            self.goal_expr = executor.parse(self.goal_expr)

        self.max_search_depth = max_search_depth
        self.max_beam_size = max_beam_size
        self.enable_reordering = enable_reordering

        self.is_goal_serializable = is_goal_serializable
        self.is_goal_refinement_compressible = is_goal_refinement_compressible

        self.enable_simulation = enable_simulation
        self.simulator = simulator
        if self.enable_simulation and self.simulator is not None:
            # TODO(Jiayuan Mao @ 2024-01-22): after we perform partial grounding for the states, we probably need to update the initial state here.
            self.simulator.set_init_state(self.state)

        self.enable_csp = enable_csp
        self.max_csp_trials = max_csp_trials
        self.max_global_csp_trials = max_global_csp_trials
        self.max_csp_branching_factor = max_csp_branching_factor
        self.use_generator_manager = use_generator_manager
        self.store_generator_manager_history = store_generator_manager_history

        self.enable_dirty_derived_predicates = enable_dirty_derived_predicates
        self.enable_greedy_execution = enable_greedy_execution

        self.allow_empty_plan_for_optimistic_goal = allow_empty_plan_for_optimistic_goal
        self.verbose = verbose

        self._search_cache = dict()
        self._search_stat = {'nr_expanded_nodes': 0}

        if self.use_generator_manager:
            self._generator_manager = GeneratorManager(self.executor, store_history=self.store_generator_manager_history)
        else:
            self._generator_manager = None

    @property
    def search_stat(self) -> Dict[str, Any]:
        return self._search_stat

    @property
    def generator_manager(self) -> Optional[GeneratorManager]:
        return self._generator_manager

    def main(self) -> Tuple[list, dict]:
        if is_and_expr(self.goal_expr):
            if len(self.goal_expr.arguments) == 1 and self.goal_expr.arguments[0].return_type.is_list_type:
                goal_set = [self.goal_expr]
            else:
                goal_set = list(self.goal_expr.arguments)
        else:
            goal_set = [self.goal_expr]

        goal_set = PartiallyOrderedPlan([TotallyOrderedPlan(
            goal_set,
            return_all_skeletons_flags=[(not self.is_goal_refinement_compressible) for _ in goal_set], is_ordered=self.is_goal_serializable
        )])

        candidate_plans = self.dfs(CROWSearchNode(
            self.state, goal_set, tuple(),
            csp=ConstraintSatisfactionProblem() if self.enable_csp else None,
            plan_tree=list(), track_all_skeletons=False,
            is_top_level=True, is_leftmost_branch=True,
            is_all_always=True,
            depth=0
        ))

        candidate_plans = [actions for _, _, actions in candidate_plans]
        return candidate_plans, self._search_stat

    @jacinle.log_function(verbose=False)
    def dfs(self, node: CROWSearchNode) -> Sequence[SearchReturn]:
        """The main entrance of the CROW planner."""
        if self.verbose:
            jacinle.log_function.print('Current goal', node.goal, f'track_all_skeletons={node.track_all_skeletons}', f'previous_actions={node.previous_actions}')
        # jacinle.log_function.print('Current goal', node.goal, f'track_all_skeletons={node.track_all_skeletons}', f'previous_actions={node.previous_actions}')
        if (rv := self._try_retrieve_cache(node)) is not None:
            return rv

        if node.depth >= self.max_search_depth:
            jacinle.log_function.print(jacinle.colored('Warning: search depth exceeded.', color='red'))
            if self.verbose:
                import ipdb; ipdb.set_trace()

        self._search_stat['nr_expanded_nodes'] += 1

        all_possible_plans = list()
        flatten_goals = list(node.goal.iter_goals())
        if not has_optimistic_constant_expression(*flatten_goals) or self.allow_empty_plan_for_optimistic_goal:
            """If the current goal contains no optimistic constant, we may directly solve the CSP."""
            rv, is_optimistic, new_csp = evaluate_bool_scalar_expression(self.executor, flatten_goals, node.state, dict(), node.csp, csp_note='goal_test')
            if rv:
                all_possible_plans.append(SearchReturn(node.state, new_csp, node.previous_actions))
                if not is_optimistic:  # If there is no optimistic value, we can stop the search from here.
                    # note that even if `track_all_skeletons` is True, we still return here.
                    # This corresponds to an early stopping behavior that defines the space of all possible plans.
                    return self._return_with_cache(node, all_possible_plans)

        all_candidate_regression_rules = gen_applicable_regression_rules(self.executor, node.state, node.goal, node.constraints, return_all_candidates=node.track_all_skeletons)
        if len_candidate_regression_rules(all_candidate_regression_rules) == 0:
            return self._return_with_cache(node, all_possible_plans)

        if self.verbose:
            rows = list()
            for chain_index, subgoal_index, candidate_regression_rules in all_candidate_regression_rules:
                cur_goal = node.goal.chains[chain_index].sequence[subgoal_index]
                for rule_item in candidate_regression_rules:
                    rows.append((cur_goal, rule_item[0]))
            jacinle.log_function.print('All candidate regression rules:')
            jacinle.log_function.print(jacinle.tabulate(rows, headers=['Goal', 'Rule'], tablefmt='rst'))

        for chain_index, subgoal_index, candidate_regression_rules in all_candidate_regression_rules:
            cur_goal, other_goals = node.goal.chains[chain_index].sequence[subgoal_index], node.goal.exclude(chain_index, subgoal_index)
            other_goals_track_all_skeletons = node.goal.chains[chain_index].get_return_all_skeletons_flag(subgoal_index)
            candidate_grounded_subgoals = gen_grounded_subgoals_with_placeholders(self.executor, node.state, cur_goal, node.constraints, candidate_regression_rules, enable_csp=self.enable_csp)

            if self.verbose:
                jacinle.log_function.print('Now trying to excluding goal', cur_goal)

            other_goals_plans: List[Tuple[State, Optional[ConstraintSatisfactionProblem], List[OperatorApplier]]]
            if len(other_goals) == 0:
                other_goals_plans = [(node.state, node.csp, node.plan_tree)]
            else:
                other_goals_plans = list()
                other_goals_plans_tmp = self.dfs(CROWSearchNode(
                    node.state, other_goals, node.constraints, node.csp, node.plan_tree,
                    track_all_skeletons=other_goals_track_all_skeletons,
                    is_leftmost_branch=node.is_leftmost_branch,
                    is_all_always=False,
                    depth=node.depth + 1
                ))
                for cur_state, cur_csp, cur_actions in other_goals_plans_tmp:
                    rv, is_optimistic, new_csp = evaluate_bool_scalar_expression(self.executor, [cur_goal], cur_state, dict(), cur_csp, csp_note='goal_test_shortcut')
                    if rv:
                        all_possible_plans.append(SearchReturn(cur_state, new_csp, cur_actions))
                        if not is_optimistic:
                            # another place where we stop the search and ignores the `track_all_skeletons` flag.
                            continue
                    other_goals_plans.append((cur_state, cur_csp, cur_actions))

            if len(other_goals_plans) == 0 or len(candidate_grounded_subgoals) == 0:
                continue

            if len(other_goals) == 0:
                max_prefix_length = 0
            else:
                max_prefix_length = 0 if not self.enable_reordering else max(x[2] for x in candidate_grounded_subgoals.values())

            prefix_stop_mark = dict()
            for prefix_length in range(max_prefix_length + 1):
                for regression_rule_index, (rule, bounded_variables) in enumerate(candidate_regression_rules):
                    grounded_subgoals, placeholder_csp, max_reorder_prefix_length = candidate_grounded_subgoals[regression_rule_index]
                    if prefix_length > max_reorder_prefix_length or (regression_rule_index in prefix_stop_mark and prefix_stop_mark[regression_rule_index]):
                        continue

                    # If the "new" item to be added in the prefix is a Find expression, we should just skip this prefix length.
                    if prefix_length > 0 and isinstance(grounded_subgoals[prefix_length - 1], BindExpression):
                        continue

                    if self.verbose:
                        jacinle.log_function.print('Applying rule', rule, bounded_variables, 'for subgoal', cur_goal, 'under prefix length', prefix_length)
                    # jacinle.log_function.print('Applying rule', rule, bounded_variables, 'for subgoal', cur_goal, 'under prefix length', prefix_length)

                    if prefix_length == 0:
                        start_csp_variable_mapping = dict()
                        previous_possible_branches = ([PossibleSearchBranch(x[0], x[1], x[2], start_csp_variable_mapping, {}) for x in other_goals_plans])
                    else:
                        previous_possible_branches = list()
                        search_goals = self.apply_regression_rule_prefix(node, grounded_subgoals, placeholder_csp, prefix_length, bounded_variables)
                        for new_chain_subgoals, new_chain_flags, new_csp, start_csp_variable_mapping, cur_reg_variable_mapping in search_goals:
                            if len(new_chain_subgoals) == 0:
                                previous_possible_branches = ([PossibleSearchBranch(x[0], x[1], x[2], start_csp_variable_mapping, cur_reg_variable_mapping) for x in other_goals_plans])
                                break

                            cur_other_goals = other_goals.add_chain(new_chain_subgoals, new_chain_flags)
                            cur_other_goals_track_all_skeletons = new_chain_flags[-1] if len(new_chain_flags) > 0 else node.track_all_skeletons
                            previous_possible_branches.extend([PossibleSearchBranch(x[0], x[1], x[2], start_csp_variable_mapping, cur_reg_variable_mapping) for x in self.dfs(CROWSearchNode(
                                node.state, cur_other_goals, node.constraints, new_csp,
                                node.plan_tree, track_all_skeletons=cur_other_goals_track_all_skeletons,
                                is_leftmost_branch=node.is_leftmost_branch,
                                is_all_always=False,
                                depth = node.depth + 1
                            ))])

                    if len(previous_possible_branches) == 0:
                        if self.verbose:
                            jacinle.log_function.print('Prefix planning failed!!! Stop.')
                        # If it's not possible to achieve the subset of goals, then it's not possible to achieve the whole goal.
                        # Therefore, this is a break, not a "continue".
                        prefix_stop_mark[regression_rule_index] = True
                        continue

                    possible_branches = previous_possible_branches

                    for i in range(prefix_length, len(grounded_subgoals)):
                        item = grounded_subgoals[i]
                        next_possible_branches = list()

                        if isinstance(item, (AchieveExpression, BindExpression)):
                            if not node.track_all_skeletons and item.refinement_compressible and len(possible_branches) > 1:
                                # TODO(Jiayuan Mao @ 2023/12/06): implement this for the case of CSP solving --- we may need to keep multiple variable bindings!
                                possible_branches = [min(possible_branches, key=lambda x: len(x.actions))]

                        prev_next_possible_branches_length = 0
                        for branch_index, (cur_state, cur_csp, cur_actions, cur_csp_variable_mapping, cur_reg_variable_mapping) in enumerate(possible_branches):
                            # prev_next_possible_branches_length = len(next_possible_branches)
                            is_leftmost_branch = node.is_leftmost_branch and all(not isinstance(x, (AchieveExpression, RegressionRuleApplier)) for x in grounded_subgoals[:i])

                            if isinstance(item, AchieveExpression):
                                new_csp = cur_csp.clone() if cur_csp is not None else None
                                subgoal, new_csp_variable_mapping = map_csp_placeholder_goal(item.goal, new_csp, placeholder_csp, cur_csp_variable_mapping, cur_reg_variable_mapping)
                                subgoal_track_all_skeletons_flag = not item.refinement_compressible or node.track_all_skeletons
                                this_next_possible_branches = ([PossibleSearchBranch(x[0], x[1], x[2], new_csp_variable_mapping, cur_reg_variable_mapping) for x in self.dfs(CROWSearchNode(
                                    cur_state, PartiallyOrderedPlan.from_single_goal(subgoal, subgoal_track_all_skeletons_flag), node.constraints + item.maintains,
                                    new_csp, cur_actions,
                                    track_all_skeletons=subgoal_track_all_skeletons_flag, is_leftmost_branch=is_leftmost_branch,
                                    is_all_always=node.is_all_always and rule.always,
                                    depth = node.depth + 1
                                ))])
                            elif isinstance(item, RegressionRuleApplier):
                                new_csp = cur_csp.clone() if cur_csp is not None else None
                                subgoal, new_csp_variable_mapping = map_csp_placeholder_regression_rule_applier(item, new_csp, placeholder_csp, cur_csp_variable_mapping, cur_reg_variable_mapping)
                                subgoal_track_all_skeletons_flag = node.track_all_skeletons
                                this_next_possible_branches = ([PossibleSearchBranch(x[0], x[1], x[2], new_csp_variable_mapping, cur_reg_variable_mapping) for x in self.dfs(CROWSearchNode(
                                    cur_state, PartiallyOrderedPlan.from_single_goal(item, subgoal_track_all_skeletons_flag), node.constraints + item.maintains,
                                    cur_csp, cur_actions,
                                    track_all_skeletons=subgoal_track_all_skeletons_flag, is_leftmost_branch=is_leftmost_branch,
                                    is_all_always=node.is_all_always and rule.always,
                                    depth=node.depth + 1
                                ))])
                            elif isinstance(item, OperatorApplier):
                                # TODO(Jiayuan Mao @ 2023/09/11): vectorize this operation, probably only useful when `track_all_skeletons` is True.
                                new_csp = cur_csp.clone() if cur_csp is not None else None
                                action, new_csp_variable_mapping = map_csp_placeholder_action(item, new_csp, placeholder_csp, cur_csp_variable_mapping, cur_reg_variable_mapping)
                                succ, new_state = self.executor.apply(action, cur_state, csp=new_csp, clone=True, action_index=len(cur_actions))
                                if succ:
                                    this_next_possible_branches = [PossibleSearchBranch(new_state, new_csp, cur_actions + [action], new_csp_variable_mapping, cur_reg_variable_mapping)]
                                else:
                                    self.executor.apply_precondition_debug(action, cur_state, csp=new_csp)
                                    if self.verbose:
                                        jacinle.log_function.print('Warning: action', action, 'failed.')
                                    this_next_possible_branches = []
                            elif isinstance(item, BindExpression) and item.is_object_bind_expression:
                                variables = cur_reg_variable_mapping.copy()
                                variables.update({x: QINDEX for x in item.variables})
                                rv = self.executor.execute(item.goal, cur_state, variables, csp=cur_csp)

                                this_next_possible_branches = list()
                                typeonly_indices_variables = list()
                                typeonly_indices_values = list()
                                for v in item.variables:
                                    if v.name not in rv.batch_variables:
                                        typeonly_indices_variables.append(v.name)
                                        typeonly_indices_values.append(range(len(node.state.object_type2name[v.dtype.typename])))
                                for indices in rv.tensor.nonzero():
                                    for typeonly_indices in itertools.product(*typeonly_indices_values):
                                        new_reg_variable_mapping = cur_reg_variable_mapping.copy()
                                        for var in item.variables:
                                            if var.name in rv.batch_variables:
                                                new_reg_variable_mapping[var.name] = ObjectConstant(
                                                    node.state.object_type2name[var.dtype.typename][indices[rv.batch_variables.index(var.name)]],
                                                    var.dtype
                                                )
                                            else:
                                                new_reg_variable_mapping[var.name] = ObjectConstant(
                                                    node.state.object_type2name[var.dtype.typename][typeonly_indices[typeonly_indices_variables.index(var.name)]],
                                                    var.dtype
                                                )
                                        this_next_possible_branches.append(PossibleSearchBranch(cur_state, cur_csp, cur_actions, cur_csp_variable_mapping, new_reg_variable_mapping))
                                        if item.refinement_compressible:
                                            break
                            elif isinstance(item, BindExpression) and not item.is_object_bind_expression:
                                if cur_csp is None:
                                    raise RuntimeError('FindExpression must be used with a CSP.')
                                new_csp = cur_csp.clone()
                                subgoal, new_csp_variable_mapping = map_csp_placeholder_goal(item.goal, new_csp, placeholder_csp, cur_csp_variable_mapping)
                                with new_csp.with_group(subgoal) as group:
                                    rv = self.executor.execute(subgoal, cur_state, cur_reg_variable_mapping, csp=new_csp).item()
                                    if isinstance(rv, OptimisticValue):
                                        new_csp.add_equal_constraint(rv)
                                mark_constraint_group_solver(self.executor, node.state, cur_reg_variable_mapping, group)
                                this_next_possible_branches = [PossibleSearchBranch(cur_state, new_csp, cur_actions, new_csp_variable_mapping, cur_reg_variable_mapping)]
                            elif isinstance(item, RuntimeAssignExpression):
                                new_reg_variable_mapping = cur_reg_variable_mapping.copy()
                                rv = self.executor.execute(item.value, cur_state, new_reg_variable_mapping, csp=cur_csp)
                                new_reg_variable_mapping[item.variable] = rv
                                this_next_possible_branches = [PossibleSearchBranch(cur_state, cur_csp, cur_actions, cur_csp_variable_mapping, new_reg_variable_mapping)]
                            elif isinstance(item, RegressionCommitFlag):
                                this_next_possible_branches = [PossibleSearchBranch(cur_state, cur_csp, cur_actions, cur_csp_variable_mapping, cur_reg_variable_mapping)]
                            else:
                                raise TypeError(f'Unknown item: {item}')

                            # TODO(Jiayuan Mao @ 2024/01/23): this is a hack to handle partial observability, and execution-based constraints.
                            # In general, there should be a more generic way to select if we can directly return.
                            if self.enable_greedy_execution and node.is_all_always:
                                if i == len(grounded_subgoals) - 1 or isinstance(grounded_subgoals[i + 1], (AchieveExpression, BindExpression)):
                                    found_plan = False
                                    for new_state, new_csp, new_actions, new_csp_variable_mapping in this_next_possible_branches:
                                        if len(new_actions) > 0:
                                            if self.verbose:
                                                jacinle.log_function.print(jacinle.colored('Greedy execution for', new_actions, color='green'))
                                            all_possible_plans.append(SearchReturn(new_state, new_csp, new_actions))
                                            found_plan = True
                                            break
                                    if found_plan:
                                        break

                            if self.enable_csp:
                                commit_csp = False
                                if isinstance(item, RegressionCommitFlag) and item.csp_serializability in (SubgoalCSPSerializability.FORALL, SubgoalCSPSerializability.SOME):
                                    commit_csp = True
                                elif isinstance(item, (AchieveExpression, BindExpression)) and item.csp_serializability in (SubgoalCSPSerializability.FORALL, SubgoalCSPSerializability.SOME):
                                    commit_csp = True
                                if commit_csp:
                                    for new_state, new_csp, new_actions, new_csp_variable_mapping in this_next_possible_branches:
                                        assignments = self.solve_csp(new_csp, self.max_csp_trials, actions=new_actions)
                                        if assignments is not None:
                                            new_state = map_csp_variable_state(new_state, new_csp, assignments)
                                            new_csp = ConstraintSatisfactionProblem()
                                            new_actions = ground_actions(self.executor, new_actions, assignments)
                                            new_csp_variable_mapping = map_csp_variable_mapping(new_csp_variable_mapping, node.csp, assignments)
                                            next_possible_branches.append((new_state, new_csp, new_actions, new_csp_variable_mapping))
                                else:
                                    next_possible_branches.extend(this_next_possible_branches)
                            else:
                                next_possible_branches.extend(this_next_possible_branches)

                            if self.verbose:
                                jacinle.log_function.print(f'Branch {branch_index + 1} of {len(possible_branches)} for {item} has {len(next_possible_branches) - prev_next_possible_branches_length} branches.')
                            prev_next_possible_branches_length = len(next_possible_branches)

                        possible_branches = next_possible_branches

                        if self.verbose:
                            jacinle.log_function.print(f'Finished search subgoal {i + 1} of {len(grounded_subgoals)}: {item}. Possible branches (length={len(possible_branches)}):')
                            for x in possible_branches:
                                if len(x[2]) > 0:
                                    action_string = '[' + ', '.join([str(x) for x in x[2]]) + ']'
                                else:
                                    action_string = '[empty]'
                                jacinle.log_function.print(jacinle.indent_text(action_string))

                    # TODO(Jiayuan Mao @ 2024/01/23): this is related to the hack above for the partial observability and execution-based constraints.
                    if not node.track_all_skeletons and len(all_possible_plans) > 1:
                        return self.postprocess_plans(node, all_possible_plans)

                    if self.enable_dirty_derived_predicates:
                        updated_possible_branches = list()
                        for cur_state, cur_csp, cur_actions, _mapping in possible_branches:
                            cur_state = self.apply_regression_rule_effect(cur_state, rule, bounded_variables)
                            updated_possible_branches.append((cur_state, cur_csp, cur_actions, _mapping))
                        possible_branches = updated_possible_branches

                    found_plan = False
                    # TODO(Jiayuan Mao @ 2023/09/11): implement this via maintains checking.
                    for cur_state, cur_csp, cur_actions, _, cur_reg_variable_mapping in possible_branches:
                        rv, is_optimistic, new_csp = evaluate_bool_scalar_expression(self.executor, flatten_goals, cur_state, cur_reg_variable_mapping, csp=cur_csp, csp_note=f'subgoal_test: {"; ".join([str(x) for x in flatten_goals])}')
                        if self.verbose:
                            jacinle.log_function.print(f'Goal test for {[str(x) for x in cur_actions]} (optimistic={is_optimistic}): {rv}')
                        if rv:
                            if self.verbose:
                                jacinle.log_function.print('Found a plan', [str(x) for x in cur_actions], 'for goal', node.goal)

                            if self.enable_csp and is_optimistic and node.is_top_level:
                                assignments = self.solve_csp(new_csp, self.max_global_csp_trials, actions=cur_actions)
                                if assignments is not None:
                                    grounded_actions = ground_actions(self.executor, cur_actions, assignments)
                                    all_possible_plans.append(SearchReturn(cur_state, cur_csp, grounded_actions))
                                    found_plan = True
                                    if self.verbose:
                                        jacinle.log_function.print(jacinle.colored('Global csp solving found a plan', [str(x) for x in grounded_actions], color='green'))
                                else:
                                    if self.verbose:
                                        jacinle.log_function.print(jacinle.colored('Global csp solving failed for the plan', [str(x) for x in cur_actions], color='red'))
                            else:
                                all_possible_plans.append(SearchReturn(cur_state, new_csp, cur_actions))
                                found_plan = True

                    if found_plan:
                        # Since we have changed the order of prefix_length for-loop and the regression rule for-loop,
                        # we need to use an additional dictionary to store whether we have found a plan for a particular regression rule.
                        prefix_stop_mark[regression_rule_index] = True

                    if not node.track_all_skeletons and len(all_possible_plans) > 1:
                        return self.postprocess_plans(node, all_possible_plans)

        return self.postprocess_plans(node, all_possible_plans)


    def apply_regression_rule_prefix(
        self, node: CROWSearchNode, grounded_subgoals, placeholder_csp: ConstraintSatisfactionProblem, prefix_length: int,
        bounded_variables: BoundedVariablesDictCompatible
    ) -> List[Tuple[
        List[ValueOutputExpression], List[bool], Optional[ConstraintSatisfactionProblem], Dict[int, Any], Dict[str, Any]
    ]]:
        """Apply the regression rule for a prefix of the subgoal."""
        start_csp_variable_mapping = dict()
        cur_reg_variable_mapping = dict()
        new_csp = node.csp.clone() if node.csp is not None else None
        new_chain_subgoals = list()
        new_chain_flags = list()
        candidate_rvs = [(new_chain_subgoals, new_chain_flags, new_csp, start_csp_variable_mapping, cur_reg_variable_mapping)]
        for i, item in enumerate(grounded_subgoals[:prefix_length]):
            next_candidate_rvs = list()
            inplace_update = True
            for new_chain_subgoals, new_chain_flags, new_csp, start_csp_variable_mapping, cur_reg_variable_mapping in candidate_rvs:
                if isinstance(item, AchieveExpression):
                    subgoal, start_csp_variable_mapping = map_csp_placeholder_goal(item.goal, new_csp, placeholder_csp, start_csp_variable_mapping, cur_reg_variable_mapping)
                    new_chain_subgoals.append(subgoal)
                    new_chain_flags.append(not item.refinement_compressible or node.track_all_skeletons)
                elif isinstance(item, BindExpression):
                    # TODO(Jiayuan Mao @ 2024/03/08): handle the case of FindExpression with CSP.
                    if item.is_object_bind_expression:
                        variables = {x: QINDEX for x in item.variables}
                        rv = self.executor.execute(item.goal, node.state, variables, csp=new_csp)

                        typeonly_indices_variables = list()
                        typeonly_indices_values = list()
                        for v in item.variables:
                            if v.name not in rv.batch_variables:
                                typeonly_indices_variables.append(v.name)
                                typeonly_indices_values.append(range(len(node.state.object_type2name[v.dtype.typename])))
                        for indices in rv.tensor.nonzero():
                            for typeonly_indices in itertools.product(*typeonly_indices_values):
                                new_reg_variable_mapping = cur_reg_variable_mapping.copy()
                                for var in item.variables:
                                    if var.name in rv.batch_variables:
                                        new_reg_variable_mapping[var.name] = ObjectConstant(
                                            node.state.object_type2name[var.dtype.typename][indices[rv.batch_variables.index(var.name)]],
                                            var.dtype
                                        )
                                    else:
                                        new_reg_variable_mapping[var.name] = ObjectConstant(
                                            node.state.object_type2name[var.dtype.typename][typeonly_indices[typeonly_indices_variables.index(var.name)]],
                                            var.dtype
                                        )
                                next_candidate_rvs.append((new_chain_subgoals, new_chain_flags, new_csp, start_csp_variable_mapping, new_reg_variable_mapping))
                            if item.refinement_compressible:
                                break
                        inplace_update = False
                    else:
                        # TODO(Jiayuan Mao @ 2023/12/06): implement this for bypassing FindExpressions that can be committed...
                        subgoal, start_csp_variable_mapping = map_csp_placeholder_goal(item.goal, new_csp, placeholder_csp, start_csp_variable_mapping, cur_reg_variable_mapping)
                        with new_csp.with_group(subgoal) as group:
                            rv = self.executor.execute(subgoal, node.state, cur_reg_variable_mapping, csp=new_csp).item()
                            if isinstance(rv, OptimisticValue):
                                new_csp.add_equal_constraint(rv)
                        mark_constraint_group_solver(self.executor, node.state, bounded_variables, group)
                elif isinstance(item, RuntimeAssignExpression):
                    rv = self.executor.execute(item.value, node.state, cur_reg_variable_mapping, csp=new_csp)
                    cur_reg_variable_mapping[item.variable] = rv
                elif isinstance(item, RegressionCommitFlag):
                    continue
                else:
                    raise TypeError(f'Unsupported item type {type(item)} in rule {item}.')
            if not inplace_update:
                candidate_rvs = next_candidate_rvs
        return candidate_rvs

    def postprocess_plans(self, node, all_possible_plans):
        if len(all_possible_plans) == 0:
            if self.verbose:
                jacinle.log_function.print('No possible plans for goal', node.goal)
            return self._return_with_cache(node, all_possible_plans)

        # TODO(Jiayuan Mao @ 2023/11/19): add unique back.
        # unique_all_possible_plans = _unique_plans(all_possible_plans)
        unique_all_possible_plans = all_possible_plans
        if len(unique_all_possible_plans) != len(all_possible_plans):
            if self.verbose:
                jacinle.log_function.print('Warning: there are duplicate plans for goal', node.goal, f'({len(unique_all_possible_plans)} unique plans vs {len(all_possible_plans)} total plans)')
                # import ipdb; ipdb.set_trace()

        unique_all_possible_plans = sorted(unique_all_possible_plans, key=lambda x: len(x.actions))
        return self._return_with_cache(node, unique_all_possible_plans)

    def solve_csp(self, csp, max_csp_trials, actions=None):
        for _ in range(max_csp_trials):
            if not self.enable_simulation:
                assignments = csp_dpll_sampling_solve(self.executor, csp, generator_manager=self.generator_manager, max_generator_trials=self.max_csp_branching_factor, verbose=True)
            else:
                assert self.simulator is not None and actions is not None, 'If simulation is enabled, you must provide the simulator, the state, and the actions.'
                # NB(Jiayuan Mao @ 2024-01-22): state is actually not used in the csp_dpll_sampling_solve_with_simulation function. So, we just provide the initial state here.
                assignments = csp_dpll_sampling_solve_with_simulation(self.executor, self.simulator, csp, self.state, actions, generator_manager=self.generator_manager, max_generator_trials=self.max_csp_branching_factor, verbose=True)
            if assignments is not None:
                return assignments
        return None

    def apply_regression_rule_effect(self, state, rule: RegressionRule, bounded_variables: BoundedVariablesDictCompatible):
        return self.executor.apply_effect(rule, state, bounded_variables=bounded_variables, clone=True)

    def _return_with_cache(self, node: CROWSearchNode, rv):
        """The cache only works for previous_actions == []. That is, we only cache the search results that start from the initial state."""
        goal_set, previous_actions = node.goal, node.previous_actions
        if len(previous_actions) == 0:
            goal_str = goal_set.gen_string()
            if goal_str not in self._search_cache:
                self._search_cache[goal_str] = rv
        return rv

    def _try_retrieve_cache(self, node: CROWSearchNode):
        goal_set, previous_actions = node.goal, node.previous_actions
        if len(previous_actions) == 0:
            goal_str = goal_set.gen_string()
            if goal_str in self._search_cache:
                return self._search_cache[goal_str]
        return None


def crow_recursive_v2(
    executor: PDSketchExecutor, state: State, goal_expr: Union[str, ValueOutputExpression], *,
    enable_reordering: bool = False,
    max_search_depth: int = 10,
    max_beam_size: int = 20,
    # Group 1: goal serialization and refinements.
    is_goal_serializable: bool = True,
    is_goal_refinement_compressible: bool = True,
    # Group 2: CSP solver.
    enable_csp: bool = True,
    max_csp_trials: int = 10,
    max_global_csp_trials: int = 100,
    max_csp_branching_factor: int = 5,
    use_generator_manager: bool = False,
    store_generator_manager_history: bool = False,
    # Group 3: simulation.
    enable_simulation: bool = False,
    simulator: Optional[PDSketchSimulatorInterface] = None,
    # Group 4: dirty derived predicates.
    enable_dirty_derived_predicates: bool = False,
    enable_greedy_execution: bool = False,
    allow_empty_plan_for_optimistic_goal: bool = False,
    verbose: bool = True,
):
    kwargs = {
        'enable_reordering': enable_reordering,
        'max_search_depth': max_search_depth,
        'max_beam_size': max_beam_size,
        'is_goal_serializable': is_goal_serializable,
        'is_goal_refinement_compressible': is_goal_refinement_compressible,
        'enable_csp': enable_csp,
        'max_csp_trials': max_csp_trials,
        'max_global_csp_trials': max_global_csp_trials,
        'max_csp_branching_factor': max_csp_branching_factor,
        'use_generator_manager': use_generator_manager,
        'store_generator_manager_history': store_generator_manager_history,
        'enable_simulation': enable_simulation,
        'simulator': simulator,
        'enable_dirty_derived_predicates': enable_dirty_derived_predicates,
        'enable_greedy_execution': enable_greedy_execution,
        'allow_empty_plan_for_optimistic_goal': allow_empty_plan_for_optimistic_goal,
        'verbose': verbose
    }
    return CROWRecursiveSearcherV2(executor, state, goal_expr, **kwargs).main()

