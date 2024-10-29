#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : optimistic_search_bilevel_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/17/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Utility functions to conduct bi-level search for optimistic planning."""

import collections
import itertools
from dataclasses import dataclass
from typing import Any, Optional, Union, Iterable, Iterator, Sequence, Tuple, List, Set, Dict

import torch
import jacinle

from concepts.dsl.dsl_types import BOOL, Variable, ObjectConstant, UnnamedPlaceholder, QINDEX
from concepts.dsl.expression import ValueOutputExpression, FunctionApplicationExpression, AssignExpression, ConstantExpression, VariableExpression, ObjectConstantExpression, \
    BoolExpression
from concepts.dsl.expression import AndExpression, is_and_expr
from concepts.dsl.constraint import ConstraintSatisfactionProblem, EqualityConstraint, is_optimistic_value
from concepts.dsl.tensor_value import TensorValue
from concepts.dsl.executors.tensor_value_executor import BoundedVariablesDictCompatible
from concepts.dm.pdsketch.executor import PDSketchExecutor
from concepts.dm.pdsketch.domain import State
from concepts.dm.pdsketch.operator import OperatorApplier

from concepts.dsl.constraint import OptimisticValue
from concepts.dm.pdsketch.operator import OperatorApplicationExpression, gen_all_partially_grounded_actions
from concepts.dm.pdsketch.regression_rule import (
    RegressionRule, AchieveExpression, RegressionRuleApplier,
    gen_all_grounded_regression_rules,
    SubgoalSerializability
)
from concepts.dm.pdsketch.regression_utils import ground_fol_expression, ground_operator_application_expression, surface_fol_downcast, evaluate_bool_scalar_expression
from concepts.dm.pdsketch.crow.crow_state import TotallyOrderedPlan, PartiallyOrderedPlan
from concepts.dm.pdsketch.planners.optimistic_search import instantiate_action

__all__ = [
    'OptimisticSearchSymbolicPlan',
    'enumerate_possible_symbolic_plans',
    'enumerate_possible_symbolic_plans_regression_1',
    'enumerate_possible_symbolic_plans_regression_c_v3', 'gen_applicable_regression_rules_v3',
    'extract_bounded_variables_from_nonzero', 'extract_bounded_variables_from_nonzero_dc'
]


@dataclass
class OptimisticSearchSymbolicPlan(object):
    actions: Sequence[OperatorApplier]
    csp: ConstraintSatisfactionProblem
    initial_state: State
    last_state: State


def enumerate_possible_symbolic_plans(
    executor: PDSketchExecutor, state: State, goal_expr: Union[str, ValueOutputExpression],
    max_actions: int = 4, primitive_action_only: bool = False, verbose: bool = False
) -> Iterable[OptimisticSearchSymbolicPlan]:
    csp = ConstraintSatisfactionProblem()
    if isinstance(goal_expr, str):
        goal_expr = executor.parse(goal_expr)
    assert isinstance(goal_expr, ValueOutputExpression)

    initial_state, state = state, state.clone()

    all_actions = gen_all_partially_grounded_actions(executor, state, action_filter=lambda x: x.operator.is_primitive)

    def dfs(state: State, actions: List[OperatorApplier], csp: ConstraintSatisfactionProblem, nr_high_level_actions: int = 0) -> Iterator[OptimisticSearchSymbolicPlan]:
        if nr_high_level_actions >= max_actions:
            return
        for a in all_actions:
            ncsp = csp.clone()
            a = instantiate_action(ncsp, a)
            succ, nstate = executor.apply(a, state, csp=ncsp, clone=True, action_index=len(actions))
            # print(f'Actions {actions} || {a} -> {succ}')
            if succ:
                ncsp_with_goal = ncsp.clone()
                nactions = actions + [a]
                rv = executor.execute(goal_expr, state=nstate, csp=ncsp_with_goal).item()
                if isinstance(rv, OptimisticValue):
                    ncsp_with_goal.add_constraint(EqualityConstraint.from_bool(rv, True), note='goal_test')
                    yield OptimisticSearchSymbolicPlan(nactions, ncsp_with_goal, initial_state, nstate)
                elif float(rv) >= 0.5:
                    yield OptimisticSearchSymbolicPlan(nactions, ncsp_with_goal, initial_state, nstate)

                yield from dfs(nstate, nactions, ncsp, nr_high_level_actions=nr_high_level_actions + 1)

    plans = list(dfs(state, [], csp))
    plans.sort(key=lambda x: len(x.actions))
    return plans


def enumerate_possible_symbolic_plans_regression_1(
    executor: PDSketchExecutor, state: State, goal_expr: Union[str, ValueOutputExpression],
    max_actions: int = 4,
) -> Iterable[OptimisticSearchSymbolicPlan]:
    domain = executor.domain
    if isinstance(goal_expr, str):
        goal_expr = executor.parse(goal_expr)
    assert isinstance(goal_expr, ValueOutputExpression)

    assert isinstance(goal_expr, FunctionApplicationExpression), 'Only support a single function application as goal expression.'
    for regression_rule in domain.regression_rules.values():
        regression_rule: RegressionRule
        for action_item in regression_rule.body:
            if isinstance(action_item, AchieveExpression) and len(action_item.maintains) > 0:
                raise NotImplementedError('enumerate_possible_symbolic_plans_regression_1 does not support maintain expressions.')
        if isinstance(regression_rule.goal_expression, BoolExpression):
            if len(regression_rule.goal_expression.arguments) > 1:
                raise NotImplementedError('enumerate_possible_symbolic_plans_regression_1 does not support multiple goal expressions.')
            goal = regression_rule.goal_expression.arguments[0]
        else:
            raise NotImplementedError('enumerate_possible_symbolic_plans_regression_1 does not support non-boolean goal expressions.')
        if not isinstance(goal.assign_expr, AssignExpression):
            raise NotImplementedError('enumerate_possible_symbolic_plans_regression_1 does not support non-assign goal expressions.')
        if not isinstance(goal.assign_expr.predicate, FunctionApplicationExpression) or goal.assign_expr.predicate.return_type != BOOL:
            raise NotImplementedError('enumerate_possible_symbolic_plans_regression_1 does not support non-boolean goal expressions.')
        if not isinstance(goal.assign_expr.value, ConstantExpression) or bool(goal.assign_expr.value.constant.item()) is not True:
            raise NotImplementedError('enumerate_possible_symbolic_plans_regression_1 does not support negations or non-constant goal expressions.')

    all_actions = gen_all_partially_grounded_actions(executor, state)
    all_regression_rules = gen_all_grounded_regression_rules(executor, state)

    def ground_object_function_application(fapp: FunctionApplicationExpression, variable_mapping: Dict[str, str]):
        arguments = list()
        for arg in fapp.arguments:
            if isinstance(arg, VariableExpression):
                arguments.append(ObjectConstantExpression(ObjectConstant(variable_mapping[arg.variable.name], arg.return_type)))
            elif isinstance(arg, ObjectConstantExpression):
                arguments.append(arg)
            else:
                raise TypeError(f'Unknown argument type: {type(arg)}')
        return FunctionApplicationExpression(fapp.function, arguments)

    def ground_object_operator_application(fapp: OperatorApplicationExpression, variable_mapping: Dict[str, str], csp: ConstraintSatisfactionProblem) -> OperatorApplier:
        arguments = list()
        for arg in fapp.arguments:
            if isinstance(arg, VariableExpression):
                arguments.append(variable_mapping[arg.variable.name])
            elif isinstance(arg, UnnamedPlaceholder):
                arguments.append(TensorValue.from_optimistic_value(csp.new_var(arg.dtype, wrap=True)))
            else:
                raise TypeError(f'Unknown argument type: {type(arg)}')
        return OperatorApplier(fapp.operator, arguments)

    def is_grounded_object_function_application_matching(expr1: FunctionApplicationExpression, expr2: FunctionApplicationExpression):
        if expr1.function != expr2.function:
            return False
        for arg1, arg2 in zip(expr1.arguments, expr2.arguments):
            if isinstance(arg1, ObjectConstantExpression) and isinstance(arg2, ObjectConstantExpression):
                if arg1.constant.name != arg2.constant.name:
                    return False
            else:
                return False
        return True

    def is_rule_applicable(state, regression_rule: RegressionRuleApplier, goal: FunctionApplicationExpression, csp: ConstraintSatisfactionProblem):
        goal_function = regression_rule.regression_rule.arguments[0]
        # TODO(Jiayuan Mao @ 2023/08/17): use "grounding" or "replace feature".
        variable_mapping = {variable.name: value for variable, value in zip(regression_rule.regression_rule.arguments, regression_rule.arguments)}
        grounded_goal_function = ground_object_function_application(goal_function, variable_mapping)
        if is_grounded_object_function_application_matching(goal, grounded_goal_function):
            for precondition in regression_rule.regression_rule.preconditions:
                rv = executor.execute(precondition.bool_expr, state=state, bounded_variables=variable_mapping, csp=csp).item()
                if isinstance(rv, OptimisticValue):
                    csp.add_constraint(EqualityConstraint.from_bool(rv, True), note='precondition')
                elif float(rv) < 0.5:
                    return False
            return True
        return False

    def dfs(state: State, goal: FunctionApplicationExpression, maintains: Set[FunctionApplicationExpression], csp: ConstraintSatisfactionProblem, previous_actions: List[OperatorApplier], nr_high_level_actions: int = 0):
        if nr_high_level_actions >= max_actions:
            return

        all_possible_plans = list()
        for rule in all_regression_rules:
            cur_csp = csp.clone()
            if not is_rule_applicable(state, rule, goal, cur_csp):
                continue

            cur_state = state.clone()
            possible_branches = [(cur_state, cur_csp, previous_actions)]
            variable_mapping = {variable.name: value for variable, value in zip(rule.regression_rule.arguments, rule.arguments)}
            for item in rule.regression_rule.body:
                next_possible_branches = list()
                for cur_state, cur_csp, actions in possible_branches:
                    if isinstance(item, AchieveExpression):
                        cur_csp_2 = cur_csp.clone()
                        rv = executor.execute(item.goal, state=cur_state, bounded_variables=variable_mapping, csp=cur_csp_2).item()
                        subgoal = ground_object_function_application(item.goal, variable_mapping)
                        if isinstance(rv, OptimisticValue):
                            cur_csp_2.add_constraint(EqualityConstraint.from_bool(rv, True), note='subgoal_test')
                            next_possible_branches.append((cur_state, cur_csp_2, actions))

                            next_possible_branches.extend(dfs(cur_state, subgoal, maintains, cur_csp, actions, nr_high_level_actions=nr_high_level_actions))
                        elif float(rv) > 0.5:
                            next_possible_branches.append((cur_state, cur_csp_2, actions))
                        else:
                            next_possible_branches.extend(dfs(cur_state, subgoal, maintains, cur_csp, actions, nr_high_level_actions=nr_high_level_actions + 1))
                    elif isinstance(item, OperatorApplicationExpression):
                        cur_csp_2 = cur_csp.clone()
                        cur_action = ground_object_operator_application(item, variable_mapping, cur_csp_2)
                        succ, cur_state = executor.apply(cur_action, cur_state, csp=cur_csp_2, clone=True, action_index=len(actions))
                        if succ:
                            next_possible_branches.append((cur_state, cur_csp_2, actions + [cur_action]))
                possible_branches = next_possible_branches

            all_possible_plans.extend(possible_branches)

        # TODO(Jiayuan Mao @ 2023/08/18): add a flag to control this behavior.
        if len(all_possible_plans) == 0:
            for action in all_actions:
                cur_csp = csp.clone()
                cur_action = instantiate_action(cur_csp, action)
                succ, cur_state = executor.apply(cur_action, state, csp=cur_csp, clone=True, action_index=len(previous_actions))
                if succ:
                    cur_csp_2 = cur_csp.clone()
                    rv = executor.execute(goal, state=cur_state, csp=cur_csp_2).item()
                    if isinstance(rv, OptimisticValue):
                        cur_csp_2.add_constraint(EqualityConstraint.from_bool(rv, True), note='goal_test')
                        all_possible_plans.append((cur_state, cur_csp_2, previous_actions + [cur_action]))
                        all_possible_plans.extend(dfs(cur_state, goal, maintains, cur_csp, previous_actions + [cur_action], nr_high_level_actions=nr_high_level_actions + 1))
                    elif float(rv) > 0.5:
                        all_possible_plans.append((cur_state, cur_csp_2, previous_actions + [cur_action]))
                    else:
                        all_possible_plans.extend(dfs(cur_state, goal, maintains, cur_csp, previous_actions + [cur_action], nr_high_level_actions=nr_high_level_actions + 1))

        if len(all_possible_plans) == 0:
            return []

        # TODO(Jiayuan Mao @ 2023/08/18): add another flag to control this argmin behavior.
        shortest_plan = min(all_possible_plans, key=lambda x: len(x[2]))
        return [shortest_plan]

    initial_state = state
    csp = ConstraintSatisfactionProblem()
    candidate_plans = dfs(state, goal_expr, set(), csp, list())
    candidate_plans = [OptimisticSearchSymbolicPlan(actions, csp, initial_state, state) for state, csp, actions in candidate_plans]
    candidate_plans.sort(key=lambda x: len(x.actions))
    return candidate_plans


def enumerate_possible_symbolic_plans_regression_c_v3(
    executor: PDSketchExecutor, state: State, goal_expr: Union[str, ValueOutputExpression], *,
    is_goal_serializable: bool = True,
    max_actions: int = 10, enable_csp: bool = False, verbose: bool = False
) -> Tuple[Iterable[OptimisticSearchSymbolicPlan], Dict[str, Any]]:
    """Enumerate all possible plans that can achieve the goal.

    Args:
        executor: the executor.
        state: the initial state.
        goal_expr: the goal expression.
        max_actions: the maximum number of actions in a plan.
        verbose: whether to print verbose information.

    Returns:
        A list of plans. Each plan is a tuple of (actions, csp, initial_state, final_state).
    """
    if isinstance(goal_expr, str):
        goal_expr = executor.parse(goal_expr)

    search_cache = dict()
    search_stat = {'nr_expanded_nodes': 0}

    # NB(Jiayuan Mao @ 2023/09/09): the cache only works for previous_actions == [].
    # That is, we only cache the search results that starts from the initial state.
    def return_with_cache(goal_set, previous_actions, rv):
        if len(previous_actions) == 0:
            goal_str = goal_set.gen_string()
            if goal_str not in search_cache:
                search_cache[goal_str] = rv
        return rv

    def try_retrive_cache(goal_set, previous_actions):
        if len(previous_actions) == 0:
            goal_str = goal_set.gen_string()
            if goal_str in search_cache:
                return search_cache[goal_str]
        return None

    @jacinle.log_function(verbose=False)
    def dfs(
        state: State, goal_set: PartiallyOrderedPlan, maintains: List[ValueOutputExpression],
        csp: Optional[ConstraintSatisfactionProblem], previous_actions: List[OperatorApplier],
        return_all: bool = False,
        nr_high_level_actions: int = 0
    ) -> Iterator[Tuple[State, ConstraintSatisfactionProblem, List[OperatorApplier]]]:
        """Depth-first search for all possible plans.

        Args:
            state: the current state.
            goal_set: the goal set.
            maintains: the list of maintains.
            csp: the current constraint satisfaction problem.
            previous_actions: the previous actions.
            return_all: whether to return all possible plans. If False, only return first plan found.
            nr_high_level_actions: the number of high-level actions.

        Returns:
            a list of plans. Each plan is a tuple of (final_state, csp, actions).
        """

        if verbose:
            jacinle.log_function.print('Current goal_set', goal_set, f'return_all={return_all}')
        #     jacinle.log_function.print('Previous actions', previous_actions)

        if (rv := try_retrive_cache(goal_set, previous_actions)) is not None:
            return rv

        all_possible_plans = list()

        flatten_goals = list(goal_set.iter_goals())
        rv, is_optimistic, new_csp = evaluate_bool_scalar_expression(executor, flatten_goals, state, dict(), csp, csp_note='goal_test')
        if rv:
            all_possible_plans.append((state, new_csp, previous_actions))
            if not is_optimistic:  # If there is no optimistic value, we can stop the search from here.
                # NB(Jiayuan Mao @ 2023/09/11): note that even if `return_all` is True, we still return here.
                # This corresponds to an early stopping behavior that defines the space of all possible plans.
                return return_with_cache(goal_set, previous_actions, all_possible_plans)

        if nr_high_level_actions > max_actions:
            return return_with_cache(goal_set, previous_actions, all_possible_plans)

        search_stat['nr_expanded_nodes'] += 1

        candidate_regression_rules = gen_applicable_regression_rules_v3(executor, state, goal_set, maintains, csp=csp)
        # if verbose:
        #     jacinle.log_function.print('Candidate regression rules', candidate_regression_rules)

        if sum(len(r) for _, _, r in candidate_regression_rules) == 0:
            return return_with_cache(goal_set, previous_actions, all_possible_plans)

        for chain_index, subgoal_index, this_candidate_regression_rules in candidate_regression_rules:
            other_goals = goal_set.exclude(chain_index, subgoal_index)
            cur_goal = goal_set.chains[chain_index].sequence[subgoal_index]
            if verbose:
                jacinle.log_function.print('Now trying to excluding goal', cur_goal)

            if len(other_goals) == 0:
                other_goals_plans = [(state, csp, previous_actions)]
            else:
                # TODO(Jiayuan Mao @ 2023/09/09): change this list to an actual generator call.
                need_return_all = any(rule.max_rule_prefix_length > 0 for rule, _ in this_candidate_regression_rules)
                other_goals_plans = list(dfs(state, other_goals, maintains, csp, previous_actions, nr_high_level_actions=nr_high_level_actions, return_all=need_return_all))
                for cur_state, cur_csp, cur_actions in other_goals_plans:
                    rv, is_optimistic, new_csp = evaluate_bool_scalar_expression(executor, [cur_goal], cur_state, dict(), cur_csp, csp_note='goal_test')
                    if rv:
                        all_possible_plans.append((cur_state, new_csp, cur_actions))
                        if not is_optimistic:
                            # NB(Jiayuan Mao @ 2023/09/11): another place where we stop the search and ignores the `return_all` flag.
                            continue

            if len(this_candidate_regression_rules) == 0:
                continue

            if len(other_goals_plans) == 0:
                continue

            if len(other_goals) == 0:
                max_prefix_length = 0
            else:
                max_prefix_length = max(rule.max_reorder_prefix_length for rule, _ in this_candidate_regression_rules)

            grounded_subgoals_cache = dict()
            prefix_stop_mark = dict()

            for prefix_length in range(max_prefix_length + 1):
                for regression_rule_index, (rule, bounded_variables) in enumerate(this_candidate_regression_rules):
                    if csp is not None:
                        regression_rule_rv, bounded_variables = bounded_variables
                    else:
                        regression_rule_rv = 1
                    if prefix_length > rule.max_reorder_prefix_length:
                        continue
                    if regression_rule_index in prefix_stop_mark and prefix_stop_mark[regression_rule_index]:
                        continue

                    if verbose:
                        jacinle.log_function.print('Applying rule', rule, 'for', cur_goal, 'and prefix length', prefix_length, 'goal set is', goal_set)

                    # construct grounded_subgoals
                    if regression_rule_index in grounded_subgoals_cache:
                        grounded_subgoals = grounded_subgoals_cache[regression_rule_index]
                    else:
                        grounded_subgoals = list()
                        for item in rule.body:
                            if isinstance(item, AchieveExpression):
                                grounded_subgoals.append(ground_fol_expression(item.goal, bounded_variables))
                        grounded_subgoals_cache[regression_rule_index] = grounded_subgoals

                    if prefix_length == 0:
                        if rule.max_rule_prefix_length > 0:
                            possible_branches = other_goals_plans
                        else:
                            possible_branches = [other_goals_plans[0]]
                    else:
                        cur_other_goals = other_goals.add_chain(grounded_subgoals[:prefix_length])
                        possible_branches = list(dfs(state, cur_other_goals, maintains, csp, previous_actions, nr_high_level_actions=nr_high_level_actions + 1, return_all=rule.max_rule_prefix_length > 0))

                    if len(possible_branches) == 0:
                        if verbose:
                            jacinle.log_function.print('Prefix planning failed!!! Stop.')
                        # If it's not possible to achieve the subset of goals, then it's not possible to achieve the whole goal set.
                        # Therefore, this is a break, not a continue.
                        prefix_stop_mark[regression_rule_index] = True
                        continue

                    for i in range(prefix_length, len(rule.body)):
                        item = rule.body[i]
                        next_possible_branches = list()

                        if isinstance(item, AchieveExpression):
                            if item.serializability is SubgoalSerializability.STRONG and len(possible_branches) > 1:
                                possible_branches = [min(possible_branches, key=lambda x: len(x[2]))]
                        need_return_all = i < rule.max_rule_prefix_length

                        for cur_state, cur_csp, actions in possible_branches:
                            if isinstance(item, AchieveExpression):
                                subgoal = grounded_subgoals[i]
                                next_possible_branches.extend(dfs(cur_state, PartiallyOrderedPlan.from_single_goal(subgoal), maintains, cur_csp, actions, nr_high_level_actions=nr_high_level_actions + 1))
                            elif isinstance(item, OperatorApplicationExpression):
                                # TODO(Jiayuan Mao @ 2023/09/11): vectorize this operation, probably only useful when `return_all` is True.
                                new_csp = cur_csp.clone() if cur_csp is not None else None
                                cur_action = ground_operator_application_expression(item, bounded_variables, csp=new_csp)
                                succ, new_state = executor.apply(cur_action, cur_state, csp=new_csp, clone=True, action_index=len(actions))
                                if succ:
                                    next_possible_branches.append((new_state, new_csp, actions + [cur_action]))
                            else:
                                raise TypeError()
                        possible_branches = next_possible_branches

                    # all_possible_plans.extend(possible_branches)

                    found_plan = False
                    # TODO(Jiayuan Mao @ 2023/09/11): implement this via maintains checking.
                    for cur_state, cur_csp, actions in possible_branches:
                        rv, is_optimistic, new_csp = evaluate_bool_scalar_expression(executor, flatten_goals, cur_state, dict(), csp=cur_csp, csp_note='goal_test')
                        if rv:
                            if verbose:
                                jacinle.log_function.print('Found a plan', [str(x) for x in actions], 'for goal set', goal_set)
                            all_possible_plans.append((cur_state, new_csp, actions))
                            found_plan = True

                    if found_plan:
                        prefix_stop_mark[regression_rule_index] = True
                        # TODO(Jiayuan Mao @ 2023/09/06): since we have changed the order of prefix_length for-loop and the regression rule for-loop.
                        # We need to use an additional dictionary to store whether we have found a plan for a particular regression rule.
                        # Right now this doesn't matter because we only use the first plan.

                    if not return_all and len(all_possible_plans) > 0:
                        break

                if not return_all and len(all_possible_plans) > 0:
                    break

            if not return_all and len(all_possible_plans) > 0:
                break

        if len(all_possible_plans) == 0:
            if verbose:
                jacinle.log_function.print('No possible plans for goal set', goal_set)
            return return_with_cache(goal_set, previous_actions, [])

        unique_all_possible_plans = _unique_plans(all_possible_plans)
        if len(unique_all_possible_plans) != len(all_possible_plans):
            if verbose:
                jacinle.log_function.print('Warning: there are duplicate plans for goal set', goal_set, f'({len(unique_all_possible_plans)} unique plans vs {len(all_possible_plans)} total plans)')
                # import ipdb; ipdb.set_trace()

        unique_all_possible_plans = sorted(unique_all_possible_plans, key=lambda x: len(x[2]))
        return return_with_cache(goal_set, previous_actions, unique_all_possible_plans)

    if is_and_expr(goal_expr):
        goal_set = list(goal_expr.arguments)
    else:
        goal_set = [goal_expr]
    goal_set = PartiallyOrderedPlan((TotallyOrderedPlan(goal_set, is_ordered=is_goal_serializable),))

    if enable_csp:
        csp = ConstraintSatisfactionProblem()
    else:
        csp = None

    candidate_plans = dfs(state, goal_set, maintains=None, csp=csp, previous_actions=list(), return_all=enable_csp)
    candidate_plans = [OptimisticSearchSymbolicPlan(actions, csp, state, final_state) for final_state, csp, actions in candidate_plans]
    return candidate_plans, search_stat


def gen_applicable_regression_rules_v3(
    executor: PDSketchExecutor, state: State, goals: PartiallyOrderedPlan,
    maintains: Set[ValueOutputExpression], csp: Optional[ConstraintSatisfactionProblem] = None
) -> List[Tuple[int, int, List[Tuple[RegressionRule, BoundedVariablesDictCompatible]]]]:
    # TODO(Jiayuan Mao @ 2023/09/10): implement maintains.
    candidate_regression_rules = list()
    for chain_index, chain in goals.iter_feasible_chains():
        if chain.is_ordered:
            subgoal_indices = [len(chain) - 1]
        else:
            subgoal_indices = list(range(len(chain)))

        for subgoal_index in subgoal_indices:
            subgoal = chain.sequence[subgoal_index]
            this_chain_candidate_regression_rules = list()

            # For each subgoal in the goal_set, try to find a list of applicable regression rules.
            # If one of the regression rules is always applicable, then we can stop searching.
            for regression_rule in executor.domain.regression_rules.values():
                goal_expr = regression_rule.goal_expression
                if (variable_binding := surface_fol_downcast(goal_expr, subgoal)) is None:
                    continue

                bounded_variables = dict()
                for v in regression_rule.goal_arguments:
                    bounded_variables[v] = variable_binding[v.name].name

                if len(regression_rule.binding_arguments) > 0:
                    for v in regression_rule.binding_arguments:
                        bounded_variables[v] = QINDEX

                if len(regression_rule.preconditions_conjunction.arguments) > 0:
                    if len(regression_rule.binding_arguments) > 4 and len(regression_rule.preconditions_conjunction.arguments) >= 2:
                        rv = extract_bounded_variables_from_nonzero_dc(executor, state, regression_rule, bounded_variables, use_optimistic=csp is not None)
                    else:
                        if csp is not None:
                            new_csp = csp.clone()
                            rv = executor.execute(regression_rule.preconditions_conjunction, state=state, bounded_variables=bounded_variables, csp=new_csp, optimistic_execution=True)
                            rv = extract_bounded_variables_from_nonzero(state, rv, regression_rule, default_bounded_variables=bounded_variables, use_optimistic=True)
                        else:
                            rv = executor.execute(regression_rule.preconditions_conjunction, state=state, bounded_variables=bounded_variables)
                            rv = extract_bounded_variables_from_nonzero(state, rv, regression_rule, default_bounded_variables=bounded_variables, use_optimistic=False)
                else:
                    rv = None

                if rv is None:
                    type_binding_arguments = [state.object_type2name[v.dtype.typename] for v in regression_rule.binding_arguments]
                    all_forall = True
                    for i, arg in enumerate(regression_rule.binding_arguments):
                        if arg.quantifier_flag == 'forall':
                            type_binding_arguments[i] = type_binding_arguments[i][:1] if len(type_binding_arguments[i]) > 0 else []
                        else:
                            all_forall = False
                    for binding_arguments in itertools.product(*type_binding_arguments):
                        cbv = bounded_variables.copy()
                        for variable, value in zip(regression_rule.binding_arguments, binding_arguments):
                            cbv[variable] = value
                        if all_forall and regression_rule.always:
                            this_chain_candidate_regression_rules = [(regression_rule, cbv)] if csp is None else [(regression_rule, (1, cbv))]
                            break
                        this_chain_candidate_regression_rules.append((regression_rule, cbv) if csp is None else (regression_rule, (1, cbv)))
                else:
                    all_forall, candidate_bounded_variables = rv
                    if all_forall and regression_rule.always:
                        this_chain_candidate_regression_rules = [(regression_rule, candidate_bounded_variables[0])]
                        break
                    else:
                        this_chain_candidate_regression_rules.extend([(regression_rule, cbv) for cbv in candidate_bounded_variables])
            candidate_regression_rules.append((chain_index, subgoal_index, this_chain_candidate_regression_rules))
    return candidate_regression_rules


def extract_bounded_variables_from_nonzero(
    state: State, value: TensorValue, regression_rule: RegressionRule,
    default_bounded_variables: Dict[Variable, Union[str, slice]],
    use_optimistic: bool = False
) -> Union[Tuple[bool, List[Dict[Variable, str]]], Tuple[bool, List[Tuple[int, Dict[Variable, str]]]]]:
    """Extract the indices of the given tensor value. The return value is a list of bounded variable dictionaries."""

    if value.tensor_optimistic_values is not None:
        assert use_optimistic, "Optimistic values are not allowed."
        possibly_true_values = torch.where(is_optimistic_value(value.tensor_optimistic_values), value.tensor_optimistic_values, (value.tensor > 0.5))
    else:
        possibly_true_values = (value.tensor > 0.5)
    tensor_variables = list(value.batch_variables)

    # for v in regression_rule.arguments:
    #     if not isinstance(default_bounded_variables[v], ObjectConstant):
    #         if v.name not in value.batch_variables:
    #             possibly_true_values = torch.broadcast_to(
    #                 possibly_true_values.unsqueeze(-1),
    #                 possibly_true_values.shape + (state.get_nr_objects_by_type(v.dtype.typename),)
    #             )
    #             tensor_variables.append(v.name)

    indices = torch.nonzero(possibly_true_values, as_tuple=True)
    if len(indices[0]) == 0:
        return False, []

    variable_name2var = {v.name: v for v in regression_rule.arguments}
    exists_variable_indices = [tensor_variables.index(v.name) for v in regression_rule.binding_arguments if v.quantifier_flag == 'exists']
    all_forall = len(exists_variable_indices) == 0

    if not use_optimistic:
        indices = list(zip(*[i.tolist() for i in indices]))
        if possibly_true_values.ndim == 0:
            value = possibly_true_values.item()
            unique_mapping = {tuple(): tuple()}
        else:
            if not all_forall:
                unique_mapping = dict()
                for index in indices:
                    exists_indices = [index[i] for i in exists_variable_indices]
                    unique_mapping[tuple(exists_indices)] = index
            else:
                unique_mapping = {tuple(): indices[0]}

        candidate_bounded_variables = list()
        for index in unique_mapping.values():
            bounded_variables = default_bounded_variables.copy()
            for i, v in enumerate(index):
                bounded_variables[variable_name2var[tensor_variables[i]]] = state.object_type2name[variable_name2var[tensor_variables[i]].dtype.typename][v]
            if all_forall:
                return True, [bounded_variables]
            candidate_bounded_variables.append(bounded_variables)
        return False, candidate_bounded_variables
    else:
        if possibly_true_values.ndim == 0:
            value = possibly_true_values.item()
            unique_mapping = {tuple(): (tuple(), value)}
        else:
            values = possibly_true_values[indices]
            indices = list(zip(*[i.tolist() for i in indices]))

            unique_mapping = collections.defaultdict(list)
            found_determined_solution = dict()
            for index, value in zip(indices, values.tolist()):
                exists_indices = tuple([index[i] for i in exists_variable_indices])
                if value == 1:
                    if not found_determined_solution.get(exists_indices, False):
                        found_determined_solution[exists_indices] = True
                        unique_mapping[exists_indices] = [(index, value)]
                else:
                    unique_mapping[exists_indices].append((index, value))

        candidate_bounded_variables = list()
        for exists_indices, indices_and_values in unique_mapping.items():
            for index, value in indices_and_values:
                bounded_variables = default_bounded_variables.copy()
                for i, v in enumerate(index):
                    bounded_variables[variable_name2var[tensor_variables[i]]] = state.object_type2name[variable_name2var[tensor_variables[i]].dtype.typename][v]
                if all_forall and value == 1:
                    return True, [(value, bounded_variables)]
                candidate_bounded_variables.append((value, bounded_variables))
        return False, candidate_bounded_variables


def extract_bounded_variables_from_nonzero_dc(
    executor: PDSketchExecutor, state: State, regression_rule: RegressionRule,
    default_bounded_variables: Dict[Variable, Union[str, slice]], use_optimistic: bool = False
) -> Union[Tuple[bool, List[Dict[Variable, str]]], Tuple[bool, List[Tuple[bool, Dict[Variable, str]]], ConstraintSatisfactionProblem]]:
    """Extract bounded variables from nonzero indices in the preconditions of a regression rule.
    The function uses divide-and-conquer to find the bounded variables so it works for regression rules with
    a large number of variabels."""

    if use_optimistic:
        raise NotImplementedError('Optimistic mode is not implemented for extract_bounded_variables_from_nonzero_dc.')

    # NB(Jiayuan Mao @ 2023/09/10): the commented-out code is part of an implementation that performs divide-and-conquer
    # on variables. The current implementation does divide-and-conquer on the precondition list instead.
    # The new version is a bit easier to implement and easier to parallelize.

    # total_binding_variables = len(regression_rule.binding_arguments)

    # group1 = regression_rule.binding_arguments[:total_binding_variables // 2]
    # group2 = regression_rule.binding_arguments[total_binding_variables // 2:]
    # group1_names = {v.name for v in group1}
    # group2_names = {v.name for v in group2}

    # expression_variables = [find_free_variables(e) for e in regression_rule.preconditions_conjunction.arguments]
    # expression_variable_names = [set(v.name for v in vs) for vs in expression_variables]

    # group1_expressions, group2_expressions, cross_expressions = list(), list(), list()
    # for e, variables in zip(regression_rule.preconditions_conjunction.arguments, expression_variable_names):
    #     k1 = len(group1_names.intersection(variables)) > 0
    #     k2 = len(group2_names.intersection(variables)) > 0
    #     if k1 and k2:
    #         cross_expressions.append(e)
    #     elif k1:
    #         group1_expressions.append(e)
    #     elif k2:
    #         group2_expressions.append(e)
    #     else:
    #         group1_expressions.append(e)
    #         group2_expressions.append(e)

    # group1_expression = AndExpression(*group1_expressions) if len(group1_expressions) > 0 else None
    # group2_expression = AndExpression(*group2_expressions) if len(group2_expressions) > 0 else None
    # cross_expression = AndExpression(*cross_expressions) if len(cross_expressions) > 0 else None

    # rv1 = executor.execute(group1_expression, state=state, bounded_variables=bounded_variables) if group1_expression is not None else None
    # rv2 = executor.execute(group2_expression, state=state, bounded_variables=bounded_variables) if group2_expression is not None else None

    # if rv1 is None or rv2 is None:
    #     raise NotImplementedError('not implemented yet.')

    total_terms = len(regression_rule.preconditions_conjunction.arguments)
    group1_expressions = regression_rule.preconditions_conjunction.arguments[:total_terms // 2]
    group2_expressions = regression_rule.preconditions_conjunction.arguments[total_terms // 2:]

    group1_expression = AndExpression(*group1_expressions)
    group2_expression = AndExpression(*group2_expressions)
    rv1 = executor.execute(group1_expression, state=state, bounded_variables=default_bounded_variables)
    rv2 = executor.execute(group2_expression, state=state, bounded_variables=default_bounded_variables)

    shared_variables = list(set(rv1.batch_variables).intersection(rv2.batch_variables))
    shared_var_indices_in_group1 = [rv1.batch_variables.index(v) for v in shared_variables]
    shared_var_indices_in_group1_neg = [i for i in range(len(rv1.batch_variables)) if i not in shared_var_indices_in_group1]
    shared_var_indices_in_group2 = [rv2.batch_variables.index(v) for v in shared_variables]
    shared_var_indices_in_group2_neg = [i for i in range(len(rv2.batch_variables)) if i not in shared_var_indices_in_group2]

    rv1_indices = collections.defaultdict(list)
    for x in rv1.tensor.nonzero().tolist():
        rv1_indices[tuple(x[i] for i in shared_var_indices_in_group1)].append(tuple(x[i] for i in shared_var_indices_in_group1_neg))
    rv2_indices = collections.defaultdict(list)
    for x in rv2.tensor.nonzero().tolist():
        rv2_indices[tuple(x[i] for i in shared_var_indices_in_group2)].append(tuple(x[i] for i in shared_var_indices_in_group2_neg))

    output_variables = shared_variables + [v for v in rv1.batch_variables if v not in shared_variables] + [v for v in rv2.batch_variables if v not in shared_variables]
    variable_name2var = {v.name: v for v in regression_rule.arguments}
    exists_indices = [output_variables.index(v.name) for v in regression_rule.binding_arguments if v.quantifier_flag == 'exists']
    all_forall = len(exists_indices) == 0

    if len(rv1_indices) == 0 or len(rv2_indices) == 0:
        return False, []

    indices = list()
    for k in set(rv1_indices.keys()).intersection(rv2_indices.keys()):
        indices.extend([k + x + y for x in rv1_indices[k] for y in rv2_indices[k]])

    if len(indices) == 0:
        return False, []

    if not all_forall:
        unique_mapping = dict()
        for index in indices:
            exists_indices = [index[i] for i in exists_indices]
            unique_mapping[tuple(exists_indices)] = index
    else:
        unique_mapping = {tuple(): indices[0]}

    candidate_bounded_variables = list()
    for index in unique_mapping.values():
        bounded_variables = default_bounded_variables.copy()
        for i, v in enumerate(index):
            bounded_variables[variable_name2var[output_variables[i]]] = state.object_type2name[variable_name2var[output_variables[i]].dtype.typename][v]
        if all_forall:
            return True, [bounded_variables]
        candidate_bounded_variables.append(bounded_variables)
    return False, candidate_bounded_variables


def _unique_plans(plans):
    plan_string_set = set()
    unique_plans = list()
    for state, csp, actions in plans:
        plan_string = ' '.join(str(x) for x in actions)
        if plan_string not in plan_string_set:
            plan_string_set.add(plan_string)
            unique_plans.append((state, csp, actions))
    return unique_plans

