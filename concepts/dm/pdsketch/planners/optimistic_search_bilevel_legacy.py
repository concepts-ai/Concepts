#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : optimistic_search_bilevel_legacy.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/10/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Union, Sequence, Iterable, Tuple, List, Set

import jacinle

from concepts.dsl.dsl_types import QINDEX
from concepts.dsl.expression import ValueOutputExpression, is_and_expr
from concepts.dsl.constraint import ConstraintSatisfactionProblem
from concepts.dm.pdsketch.domain import State
from concepts.dm.pdsketch.executor import PDSketchExecutor
from concepts.dm.pdsketch.operator import OperatorApplier, OperatorApplicationExpression
from concepts.dm.pdsketch.regression_rule import AchieveExpression
from concepts.dm.pdsketch.regression_utils import ground_fol_expression, ground_operator_application_expression, surface_fol_downcast
from concepts.dm.pdsketch.crow.crow_state import TotallyOrderedPlan, PartiallyOrderedPlan
from concepts.dm.pdsketch.planners.optimistic_search_bilevel_utils import extract_bounded_variables_from_nonzero, extract_bounded_variables_from_nonzero_dc

__all__ = [
    'enumerate_possible_symbolic_plans_regression_c_v1', 'gen_applicable_regression_rules_v1',
    'enumerate_possible_symbolic_plans_regression_c_v2', 'gen_applicable_regression_rules_v2'
]


def enumerate_possible_symbolic_plans_regression_c_v1(
    executor: PDSketchExecutor, state: State, goal_expr: Union[str, ValueOutputExpression],
    max_actions: int = 10,
) -> Iterable[Tuple[Sequence[OperatorApplier], ConstraintSatisfactionProblem, State, State]]:
    """Enumerate all possible plans that can achieve the goal.

    Args:
        executor: the executor.
        state: the initial state.
        goal_expr: the goal expression.
        max_actions: the maximum number of actions in a plan.

    Returns:
        A list of plans. Each plan is a tuple of (actions, csp, initial_state, final_state).
    """
    if isinstance(goal_expr, str):
        goal_expr = executor.parse(goal_expr)

    @jacinle.log_function(verbose=False)
    def dfs(state, goal_set, maintains, csp, previous_actions, nr_high_level_actions=0):
        jacinle.log_function.print('Current goal_set', goal_set)
        rv = min([executor.execute(goal, state=state, bounded_variables=dict()).item() for goal in goal_set])
        if rv > 0.5:
            return [(state, None, previous_actions)]

        if nr_high_level_actions > max_actions:
            return []

        # jacinle.log_function.print('Current state', state)
        # import ipdb; ipdb.set_trace()

        all_possible_plans = list()
        candidate_regression_rules = gen_applicable_regression_rules_v1(executor, state, goal_set, maintains)
        jacinle.log_function.print('Candidate regression rules', candidate_regression_rules)

        if len(candidate_regression_rules) == 0:
            print('Entering a debug mode: no candidate regression rules.')
            import ipdb; ipdb.set_trace()
            return []

        for goal_index, rule, bounded_variables in candidate_regression_rules:
            grounded_subgoals = list()
            for item in rule.body[:-1]:
                assert isinstance(item, AchieveExpression)
                grounded_subgoals.append(ground_fol_expression(item.goal, bounded_variables))
            assert isinstance(rule.body[-1], OperatorApplicationExpression)

            if len(goal_set) == 1:  # If there is only one single goal left, we should just directly try this regression rule.
                max_prefix_length = 0
            else:
                # TODO(Jiayuan Mao @ 2023/09/06): find the last [always] achievement statement.
                max_prefix_length = len(rule.body) - 1

            for prefix_length in range(0, max_prefix_length + 1):  # assuming the last item is OperatorApplicationExpression
                jacinle.log_function.print('Applying rule', rule, 'for', goal_set[goal_index], 'and prefix length', prefix_length)
                other_goals = goal_set[:goal_index] + goal_set[goal_index + 1:] + grounded_subgoals[:prefix_length]

                if len(other_goals) > 0:
                    # jacinle.log_function.print('Other goals', other_goals)
                    possible_branches = dfs(state, other_goals, maintains, csp, previous_actions, nr_high_level_actions=nr_high_level_actions + 1)
                else:
                    possible_branches = [(state, csp, previous_actions)]

                if len(possible_branches) == 0:
                    jacinle.log_function.print('Prefix planning failed!!! Stop.')
                    # If it's not possible to achieve the subset of goals, then it's not possible to achieve the whole goal set.
                    # Therefore, this is a break, not a continue.
                    break

                for i in range(prefix_length, len(rule.body)):
                    item = rule.body[i]
                    next_possible_branches = list()
                    jacinle.log_function.print(f'Now enter item #{i}: {item}')
                    for cur_state, cur_csp, actions in possible_branches:
                        if isinstance(item, AchieveExpression):
                            subgoal = grounded_subgoals[i]
                            # jacinle.log_function.print('Achieve subgoal', subgoal)
                            next_possible_branches.extend(dfs(cur_state, [subgoal], maintains, None, actions, nr_high_level_actions=nr_high_level_actions + 1))
                        elif isinstance(item, OperatorApplicationExpression):
                            cur_action = ground_operator_application_expression(item, bounded_variables)
                            succ, cur_state = executor.apply(cur_action, cur_state, csp=None, clone=True)
                            jacinle.log_function.print('Applying action', cur_action, 'with result', succ)
                            if succ:
                                next_possible_branches.append((cur_state, None, actions + [cur_action]))
                        else:
                            raise TypeError()
                    possible_branches = next_possible_branches

                # all_possible_plans.extend(possible_branches)

                found_plan = False
                for plan in possible_branches:
                    rv = min([executor.execute(goal, state=plan[0], bounded_variables=dict()).item() for goal in goal_set])
                    if rv > 0.5:
                        jacinle.log_function.print('Found plan', plan[2])
                        found_plan = True
                        all_possible_plans.append(plan)
                    else:
                        jacinle.log_function.print('Plan', plan[2], 'failed')
                if found_plan:
                    break

            # TODO(Jiayuan Mao @ 2023/09/06): think about what's the semantics for this early return.
            # The idea is that, as soon as we found a plan using one of the regression rules, we return it, and don't worry about the other rules.
            # Therefore, it is not guaranteed that the returned plan is the shortest (optimal) one.
            if len(all_possible_plans) > 0:
                break

        if len(all_possible_plans) == 0:
            jacinle.log_function.print('No possible plans for goal set', goal_set)
            return []

        shortest_plan = min(all_possible_plans, key=lambda x: len(x[2]))
        return [shortest_plan]

    if is_and_expr(goal_expr):
        goal_set = list(goal_expr.arguments)
    else:
        goal_set = [goal_expr]

    candidate_plans = dfs(state, goal_set, None, None, list())
    candidate_plans = [(actions, csp, state, final_state) for final_state, csp, actions in candidate_plans]
    return candidate_plans


def gen_applicable_regression_rules_v1(executor: PDSketchExecutor, state: State, goal_set: List[ValueOutputExpression], maintains: Set[ValueOutputExpression]):
    """Generate applicable regression rules for the given goal set and maintains set."""

    # TODO(Jiayuan Mao @ 2023/09/04): implement CSP.
    # TODO(Jiayuan Mao @ 2023/09/04): implement "maintains".

    candidate_regression_rules = list()
    for i, goal in enumerate(goal_set):
        for regression_rule in executor.domain.regression_rules.values():
            goal_expr = regression_rule.goal_expression
            if (variable_binding := surface_fol_downcast(goal_expr, goal)) is None:
                continue
            bounded_variables = dict()
            for v in regression_rule.goal_arguments:
                bounded_variables[v] = variable_binding[v.name].name

            if len(regression_rule.binding_arguments) > 0:
                for v in regression_rule.binding_arguments:
                    bounded_variables[v] = QINDEX

            if len(regression_rule.preconditions_conjunction.arguments) > 0:
                rv = executor.execute(regression_rule.preconditions_conjunction, state=state, bounded_variables=bounded_variables)
            else:
                rv = None

            if rv is None:
                candidate_regression_rules.append((i, regression_rule, bounded_variables))
            else:
                all_forall, candidate_bounded_variables = extract_bounded_variables_from_nonzero(state, rv, regression_rule, default_bounded_variables=bounded_variables)
                if all_forall and regression_rule.always:
                    candidate_regression_rules.append((i, regression_rule, candidate_bounded_variables[0]))
                else:
                    candidate_regression_rules.extend([(i, regression_rule, cbv) for cbv in candidate_bounded_variables])
    return candidate_regression_rules


def enumerate_possible_symbolic_plans_regression_c_v2(
    executor: PDSketchExecutor, state: State, goal_expr: Union[str, ValueOutputExpression],
    max_depth: int = 10, enable_reordering: bool = True,
    verbose: bool = False
) -> Iterable[Tuple[Sequence[OperatorApplier], ConstraintSatisfactionProblem, State, State]]:
    """Enumerate all possible plans that can achieve the goal.

    Args:
        executor: the executor.
        state: the initial state.
        goal_expr: the goal expression.
        max_depth: the maximum depth of goal regression search.
        enable_reordering: whether to enable reordering of the subgoals coming from different chains.
        verbose: whether to print verbose information.

    Returns:
        A list of plans. Each plan is a tuple of (actions, csp, initial_state, final_state).
    """
    if isinstance(goal_expr, str):
        goal_expr = executor.parse(goal_expr)

    search_cache = dict()
    search_stat = {'nr_expanded_nodes': 0}

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
    def dfs(state: State, goal_set: PartiallyOrderedPlan, maintains, csp, previous_actions, nr_high_level_actions=0):
        if verbose:
            jacinle.log_function.print('Current goal_set', goal_set)
            jacinle.log_function.print('Previous actions', previous_actions)

        if (rv := try_retrive_cache(goal_set, previous_actions)) is not None:
            return rv

        if len(goal_set) == 0:
            return return_with_cache(goal_set, previous_actions, [(state, None, previous_actions)])
        rv = min([executor.execute(goal, state=state, bounded_variables=dict()).item() for goal in goal_set.iter_goals()])
        if rv > 0.5:
            return return_with_cache(goal_set, previous_actions, [(state, None, previous_actions)])

        if nr_high_level_actions > max_depth:
            return return_with_cache(goal_set, previous_actions, [])

        search_stat['nr_expanded_nodes'] += 1

        all_possible_plans = list()
        candidate_regression_rules = gen_applicable_regression_rules_v2(executor, state, goal_set, maintains)
        if verbose:
            jacinle.log_function.print('Candidate regression rules', candidate_regression_rules)

        if sum(len(r) for _, _, r in candidate_regression_rules) == 0:
            # print('Entering a debug mode: no candidate regression rules.')
            # import ipdb; ipdb.set_trace()
            return return_with_cache(goal_set, previous_actions, [])

        for chain_index, subgoal_index, this_candidate_regression_rules in candidate_regression_rules:
            other_goals = goal_set.exclude(chain_index, subgoal_index)
            cur_goal = goal_set.chains[chain_index].sequence[subgoal_index]
            if verbose:
                jacinle.log_function.print('Now trying to excluding goal', cur_goal)

            if len(other_goals) == 0:
                other_goals_plans = [(state, None, previous_actions)]
            else:
                other_goals_plans = dfs(state, other_goals, maintains, csp, previous_actions, nr_high_level_actions=nr_high_level_actions)
                for cur_state, cur_csp, cur_actions in other_goals_plans:
                    rv = executor.execute(cur_goal, state=state, bounded_variables=dict()).item()
                    if rv > 0.5:  # Found a plan that directly achieve the goal without any regression.
                        all_possible_plans.append((cur_state, cur_csp, cur_actions))

            if len(all_possible_plans) > 0:
                break

            if len(other_goals) == 0 or not enable_reordering:
                max_prefix_length = 0
            else:
                max_prefix_length = max(len(rule.body) - 1 for rule, _ in this_candidate_regression_rules)
            grounded_subgoals_cache = dict()

            for prefix_length in range(max_prefix_length + 1):
                for regression_rule_index, (rule, bounded_variables) in enumerate(this_candidate_regression_rules):
                    if prefix_length > len(rule.body) - 1:
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
                        possible_branches = other_goals_plans
                    else:
                        cur_other_goals = other_goals.add_chain(grounded_subgoals[:prefix_length])
                        possible_branches = dfs(state, cur_other_goals, maintains, csp, previous_actions, nr_high_level_actions=nr_high_level_actions + 1)

                    if len(possible_branches) == 0:
                        if verbose:
                            jacinle.log_function.print('Prefix planning failed!!! Stop.')
                        # If it's not possible to achieve the subset of goals, then it's not possible to achieve the whole goal set.
                        # Therefore, this is a break, not a continue.
                        break

                    for i in range(prefix_length, len(rule.body)):
                        item = rule.body[i]
                        next_possible_branches = list()
                        if verbose:
                            jacinle.log_function.print(f'Now enter item #{i}: {item}')
                        for cur_state, cur_csp, actions in possible_branches:
                            if isinstance(item, AchieveExpression):
                                subgoal = grounded_subgoals[i]
                                # jacinle.log_function.print('Achieve subgoal', subgoal)
                                next_possible_branches.extend(dfs(cur_state, PartiallyOrderedPlan.from_single_goal(subgoal), maintains, None, actions, nr_high_level_actions=nr_high_level_actions + 1))
                            elif isinstance(item, OperatorApplicationExpression):
                                cur_action = ground_operator_application_expression(item, bounded_variables)
                                succ, cur_state = executor.apply(cur_action, cur_state, csp=None, clone=True)
                                # if verbose:
                                #     jacinle.log_function.print('Applying action', cur_action, 'with result', succ)
                                if succ:
                                    next_possible_branches.append((cur_state, None, actions + [cur_action]))
                            else:
                                raise TypeError()
                        possible_branches = next_possible_branches

                    # all_possible_plans.extend(possible_branches)

                    found_plan = False
                    for plan in possible_branches:
                        rv = min([executor.execute(goal, state=plan[0], bounded_variables=dict()).item() for goal in goal_set.iter_goals()])
                        if rv > 0.5:
                            if verbose:
                                jacinle.log_function.print('Found plan', plan[2], 'for goal', goal_set)
                            found_plan = True
                            all_possible_plans.append(plan)
                        else:
                            if verbose:
                                jacinle.log_function.print('Plan', plan[2], 'failed')
                    if found_plan:
                        # TODO(Jiayuan Mao @ 2023/09/06): since we have changed the order of prefix_length for-loop and the regression rule for-loop.
                        # We need to use an additional dictionary to store whether we have found a plan for a particular regression rule.
                        # Right now this doens't matter because we only use the first plan.
                        break

                # TODO(Jiayuan Mao @ 2023/09/06): think about what's the semantics for this early return.
                # The idea is that, as soon as we found a plan using one of the regression rules, we return it, and don't worry about the other rules.
                # Therefore, it is not guaranteed that the returned plan is the shortest (optimal) one.
                if len(all_possible_plans) > 0:
                    break

            if len(all_possible_plans) > 0:
                break

        if len(all_possible_plans) == 0:
            if verbose:
                jacinle.log_function.print('No possible plans for goal set', goal_set)
            return return_with_cache(goal_set, previous_actions, [])

        shortest_plan = min(all_possible_plans, key=lambda x: len(x[2]))
        return return_with_cache(goal_set, previous_actions, [shortest_plan])

    if is_and_expr(goal_expr):
        goal_set = list(goal_expr.arguments)
    else:
        goal_set = [goal_expr]
    goal_set = PartiallyOrderedPlan((TotallyOrderedPlan(goal_set, is_ordered=True),))

    candidate_plans = dfs(state, goal_set, None, None, list())
    candidate_plans = [(actions, csp, state, final_state) for final_state, csp, actions in candidate_plans]
    return candidate_plans, search_stat


def gen_applicable_regression_rules_v2(executor: PDSketchExecutor, state: State, goals: PartiallyOrderedPlan, maintains: Set[ValueOutputExpression]):
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
                        rv = extract_bounded_variables_from_nonzero_dc(executor, state, regression_rule, bounded_variables, use_optimistic=False)
                    else:
                        rv = executor.execute(regression_rule.preconditions_conjunction, state=state, bounded_variables=bounded_variables)
                        rv = extract_bounded_variables_from_nonzero(state, rv, regression_rule, default_bounded_variables=bounded_variables, use_optimistic=False)
                else:
                    rv = None

                if rv is None:
                    this_chain_candidate_regression_rules = [(regression_rule, bounded_variables)]
                    if regression_rule.always:
                        break
                else:
                    all_forall, candidate_bounded_variables = rv
                    if all_forall and regression_rule.always:
                        this_chain_candidate_regression_rules = [(regression_rule, candidate_bounded_variables[0])]
                        break
                    else:
                        this_chain_candidate_regression_rules.extend([(regression_rule, cbv) for cbv in candidate_bounded_variables])
            candidate_regression_rules.append((chain_index, subgoal_index, this_chain_candidate_regression_rules))
    return candidate_regression_rules
