#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : crow_planner_execution.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/14/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""A temporary implementation of the CROW planner but with an execution interface. The current implementation assumes that all regression rules are annotated with
[[always=true]], and furthermore, it will always execute primitive actions. That is, every single action is appended with a commit flag."""

import warnings
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import jacinle
from concepts.dsl.dsl_types import Variable, ObjectConstant
from concepts.dsl.constraint import ConstraintSatisfactionProblem, OptimisticValue
from concepts.dsl.expression import ListExpansionExpression, ValueOutputExpression, is_and_expr
from concepts.dm.pdsketch.crow.crow_state import PartiallyOrderedPlan, TotallyOrderedPlan
from concepts.dm.pdsketch.regression_utils import evaluate_bool_scalar_expression, gen_applicable_regression_rules, len_candidate_regression_rules, ground_operator_application_expression, ground_regression_application_expression, ground_fol_expression
from concepts.dm.pdsketch.regression_utils import create_find_expression_csp_variable, has_optimistic_constant_expression, map_csp_placeholder_action, map_csp_placeholder_goal, map_csp_variable_mapping, map_csp_variable_state, mark_constraint_group_solver
from concepts.dm.pdsketch.csp_solvers.dpll_sampling import csp_dpll_sampling_solve
from concepts.dm.pdsketch.domain import State
from concepts.dm.pdsketch.executor import PDSketchExecutor, PDSketchSGC
from concepts.dm.pdsketch.operator import OperatorApplicationExpression, OperatorApplier
from concepts.dm.pdsketch.planners.optimistic_search import ground_actions
from concepts.dm.pdsketch.regression_rule import RegressionRule, AchieveExpression, RegressionCommitFlag, BindExpression, RegressionRuleApplicationExpression, RegressionRuleApplier, SubgoalCSPSerializability

__all__ = ['crow_recursive_simple_with_execution']

warnings.warn('This is a temporary implementation of the CROW planner but with an execution interface. The current implementation makes strong assumptions. This is not the final version and is likely to be removed in the future.', FutureWarning)

# TODO: implement state variables with [[execution]] flags.
# TODO: implement an actual macro operator application mechanism.


def make_rule_applier(rule: RegressionRule, bounded_variables: Dict[str, ValueOutputExpression]) -> RegressionRuleApplier:
    """Make a rule applier from a regression rule and a set of bounded variables."""
    canonized_bounded_variables = dict()
    for k, v in bounded_variables.items():
        if isinstance(k, Variable):
            k = k.name
        if isinstance(v, ObjectConstant):
            v = v.name
        canonized_bounded_variables[k] = v
    arguments = [canonized_bounded_variables[x.name] for x in rule.arguments]
    return RegressionRuleApplier(rule, arguments)


def crow_recursive_simple_with_execution(
    executor: PDSketchExecutor, state: State, goal_expr: Union[str, ValueOutputExpression], *,
    is_goal_serializable: bool = True,
    is_goal_refinement_compressible: bool = True,
    enable_reordering: bool = False,
    enable_csp: bool = False,
    enable_greedy_prefix_execution: bool = True,
    max_search_depth: int = 10,
    max_csp_branching_factor: int = 5, max_beam_size: int = 20,
    allow_empty_plan_for_optimistic_goal: bool = False,
    verbose: bool = True,
    verbose_rule_matching: bool = False
) -> Tuple[Iterable[Any], Dict[str, Any]]:
    """Compositional Regression and Optimization Wayfinder.

    Args:
        executor: the executor.
        state: the initial state.
        goal_expr: the goal expression.
        is_goal_serializable: whether the goal is serialized already. Otherwise, it will be treated as a conjunction.
        is_goal_refinement_compressible: whether the goals are refinement compressible.
        enable_reordering: whether to enable reordering of subgoals in regression rules.
        enable_csp: whether to enable CSP solving.
        enable_greedy_prefix_execution: whether to enable the HPN execution strategy. As soon as we have found a sequence of actions (could be a prefix), we will execute them.
        max_search_depth: the maximum number of actions in a plan.
        max_csp_branching_factor: the maximum branching factor of the CSP solver.
        max_beam_size: the maximum beam size for keep tracking of CSP solutions.
        allow_empty_plan_for_optimistic_goal: whether to allow empty plans for optimistic goals.
        verbose: whether to print verbose information.
        verbose_rule_matching: whether to print verbose information during the regression rule matching.

    Returns:
        A list of plans. Each plan is a tuple of (actions, csp, initial_state, final_state).
    """
    if isinstance(goal_expr, str):
        goal_expr = executor.parse(goal_expr)

    search_cache = dict()
    search_stat = {'nr_expanded_nodes': 0}

    # NB(Jiayuan Mao @ 2023/09/09): the cache only works for previous_actions == [].
    # That is, we only cache the search results that start from the initial state.
    def return_with_cache(goal_set, previous_actions, rv):
        if len(previous_actions) == 0:
            goal_str = goal_set.gen_string()
            if goal_str not in search_cache:
                search_cache[goal_str] = rv
        return rv

    def try_retrieve_cache(goal_set, previous_actions):
        if len(previous_actions) == 0:
            goal_str = goal_set.gen_string()
            if goal_str in search_cache:
                return search_cache[goal_str]
        return None

    @jacinle.log_function(verbose=False)
    def dfs(
        s: State, g: PartiallyOrderedPlan, c: Tuple[ValueOutputExpression, ...],
        csp: Optional[ConstraintSatisfactionProblem], previous_actions: List[OperatorApplier],
        return_all_skeletons: bool = False,
        tail_csp_solve: bool = False,
        search_depth: int = 0
    ) -> Iterator[Tuple[State, ConstraintSatisfactionProblem, List[OperatorApplier]]]:
        """Depth-first search for all possible plans.

        Args:
            s: the current state.
            g: the current goal.
            c: the list of constraints to maintain.
            csp: the current constraint satisfaction problem.
            previous_actions: the previous actions.
            return_all_skeletons: whether to return all possible plans. If False, only return the first plan found.
            tail_csp_solve: whether to solve the CSP after the expansion of the current goal.
            search_depth: the current search depth.

        Returns:
            a list of plans. Each plan is a tuple of (final_state, csp, actions).
        """

        # NB(Jiayuan Mao @ 2024/01/14): First hack.
        return_all_skeletons = False

        if verbose:
            jacinle.log_function.print('Current goal', g, f'return_all_skeletons={return_all_skeletons}', f'previous_actions={previous_actions}')

        if (rv := try_retrieve_cache(g, previous_actions)) is not None:
            return rv

        all_possible_plans = list()

        flatten_goals = list(g.iter_goals())
        if not has_optimistic_constant_expression(*flatten_goals) or allow_empty_plan_for_optimistic_goal:
            """If the current goal contains no optimistic constant, we may directly solve the CSP."""
            rv, is_optimistic, new_csp = evaluate_bool_scalar_expression(executor, flatten_goals, s, dict(), csp, csp_note='goal_test')
            if rv:
                all_possible_plans.append((s, new_csp, previous_actions))
                if not is_optimistic:  # If there is no optimistic value, we can stop the search from here.
                    # NB(Jiayuan Mao @ 2023/09/11): note that even if `return_all_skeletons` is True, we still return here.
                    # This corresponds to an early stopping behavior that defines the space of all possible plans.
                    return return_with_cache(g, previous_actions, all_possible_plans)

        if search_depth > max_search_depth:
            return return_with_cache(g, previous_actions, all_possible_plans)

        search_stat['nr_expanded_nodes'] += 1

        candidate_regression_rules = gen_applicable_regression_rules(executor, s, g, c, verbose=verbose_rule_matching)
        if len_candidate_regression_rules(candidate_regression_rules) == 0:
            return return_with_cache(g, previous_actions, all_possible_plans)

        some_rule_success = False  # If return_all_skeletons is False, we will stop the search once some rule application succeeds.

        for chain_index, subgoal_index, this_candidate_regression_rules in candidate_regression_rules:
            cur_goal = g.chains[chain_index].sequence[subgoal_index]
            other_goals = g.exclude(chain_index, subgoal_index)
            other_goals_return_all_skeletons = g.chains[chain_index].get_return_all_skeletons_flag(subgoal_index)

            if verbose:
                jacinle.log_function.print('Now trying to excluding goal', cur_goal)

            grounded_subgoals_cache = dict()
            for regression_rule_index, (rule, bounded_variables) in enumerate(this_candidate_regression_rules):
                grounded_subgoals = list()
                placeholder_csp = ConstraintSatisfactionProblem() if enable_csp else None
                placeholder_bounded_variables = bounded_variables.copy()
                rule_applier = make_rule_applier(rule, bounded_variables)
                for i, item in enumerate(rule.body):
                    if isinstance(item, AchieveExpression):
                        grounded_subgoals.append(AchieveExpression(ground_fol_expression(item.goal, placeholder_bounded_variables), maintains=[], serializability=item.serializability, csp_serializability=item.csp_serializability))
                    elif isinstance(item, BindExpression):
                        for variable in item.variables:
                            placeholder_bounded_variables[variable] = create_find_expression_csp_variable(variable, csp=placeholder_csp, bounded_variables=placeholder_bounded_variables)
                        grounded_subgoals.append(BindExpression([], ground_fol_expression(item.goal, placeholder_bounded_variables), serializability=item.serializability, csp_serializability=item.csp_serializability, ordered=item.ordered))
                    elif isinstance(item, OperatorApplicationExpression):
                        cur_action = ground_operator_application_expression(item, placeholder_bounded_variables, csp=placeholder_csp, rule_applier=rule_applier)
                        grounded_subgoals.append(cur_action)
                    elif isinstance(item, RegressionRuleApplicationExpression):
                        cur_action = ground_regression_application_expression(item, placeholder_bounded_variables, csp=placeholder_csp)
                        grounded_subgoals.append(cur_action)
                    elif isinstance(item, ListExpansionExpression):
                        subgoals = executor.execute(item.expression, s, placeholder_bounded_variables, sgc=PDSketchSGC(s, cur_goal, c))
                        assert isinstance(subgoals, TotallyOrderedPlan), f'ListExpansionExpression must be used with a TotallyOrderedPlan, got {type(subgoals)}'
                        grounded_subgoals.extend(subgoals.sequence)
                    elif isinstance(item, RegressionCommitFlag):
                        grounded_subgoals.append(item)
                    else:
                        raise ValueError(f'Unknown item type {type(item)} in rule {item}.')

                # pass the serializability information to the previous subgoal.
                max_reorder_prefix_length = 0
                for i, item in enumerate(grounded_subgoals):
                    if isinstance(item, RegressionCommitFlag):
                        if i > 0 and isinstance(grounded_subgoals[i - 1], (AchieveExpression, BindExpression)):
                            grounded_subgoals[i - 1].serializability = item.goal_serializability
                    if isinstance(item, (AchieveExpression, BindExpression)):
                        if item.sequential_decomposable is False:
                            max_reorder_prefix_length = i + 1

                grounded_subgoals_cache[regression_rule_index] = (grounded_subgoals, placeholder_csp, max_reorder_prefix_length)

            if len(other_goals) == 0:
                other_goals_plans = [(s, csp, previous_actions)]
            else:
                # TODO(Jiayuan Mao @ 2023/09/09): change this list to an actual generator call.
                other_goals_plans = list()
                other_goals_plans_tmp = list(dfs(s, other_goals, c, csp, previous_actions, search_depth=search_depth, return_all_skeletons=other_goals_return_all_skeletons))
                for cur_state, cur_csp, cur_actions in other_goals_plans_tmp:
                    rv, is_optimistic, new_csp = evaluate_bool_scalar_expression(executor, [cur_goal], cur_state, dict(), cur_csp, csp_note='goal_test_shortcut')
                    if rv:
                        all_possible_plans.append((cur_state, new_csp, cur_actions))
                        if not is_optimistic:
                            # NB(Jiayuan Mao @ 2023/09/11): another place where we stop the search and ignores the `return_all_skeletons` flag.
                            continue
                    other_goals_plans.append((cur_state, cur_csp, cur_actions))

            if len(this_candidate_regression_rules) == 0 or len(other_goals_plans) == 0:
                continue

            if len(other_goals) == 0:
                max_prefix_length = 0
            else:
                max_prefix_length = 0 if not enable_reordering else max([x[2] for x in grounded_subgoals_cache.values()])

            prefix_stop_mark = dict()

            for prefix_length in range(max_prefix_length + 1):
                for regression_rule_index, (rule, bounded_variables) in enumerate(this_candidate_regression_rules):
                    grounded_subgoals, placeholder_csp, max_reorder_prefix_length = grounded_subgoals_cache[regression_rule_index]
                    if prefix_length > max_reorder_prefix_length:
                        continue
                    if regression_rule_index in prefix_stop_mark and prefix_stop_mark[regression_rule_index]:
                        continue

                    if verbose:
                        jacinle.log_function.print('Applying rule', rule, 'for', cur_goal, 'and prefix length', prefix_length, 'goal is', g)

                    if prefix_length == 0:
                        previous_possible_branches = other_goals_plans
                        start_csp_variable_mapping = dict()
                    else:
                        start_csp_variable_mapping = dict()
                        new_csp = cur_csp.clone()
                        new_chain_subgoals = list()
                        new_chain_flags = list()
                        for i, item in enumerate(grounded_subgoals[:prefix_length]):
                            if isinstance(item, AchieveExpression):
                                subgoal, start_csp_variable_mapping = map_csp_placeholder_goal(item.goal, new_csp, placeholder_csp, start_csp_variable_mapping)
                                new_chain_subgoals.append(item.goal)
                                new_chain_flags.append(not item.refinement_compressible or return_all_skeletons)
                            elif isinstance(item, BindExpression):
                                # TODO(Jiayuan Mao @ 2023/12/06): implement this for bypassing FindExpressions that can be commited...
                                subgoal, start_csp_variable_mapping = map_csp_placeholder_goal(item.goal, new_csp, placeholder_csp, start_csp_variable_mapping)
                                with new_csp.with_group(subgoal) as group:
                                    rv = executor.execute(subgoal, s, {}, csp=new_csp).item()
                                    if isinstance(rv, OptimisticValue):
                                        new_csp.add_equal_constraint(rv)
                                mark_constraint_group_solver(executor, s, bounded_variables, group)
                            elif isinstance(item, RegressionCommitFlag):
                                continue
                            else:
                                raise TypeError(f'Unsupported item type {type(item)} in rule {item}.')

                        cur_other_goals = other_goals.add_chain(new_chain_subgoals, new_chain_flags)
                        cur_other_goals_return_all_skeletons = new_chain_flags[-1] if len(new_chain_flags) > 0 else return_all_skeletons
                        previous_possible_branches = list(dfs(s, cur_other_goals, c, new_csp, previous_actions, search_depth=search_depth, return_all_skeletons=cur_other_goals_return_all_skeletons))

                    if len(previous_possible_branches) == 0:
                        if verbose:
                            jacinle.log_function.print('Prefix planning failed!!! Stop.')
                        # If it's not possible to achieve the subset of goals, then it's not possible to achieve the whole goal.
                        # Therefore, this is a break, not a continue.
                        prefix_stop_mark[regression_rule_index] = True
                        continue

                    for prev_state, prev_csp, prev_actions in previous_possible_branches:
                        # construct the new csp and the sequence of grounded subgoals.
                        possible_branches = [(prev_state, prev_csp, prev_actions, start_csp_variable_mapping)]
                        for i in range(prefix_length, len(grounded_subgoals)):
                            item = grounded_subgoals[i]
                            next_possible_branches = list()

                            if isinstance(item, (AchieveExpression, BindExpression)):
                                if not return_all_skeletons and item.refinement_compressible and len(possible_branches) > 1:
                                    # TODO(Jiayuan Mao @ 2023/12/06): implement this for the case of CSP solving --- we may need to keep multiple variable bindings!
                                    possible_branches = [min(possible_branches, key=lambda x: len(x[2]))]

                            prev_next_possible_branches_length = 0
                            for branch_index, (cur_state, cur_csp, cur_actions, cur_csp_variable_mapping) in enumerate(possible_branches):
                                # prev_next_possible_branches_length = len(next_possible_branches)

                                if isinstance(item, AchieveExpression):
                                    new_csp = cur_csp.clone() if cur_csp is not None else None
                                    subgoal, new_csp_variable_mapping = map_csp_placeholder_goal(item.goal, new_csp, placeholder_csp, cur_csp_variable_mapping)
                                    subgoal_return_all_skeletons_flag = not item.refinement_compressible or return_all_skeletons
                                    this_next_possible_branches = ([(*x, new_csp_variable_mapping) for x in dfs(
                                        cur_state, PartiallyOrderedPlan.from_single_goal(subgoal, subgoal_return_all_skeletons_flag), c + item.maintains,
                                        new_csp, cur_actions,
                                        return_all_skeletons=subgoal_return_all_skeletons_flag, search_depth=search_depth + 1
                                    )])
                                elif isinstance(item, RegressionRuleApplier):
                                    # TODO(Jiayuan Mao @ 2024/01/18): fix the instantiation part...?
                                    subgoal_return_all_skeletons_flag = return_all_skeletons
                                    this_next_possible_branches = ([(*x, cur_csp_variable_mapping) for x in dfs(
                                        # TODO(Jiayuan Mao @ 2024/01/18): fix the maintains.
                                        cur_state, PartiallyOrderedPlan.from_single_goal(item, subgoal_return_all_skeletons_flag), c,
                                        cur_csp, cur_actions,
                                        return_all_skeletons=subgoal_return_all_skeletons_flag, search_depth=search_depth + 1
                                    )])
                                elif isinstance(item, BindExpression):
                                    if cur_csp is None:
                                        raise RuntimeError('FindExpression must be used with a CSP.')
                                    new_csp = cur_csp.clone()
                                    subgoal, new_csp_variable_mapping = map_csp_placeholder_goal(item.goal, new_csp, placeholder_csp, cur_csp_variable_mapping)
                                    with new_csp.with_group(subgoal) as group:
                                        rv = executor.execute(subgoal, cur_state, {}, csp=new_csp).item()
                                        if isinstance(rv, OptimisticValue):
                                            new_csp.add_equal_constraint(rv)
                                    mark_constraint_group_solver(executor, state, bounded_variables, group)
                                    this_next_possible_branches = [(cur_state, new_csp, cur_actions, new_csp_variable_mapping)]
                                elif isinstance(item, OperatorApplier):
                                    # TODO(Jiayuan Mao @ 2023/09/11): vectorize this operation, probably only useful when `return_all_skeletons` is True.
                                    new_csp = cur_csp.clone() if cur_csp is not None else None
                                    subaction, new_csp_variable_mapping = map_csp_placeholder_action(item, new_csp, placeholder_csp, cur_csp_variable_mapping)
                                    succ, new_state = executor.apply(subaction, cur_state, csp=new_csp, clone=True, action_index=len(cur_actions))
                                    if succ:
                                        this_next_possible_branches = [(new_state, new_csp, cur_actions + [subaction], new_csp_variable_mapping)]
                                    else:
                                        jacinle.log_function.print('Warning: action', subaction, 'failed.')
                                        executor.apply_precondition_debug(subaction, cur_state, csp=new_csp, logging_mode='log_function')
                                        this_next_possible_branches = []
                                elif isinstance(item, RegressionCommitFlag):
                                    this_next_possible_branches = [(cur_state, cur_csp, cur_actions, cur_csp_variable_mapping)]
                                else:
                                    raise TypeError(f'Unknown item: {item}')

                                # The following code handles the case of HPN execution.
                                if enable_greedy_prefix_execution:
                                    if isinstance(item, OperatorApplier):
                                        if (i == len(grounded_subgoals) - 1 or isinstance(grounded_subgoals[i + 1], (AchieveExpression, BindExpression))):
                                            # If we have found a sequence of actions to execute, we can stop the search.
                                            for new_state, new_csp, new_actions, new_csp_variable_mapping in this_next_possible_branches:
                                                if len(new_actions) > 0:
                                                    all_possible_plans.append((new_state, new_csp, new_actions))

                                            if len(all_possible_plans) > 0 and not return_all_skeletons:
                                                break
                                    else:
                                        for new_state, new_csp, new_actions, new_csp_variable_mapping in this_next_possible_branches:
                                            if len(new_actions) > 0:
                                                all_possible_plans.append((new_state, new_csp, new_actions))

                                        if len(all_possible_plans) > 0 and not return_all_skeletons:
                                            break

                                commit_csp = False
                                if isinstance(item, RegressionCommitFlag) and item.csp_serializability in (SubgoalCSPSerializability.FORALL, SubgoalCSPSerializability.SOME):
                                    commit_csp = True
                                elif isinstance(item, (AchieveExpression, BindExpression)) and item.csp_serializability in (SubgoalCSPSerializability.FORALL, SubgoalCSPSerializability.SOME):
                                    commit_csp = True
                                if commit_csp:
                                    for new_state, new_csp, new_actions, new_csp_variable_mapping in this_next_possible_branches:
                                        assignments = csp_dpll_sampling_solve(executor, cur_csp)
                                        if assignments is not None:
                                            new_state = map_csp_variable_state(cur_state, cur_csp, assignments)
                                            new_csp = ConstraintSatisfactionProblem()
                                            new_actions = ground_actions(executor, cur_actions, assignments)
                                            new_csp_variable_mapping = map_csp_variable_mapping(cur_csp_variable_mapping, csp, assignments)
                                            next_possible_branches.append((new_state, new_csp, new_actions, new_csp_variable_mapping))
                                            # TODO(Jiayuan Mao @ 2023/11/27): okay we need to implement some kind of tracking of "bounded_variables."
                                            # This need to be done by tracking some kind of mapping for optimistic variables in "grounded_subgoals."
                                else:
                                    next_possible_branches.extend(this_next_possible_branches)

                                jacinle.log_function.print(f'Branch {branch_index + 1} of {len(possible_branches)} for {item} has {len(next_possible_branches) - prev_next_possible_branches_length} branches.')
                                prev_next_possible_branches_length = len(next_possible_branches)

                            possible_branches = next_possible_branches

                            jacinle.log_function.print(f'Finished search subgoal {i + 1} of {len(grounded_subgoals)}: {item}. Possible branches (length={len(possible_branches)}):')
                            for x in possible_branches:
                                jacinle.log_function.print(jacinle.indent_text(str(x[2])))

                        if enable_csp:
                            found_plan = False
                            # TODO(Jiayuan Mao @ 2023/09/11): implement this via maintains checking.
                            for cur_state, cur_csp, actions, _ in possible_branches:
                                rv, is_optimistic, new_csp = evaluate_bool_scalar_expression(executor, flatten_goals, cur_state, dict(), csp=cur_csp, csp_note=f'subgoal_test: {"; ".join([str(x) for x in flatten_goals])}')
                                if rv:
                                    if verbose:
                                        jacinle.log_function.print('Found a plan', [str(x) for x in actions], 'for goal', g)

                                    if is_optimistic and tail_csp_solve:
                                        assignments = csp_dpll_sampling_solve(executor, new_csp, verbose=True)
                                        if assignments is not None:
                                            all_possible_plans.append((cur_state, actions, ground_actions(executor, actions, assignments)))
                                            found_plan = True
                                    else:
                                        all_possible_plans.append((cur_state, new_csp, actions))
                                        found_plan = True

                            if found_plan:
                                prefix_stop_mark[regression_rule_index] = True
                                some_rule_success = True
                                # TODO(Jiayuan Mao @ 2023/09/06): since we have changed the order of prefix_length for-loop and the regression rule for-loop.
                                # We need to use an additional dictionary to store whether we have found a plan for a particular regression rule.
                                # Right now this doesn't matter because we only use the first plan.
                        else:
                            all_possible_plans.extend(p[:3] for p in possible_branches)
                            if len(all_possible_plans) > 0 and not return_all_skeletons:
                                some_rule_success = True
                                break

                        if not return_all_skeletons and some_rule_success:
                            break

                    # Break for-loop for `for prev_state in previous_possible_branches`.
                    if not return_all_skeletons and some_rule_success:
                        break

                # Break for-loop for `for rule in regression_rules`
                if not return_all_skeletons and some_rule_success:
                    break

            # Break for-loop for `for prefix_length in range(1, rule.max_rule_prefix_length + 1):`
            if not return_all_skeletons and some_rule_success:
                break

        if len(all_possible_plans) == 0:
            if verbose:
                jacinle.log_function.print('No possible plans for goal', g)
            return return_with_cache(g, previous_actions, [])

        # TODO(Jiayuan Mao @ 2023/11/19): add unique back.
        # unique_all_possible_plans = _unique_plans(all_possible_plans)
        unique_all_possible_plans = all_possible_plans
        if len(unique_all_possible_plans) != len(all_possible_plans):
            if verbose:
                jacinle.log_function.print('Warning: there are duplicate plans for goal', g, f'({len(unique_all_possible_plans)} unique plans vs {len(all_possible_plans)} total plans)')
                # import ipdb; ipdb.set_trace()

        unique_all_possible_plans = sorted(unique_all_possible_plans, key=lambda x: len(x[2]))
        return return_with_cache(g, previous_actions, unique_all_possible_plans)

    if is_and_expr(goal_expr):
        if len(goal_expr.arguments) == 1 and goal_expr.arguments[0].return_type.is_list_type:
            goal_set = [goal_expr]
        else:
            goal_set = list(goal_expr.arguments)
    else:
        goal_set = [goal_expr]

    goal_set = PartiallyOrderedPlan((TotallyOrderedPlan(goal_set, return_all_skeletons_flags=[(not is_goal_refinement_compressible) for _ in goal_set], is_ordered=is_goal_serializable),))

    candidate_plans = dfs(state, goal_set, tuple(), csp=ConstraintSatisfactionProblem() if enable_csp else None, previous_actions=list(), tail_csp_solve=True)
    candidate_plans = [actions for final_state, csp, actions in candidate_plans]
    return candidate_plans, search_stat


