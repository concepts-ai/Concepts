#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : discrete_search.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 11/17/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Discrete-space search PDSketch planners."""

import jacinle
import jactorch
import heapq as hq
from typing import Optional, Any, Union, Tuple, Sequence, List, Dict, Callable
from dataclasses import dataclass

from concepts.dsl.expression import ValueOutputExpression
from concepts.dsl.tensor_value import TensorValue
from concepts.algorithm.search.heuristic_search import QueueNode
from concepts.dm.pdsketch.operator import OperatorApplier, gen_all_grounded_actions
from concepts.dm.pdsketch.domain import State
from concepts.dm.pdsketch.executor import PDSketchExecutor
from concepts.dm.pdsketch.planners.solution_score_tracker import MostPromisingTrajectoryTracker
from concepts.dm.pdsketch.csp_solvers.brute_force_sampling import ContinuousValueDict
from concepts.dm.pdsketch.strips.strips_expression import SState
from concepts.dm.pdsketch.strips.strips_grounding import GStripsTranslatorOptimistic
from concepts.dm.pdsketch.strips.strips_heuristics import StripsHeuristic
from concepts.dm.pdsketch.strips.strips_search import get_priority_func

__all__ = [
    'apply_action', 'goal_test', 'prepare_search',
    'validate_plan', 'brute_force_search'
]



def apply_action(executor: PDSketchExecutor, state: State, action: OperatorApplier, forward_derived=True) -> Tuple[bool, State]:
    """Apply an action to a state. If success, the function also forwards the axioms and optionally derived predicates.

    Args:
        executor: the executor.
        state: the state to be applied.
        action: the action to be applied.
        forward_derived: whether to forward the derived predicates.

    Returns:
        a tuple of (success, new_state).
    """
    succ, ns = executor.apply(action, state)
    # if succ:
    #     ns = executor.forward_predicates_and_axioms(ns, forward_state_variables=False, forward_axioms=True, forward_derived=forward_derived)
    return succ, ns


def goal_test(
    executor: PDSketchExecutor, state: State, goal_expr: ValueOutputExpression,
    trajectory: Sequence[OperatorApplier], mpt_tracker: Optional[MostPromisingTrajectoryTracker] = None, verbose=False
) -> bool:
    """Test whether a state satisfies a goal expression.

    Args:
        executor: the executor.
        state: the state to be tested.
        goal_expr: the goal expression.
        trajectory: the trajectory that leads to the current state.
        mpt_tracker: the tracker for the most promising trajectory.
        verbose: whether to print the verbose information.

    Returns:
        True if the state satisfies the goal expression.
    """
    score = executor.execute(goal_expr, state).item()

    threshold = 0.5
    if mpt_tracker is not None:
        if mpt_tracker.check(score):
            mpt_tracker.update(score, trajectory)
        threshold = mpt_tracker.threshold

    return score > threshold


def prepare_search(
    func_name,
    executor: PDSketchExecutor, state: State, goal_expr: Union[str, ValueOutputExpression], *,
    actions: Optional[Sequence[OperatorApplier]] = None, continuous_values: Optional[ContinuousValueDict] = None, action_filter: Callable[[OperatorApplier], bool] = None,
    use_only_macro_operators: bool = False, allow_macro_operators: bool = False,
    forward_state_variables: bool = True, forward_derived: bool = True,
    verbose=False
) -> Tuple[State, ValueOutputExpression, List[OperatorApplier]]:
    """Prepare for discrete space search.

    Args:
        func_name: the name of the calling function.
        executor: the executor.
        state: the initial state.
        goal_expr: the goal expression.
        actions: the actions to be considered.
        continuous_values: the continuous values for action parameters. If None, all action parameters should be discrete.
        action_filter: the action filter. If None, use all possible actions. It should be a function that takes in an action and returns a boolean.
        use_only_macro_operators: whether to use only macro operators.
        allow_macro_operators: whether to allow macro operators.
        forward_state_variables: whether to forward state variables.
        forward_derived: whether to forward derived variables.
        verbose: whether to print verbose information.

    Returns:
        the initial state, the goal expression, and the list of possible actions.
    """
    # NB(Jiayuan Mao @ 08/05): sanity check.
    # state = executor.forward_predicates_and_axioms(state, forward_state_variables, False, forward_derived)

    if actions is None:
        actions = gen_all_grounded_actions(
            executor, state, continuous_values, action_filter=action_filter,
            use_only_macro_operators=use_only_macro_operators, allow_macro_operators=allow_macro_operators
        )
    goal_expr = executor.parse(goal_expr)

    if verbose:
        # print(func_name+  '::initial_state', state)
        print(func_name + '::actions nr', len(actions))
        print(func_name + '::goal_expr', goal_expr)

    return state, goal_expr, actions


@jactorch.no_grad_func
def validate_plan(
    executor: PDSketchExecutor, state: State, goal_expr: Union[str, ValueOutputExpression], actions: Sequence[OperatorApplier], *,
    forward_state_variables=True, forward_derived=True,
) -> Tuple[State, TensorValue]:
    """Validate a plan by executing it on the given state.

    Args:
        executor: the executor.
        state: the initial state.
        goal_expr: the goal expression.
        actions: the sequence of actions to execute.
        forward_state_variables: whether to forward state variables.
        forward_derived: whether to forward derived predicates.

    Returns:
        the final state and the goal value (the execution result of the goal expression).
    """
    # TODO(Jiayuan Mao @ 11/28): update!.
    # state = executor.forward_predicates_and_axioms(state, forward_state_variables, False, forward_derived)

    if isinstance(goal_expr, str):
        goal_expr = executor.domain.parse(goal_expr)
    else:
        assert isinstance(goal_expr, ValueOutputExpression)

    for action in actions:
        succ, state = apply_action(executor, state, action, forward_derived=forward_derived)
        assert succ, f'Action application failed: {action}.'

    score = executor.execute(goal_expr, state)
    return state, score


@jactorch.no_grad_func
def brute_force_search(
    executor: PDSketchExecutor, state: State, goal_expr: Union[str, ValueOutputExpression], *,
    max_depth: int = 5,
    actions: Optional[Sequence[OperatorApplier]] = None, continuous_values: Optional[ContinuousValueDict] = None, action_filter: Callable[[OperatorApplier], bool] = None,
    use_tuple_desc: bool = True,
    use_only_macro_operators: bool = False,
    allow_macro_operators: bool = False,
    forward_state_variables: bool = True, forward_derived: bool = True,
    verbose: bool = False, return_extra_info: bool = False
) -> Union[
    Optional[Sequence[OperatorApplier]],
    Tuple[Optional[Sequence[OperatorApplier]], Dict[str, Any]]
]:
    """Brute-force search for a plan that satisfies the goal expression.

    Args:
        executor: the executor.
        state: the initial state.
        goal_expr: the goal expression.
        max_depth: the maximum depth of the search.
        actions: the actions to use. If None, use all possible actions.
        continuous_values: the continuous values for action parameters. If None, all action parameters should be discrete.
        action_filter: the action filter. If None, use all possible actions. It should be a function that takes in an action and returns a boolean.
        use_tuple_desc: whether to use tuple description to prune the search space.
        use_only_macro_operators: whether to use only macro operators.
        allow_macro_operators: whether to allow macro operators.
        forward_state_variables: whether to forward state variables before the search starts.
        forward_derived: whether to forward derived predicates after applying actions.
        verbose: whether to print verbose information.
        return_extra_info: whether to return extra information, such as the number of expanded nodes.

    Returns:
        the trajectory if succeeded, otherwise None.
        When `return_extra_info` is True, return a tuple of (trajectory, extra_info), where extra_info is a dictionary.
    """
    state, goal_expr, actions = prepare_search(
        'bfs', executor, state, goal_expr,
        actions=actions, continuous_values=continuous_values, action_filter=action_filter,
        use_only_macro_operators=use_only_macro_operators, allow_macro_operators=allow_macro_operators,
        verbose=verbose, forward_state_variables=forward_state_variables, forward_derived=forward_derived,
    )

    if verbose:
        print('bfs::available actions', len(actions))
        if len(actions) < 100:
            for a in actions:
                print('  action::', a)

    states = [(state, tuple())]
    visited = set()
    if use_tuple_desc:
        visited.add(state.generate_tuple_description(executor.domain))

    nr_expanded_states = 0
    nr_tested_actions = 0

    def wrap_extra_info(trajectory):
        if return_extra_info:
            return trajectory, {'nr_expansions': nr_expanded_states, 'nr_tested_actions': nr_tested_actions}
        return trajectory

    pbar = None
    for depth in range(max_depth):
        if verbose:
            pbar = jacinle.tqdm_pbar(desc=f'bfs::depth={depth}')
        with jacinle.cond_with(pbar, verbose):
            next_states = list()

            for s, traj in states:
                nr_expanded_states += 1
                goal_reached = goal_test(executor, s, goal_expr, traj)
                if goal_reached:
                    if verbose:
                        print('bfs::search succeeded.')
                        print('bfs::total_expansions:', nr_expanded_states)
                    return wrap_extra_info(traj)

                for a in actions:
                    nr_tested_actions += 1
                    succ, ns = apply_action(executor, s, a, forward_derived=forward_derived)
                    if verbose:
                        pbar.update()
                    nt = traj + (a, )
                    if succ:
                        if use_tuple_desc:
                            nst = ns.generate_tuple_description(executor.domain)
                            if nst not in visited:
                                next_states.append((ns, nt))
                                visited.add(nst)
                        else:  # unconditionally expand
                            next_states.append((ns, nt))

            states = next_states

            if verbose:
                pbar.set_description(f'bfs::depth={depth}, states={len(states)}')
                pbar.close()

    # Run a final check on all states at the maximum depth.
    for s, traj in states:
        goal_reached = goal_test(executor, s, goal_expr, traj)
        if goal_reached:
            if verbose:
                print('bfs::search succeeded.')
                print('bfs::total_expansions:', nr_expanded_states)
            return wrap_extra_info(traj)

    if verbose:
        print('bfs::search failed.')
        print('bfs::total_expansions:', nr_expanded_states)

    return wrap_extra_info(None)


@dataclass
class HeuristicSearchState(object):
    """The state for heuristic search."""

    state: State
    """The state."""

    strips_state: SState
    """The STRIPS state."""

    trajectory: Tuple[OperatorApplier, ...]
    """The trajectory."""

    g: float
    """The cost so far."""


@jactorch.no_grad_func
def heuristic_search_strips(
    executor: PDSketchExecutor, state: State, goal_expr: Union[str, ValueOutputExpression],
    strips_heuristic: str = 'hff', *,
    max_expansions: int = 100000, max_depth: int = 100,  # search related parameters.
    heuristic_weight: float = float('inf'),  # heuristic related parameters.
    external_heuristic_function: Callable[[State, ValueOutputExpression], int] = None,  # external heuristic related parameters.
    actions: Optional[Sequence[OperatorApplier]] = None, continuous_values: Optional[ContinuousValueDict] = None, action_filter: Callable[[OperatorApplier], bool] = None,
    strips_forward_relevance_analysis: bool = False, strips_backward_relevance_analysis: bool = True,
    strips_use_sas: bool = False,  # whether to use SAS Strips compiler (AODiscretization)
    use_strips_op: bool = False,
    use_tuple_desc: bool = True, # pruning related parameters.
    forward_state_variables: bool = True, forward_derived: bool = False,  # initialization related parameters.
    track_most_promising_trajectory: bool = False, prob_goal_threshold: float = 0.5,  # non-optimal trajectory tracking related parameters.
    verbose: bool = False, return_extra_info: bool = False
) -> Union[
    Optional[Sequence[OperatorApplier]],
    Tuple[Optional[Sequence[OperatorApplier]], Dict[str, Any]]
]:
    """Perform heuristic search with STRIPS-based heuristics.

    Args:
        executor: the executor.
        state: the initial state.
        goal_expr: the goal expression.
        strips_heuristic: the heuristic to use. Should be a string. Use 'external' to use the external heuristic function.
        max_expansions: the maximum number of expanded nodes.
        max_depth: the maximum depth of the search.
        heuristic_weight: the weight of the heuristic. Use float('inf') to do greedy best-first search.
        external_heuristic_function: the external heuristic function. Should be a function that takes in a state and a goal expression, and returns an integer.
        actions: the actions to use. If None, use all possible actions.
        continuous_values: the continuous values for action parameters. If None, all action parameters should be discrete.
        action_filter: the action filter. If None, use all possible actions. It should be a function that takes in an action and returns a boolean.
        strips_forward_relevance_analysis: whether to perform forward relevance analysis when translating the problem into STRIPS.
        strips_backward_relevance_analysis: whether to perform backward relevance analysis when translating the problem into STRIPS.
        strips_use_sas: whether to use SAS Strips compiler (AODiscretization).
        use_strips_op: whether to use STRIPS operators when applying actions. Recommended to be False.
        use_tuple_desc: whether to use tuple description to prune the search space.
        forward_state_variables: whether to forward state variables before the search starts.
        forward_derived: whether to forward derived predicates after applying actions.
        track_most_promising_trajectory: whether to track the most promising trajectory.
        prob_goal_threshold: the probability threshold for the most promising trajectory. When there is a trajectory with probability greater than this threshold, the search will stop.
        verbose: whether to print verbose information.
        return_extra_info: whether to return extra information, such as the number of expanded nodes.

    Returns:
        the trajectory if succeeded, otherwise None.
        When `return_extra_info` is True, return a tuple of (trajectory, extra_info), where extra_info is a dictionary.
    """

    state, goal_expr, actions = prepare_search(
        'hsstrips', executor, state, goal_expr,
        actions=actions, action_filter=action_filter,
        continuous_values=continuous_values,
        forward_state_variables=forward_state_variables, forward_derived=forward_derived,
        verbose=verbose
    )

    if strips_use_sas:
        raise NotImplementedError('SAS strips is not implemented yet.')
    else:
        strips_translator = GStripsTranslatorOptimistic(executor, use_string_name=True, prob_goal_threshold=prob_goal_threshold)

    strips_task = strips_translator.compile_task(
        state, goal_expr, actions,
        is_relaxed=False,
        forward_relevance_analysis=strips_forward_relevance_analysis,
        backward_relevance_analysis=strips_backward_relevance_analysis,
    )

    if strips_heuristic == 'external' and external_heuristic_function is not None:
        pass
    else:
        heuristic = StripsHeuristic.from_type(
            strips_heuristic, strips_task, strips_translator,
            forward_relevance_analysis=strips_forward_relevance_analysis,
            backward_relevance_analysis=strips_backward_relevance_analysis,
        )

    # from IPython import embed; embed()
    # import ipdb; ipdb.set_trace()

    # print(strips_task.goal)
    # print(strips_task.operators)
    # print(heuristic.relaxed.goal)
    # print(heuristic.relaxed.operators)

    mpt_tracker = None
    if track_most_promising_trajectory:
        mpt_tracker = MostPromisingTrajectoryTracker(True, prob_goal_threshold)

    initial_state = HeuristicSearchState(state, strips_task.state, tuple(), 0)
    queue: List[QueueNode] = list()
    visited = set()

    if strips_heuristic == 'external' and external_heuristic_function is not None:
        def heuristic_fn(state: HeuristicSearchState, goal_expr=goal_expr) -> int:
            return external_heuristic_function(state.state, goal_expr)
    else:
        def heuristic_fn(state: HeuristicSearchState) -> int:
            return heuristic.compute(state.strips_state)
    priority_func = get_priority_func(heuristic_fn, heuristic_weight)

    def push_node(node: HeuristicSearchState):
        added = False
        if use_tuple_desc:
            nst = node.state.generate_tuple_description(executor.domain)
            if nst not in visited:
                added = True
                visited.add(nst)
        else:  # unconditionally expand
            added = True

        if added:
            hq.heappush(queue, QueueNode(priority_func(node, node.g), node))
            if heuristic_search_strips.DEBUG:
                print('  hsstrips::push_node:', *node.trajectory)
                print('   ', 'heuristic =', heuristic.compute(node.strips_state), 'g =', node.g)

    push_node(initial_state)

    nr_expanded_states = 0
    nr_tested_actions = 0

    def wrap_extra_info(trajectory):
        if return_extra_info:
            return trajectory, {'nr_expansions': nr_expanded_states, 'nr_tested_actions': nr_tested_actions}
        return trajectory

    pbar = None
    if verbose:
        pbar = jacinle.tqdm_pbar(desc='heuristic_search::expanding')

    while len(queue) > 0 and nr_expanded_states < max_expansions:
        priority, node = hq.heappop(queue)
        nr_expanded_states += 1

        """
        Name convention:
            - node: current node.
            - nnode: next node.
            - s: the state of the search tree.
            - ns: the state of the search tree after the action is applied.
            - a: the action applied.
            - ss: the strips state of the search tree.
            - nss: the strips state of the search tree after the action is applied.
            - traj: the path from the root to the node in the search tree.
            - nt: the path in the search tree after the action is applied.
            - g: the cost of the path from the root to the node in the search tree.
        """

        s, ss, traj = node.state, node.strips_state, node.trajectory
        if heuristic_search_strips.DEBUG:
            print('hsstrips::pop_node:')
            print('  trajectory:', *traj, sep='\n  ')
            print('  priority =', priority, 'g =', node.g)
            if heuristic_search_strips.DEBUG_INTERACTIVE:
                input('  Continue?')
        if verbose:
            pbar.set_description(f'heuristic_search::expanding: priority = {priority} g = {node.g}')
            pbar.update()

        if heuristic_search_strips.DEBUG:
            print('hsstrips::pop_node:', *traj)
            print('  priority =', priority, 'g =', node.g)

        goal_reached = goal_test(
            executor, s, goal_expr,
            trajectory=traj,
            verbose=verbose,
            mpt_tracker=mpt_tracker
        )
        if goal_reached:
            if verbose:
                print('hsstrips::search succeeded.')
                print('hsstrips::total_expansions:', nr_expanded_states)
            return wrap_extra_info(traj)

        if len(traj) >= max_depth:
            continue

        for sa in strips_task.operators:
            a = sa.raw_operator
            succ, ns = apply_action(executor, s, a)
            nt = traj + (a, )

            if succ:
                nss = sa.apply(ss) if use_strips_op else strips_translator.compile_state(ns.clone(), forward_derived=False)
                nnode = HeuristicSearchState(ns, nss, nt, node.g + 1)
                push_node(nnode)

    if verbose:
        print('hsstrips::search failed.')
        print('hsstrips::total_expansions:', nr_expanded_states)

    if mpt_tracker is not None:
        return wrap_extra_info(mpt_tracker.solution)

    return wrap_extra_info(None)


heuristic_search_strips.DEBUG = False
heuristic_search_strips.set_debug = lambda x = True: setattr(heuristic_search_strips, 'DEBUG', x)

heuristic_search_strips.DEBUG_INTERACTIVE = False
heuristic_search_strips.set_debug_interactive = lambda x = True: setattr(heuristic_search_strips, 'DEBUG_INTERACTIVE', x)

