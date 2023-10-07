#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : optimistic_search.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 11/20/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""CSP-based PDSketch planner. At the high-level,
this planner defers the decision of all continuous parameters to a CSP solver. During forward search,
the planner only make decisions about discrete parameters (mostly object-typed parameters), and record
all preconditions and goal tests that involve continuous parameters to an accumulated CSP data structure.

At each step, the planner will internally call a CSP solver to solve the accumulated CSP. If the CSP solver
returns a solution, the planner will generate the corresponding fully-grounded trajectory based on the
discrete parameters and the CSP solution.
"""

import heapq as hq
from dataclasses import dataclass
from typing import Any, Optional, Union, Sequence, Tuple, List, Dict, Callable

import jactorch

from concepts.dsl.dsl_types import NamedTensorValueType, PyObjValueType, UnnamedPlaceholder
from concepts.dsl.tensor_value import TensorValue
from concepts.dsl.expression import ValueOutputExpression
from concepts.dsl.constraint import OptimisticValue, EqualityConstraint, ConstraintSatisfactionProblem, AssignmentDict, ground_assignment_value
from concepts.algorithm.search.heuristic_search import QueueNode
from concepts.pdsketch.operator import OperatorApplier, gen_all_partially_grounded_actions
from concepts.pdsketch.domain import State, Domain
from concepts.pdsketch.executor import PDSketchExecutor
from concepts.pdsketch.csp_solvers.dpll_sampling import csp_dpll_sampling_solve, csp_dpll_simplify, CSPNoGenerator
from concepts.pdsketch.planners.search_utils import MostPromisingTrajectoryTracker
from concepts.pdsketch.strips.strips_expression import SState
from concepts.pdsketch.strips.strips_heuristics import StripsHeuristic
from concepts.pdsketch.strips.strips_grounding import GStripsTranslatorOptimistic
from concepts.pdsketch.strips.strips_search import get_priority_func


__all__ = [
    'instantiate_action', 'ground_action', 'ground_actions', 'apply_action', 'goal_test',
    'prepare_optimistic_search', 'construct_csp_from_optimistic_plan', 'solve_optimistic_plan', 'optimistic_search', 'optimistic_search_strips'
]


def instantiate_action(csp: ConstraintSatisfactionProblem, action: OperatorApplier) -> OperatorApplier:
    """Instantiate an action by replacing all placeholder values with a new optimistic variable.

    Args:
        csp: the CSP to which the new optimistic variable will be added.
        action: the action to be instantiated.

    Returns:
        the instantiated action.
    """
    new_arguments = list()
    for arg in action.arguments:
        if isinstance(arg, UnnamedPlaceholder):
            assert isinstance(arg.dtype, (NamedTensorValueType, PyObjValueType))
            new_arguments.append(TensorValue.from_optimistic_value(csp.new_actionable_var(arg.dtype, wrap=True)))
        else:
            new_arguments.append(arg)
    return OperatorApplier(action.operator, new_arguments)


def ground_action(executor: PDSketchExecutor, action: OperatorApplier, assignments: AssignmentDict) -> OperatorApplier:
    """Ground a single action with a given assignment to the underlying CSP. Basically, this function looks up the
    assigned value of each optimistic variable that appear in action parameters.

    Args:
        executor: the executor.
        action: the action to be grounded.
        assignments: the solution to the underlying CSP.

    Returns:
        the grounded action.
    """
    new_arguments = list()
    for arg in action.arguments:
        if isinstance(arg, TensorValue):
            if arg.tensor_optimistic_values is not None:
                argv = arg.tensor_optimistic_values.item()
                assert isinstance(argv, int)
                argv = ground_assignment_value(assignments, argv)
                new_arguments.append(argv)
            else:
                new_arguments.append(arg)
        elif isinstance(arg, str):
            new_arguments.append(arg)
        else:
            raise TypeError(f'Unsupported argument type: {type(arg)}.')
    return OperatorApplier(action.operator, new_arguments)


def ground_actions(executor: PDSketchExecutor, actions: Sequence[OperatorApplier], assignments: AssignmentDict) -> List[OperatorApplier]:
    """Ground a list of actions with a given assignment to the underlying CSP. Basically, this function looks up the
    assigned value of each optimistic variable that appear in action parameters.

    Args:
        executor: the executor.
        actions: the list of actions to be grounded.
        assignments: the solution to the underlying CSP.

    Returns:
        the grounded actions.
    """

    return [ground_action(executor, action, assignments) for action in actions]


def apply_action(executor: PDSketchExecutor, state: State, action: OperatorApplier, csp: ConstraintSatisfactionProblem) -> Tuple[Tuple[bool, State], ConstraintSatisfactionProblem]:
    """Apply an action to a state.

    Args:
        executor: the executor.
        state: the state to be updated.
        action: the action to be applied.
        csp: the CSP to which the new constraints will be added.

    Returns:
        a tuple of (goal test result, the updated state), and the updated CSP.
    """

    csp = csp.clone()
    succ, state = executor.apply(action, state, csp=csp)
    return (succ, state), csp


def goal_test(
    executor: PDSketchExecutor, state: State, goal_expr: ValueOutputExpression, csp: ConstraintSatisfactionProblem, trajectory: Sequence[OperatorApplier],
    csp_max_generator_trials: int = 3, mpt_tracker: Optional[MostPromisingTrajectoryTracker] = None,
    verbose: bool = False
) -> Optional[List[OperatorApplier]]:
    """Test whether a state satisfies a goal expression with DPLL+Sampling CSP solver.

    Args:
        executor: the executor.
        state: the state to be tested.
        goal_expr: the goal expression.
        csp: the CSP that contains all constraints that has been accumulated so far.
        trajectory: the trajectory that leads to the current state.
        csp_max_generator_trials: the maximum number of trials for calling CSP generators.
        mpt_tracker: the tracker for the most promising trajectory.
        verbose: whether to print verbose information.

    Returns:
        the trajectory of grounded actions that leads to the goal state if the goal is satisfied, otherwise None.
    """

    csp = csp.clone()
    rv = executor.execute(goal_expr, state=state, csp=csp).item()
    # print("  opt::goal_test", *trajectory, sep='\n    ')
    # print("  opt::goal_test", rv)
    if isinstance(rv, OptimisticValue):
        csp.add_constraint(EqualityConstraint.from_bool(rv, True), note='goal_test')
        try:
            if verbose:
                pass
                # print("  opt::final_csp_solve", *trajectory, sep='\n    ')
                # print("  opt::final_csp_solve", jacinle.indent_text(csp).lstrip())
            assignments = csp_dpll_sampling_solve(executor, csp, max_generator_trials=csp_max_generator_trials)
        except CSPNoGenerator:
            return None
        if assignments is not None:
            # TODO(Jiayuan Mao @ 2022/12/16): implement mpt_tracker?
            return ground_actions(executor, trajectory, assignments)
        else:
            return None
    else:
        rv = float(rv)
        threshold = mpt_tracker.threshold if mpt_tracker is not None else 0.5
        if rv > threshold or mpt_tracker is not None:
            if rv <= threshold:
                if not mpt_tracker.check(rv):
                    return None

            try:
                assignments = csp_dpll_sampling_solve(executor, csp, max_generator_trials=csp_max_generator_trials)
            except CSPNoGenerator:
                raise None
            if assignments is not None:
                plan = ground_actions(executor, trajectory, assignments)
                if mpt_tracker is not None:
                    mpt_tracker.update(rv, plan)
                if rv > threshold:
                    return ground_actions(executor, trajectory, assignments)
            else:
                return None
        else:
            return None


def optimistic_search_domain_check(domain: Domain):
    """Check whether a domain is suitable for optimistic search.

    Args:
        domain: the domain to be checked.

    Raises:
        NotImplementedError: if the domain is not suitable for optimistic search.
    """
    for op in domain.operators.values():
        if op.is_axiom:
            raise NotImplementedError('Optimistic search does not support axioms.')


def prepare_optimistic_search(
    func_name: str,
    executor: PDSketchExecutor, state: State, goal_expr: Union[str, ValueOutputExpression], *,
    actions: Optional[Sequence[OperatorApplier]] = None, action_filter: Callable[[OperatorApplier], bool] = None,
    verbose: bool = False, forward_state_variables=True, forward_derived=True
) -> Tuple[State, ValueOutputExpression, Sequence[OperatorApplier]]:
    """Prepare for optimistic search.

    Args:
        func_name: the name of the calling function.
        executor: the executor.
        state: the initial state.
        goal_expr: the goal expression.
        actions: the actions to be considered.
        action_filter: the action filter. If None, use all possible actions. It should be a function that takes in an action and returns a boolean.
        verbose: whether to print verbose information.
        forward_state_variables: whether to forward state variables.
        forward_derived: whether to forward derived variables.

    Returns:
        the initial state, the goal expression, and the actions to be considered.
    """
    # state = session.forward_predicates_and_axioms(state, forward_state_variables, False, forward_derived)
    if actions is None:
        actions = gen_all_partially_grounded_actions(executor, state, action_filter=action_filter)

    goal_expr = executor.domain.parse(goal_expr)

    if verbose:
        print(func_name + '::initial_state', state)
        print(func_name + '::actions nr', len(actions))
        if len(actions) < 20:
            for action in actions:
                print(' ', action)
        print(func_name + '::goal_expr', goal_expr)

    return state, goal_expr, actions


@jactorch.no_grad_func
def construct_csp_from_optimistic_plan(
    executor: PDSketchExecutor, state: State, goal_expr: Union[str, ValueOutputExpression], actions: Sequence[OperatorApplier], *,
    simplify: bool = False, verbose: bool = False
) -> Tuple[List[OperatorApplier], ConstraintSatisfactionProblem]:
    """Construct a CSP from an optimistic plan.

    Args:
        executor: the executor.
        state: the initial state.
        goal_expr: the goal expression.
        actions: a list of partially grounded actions.
        simplify: whether to simplify the CSP.
        verbose: whether to print verbose information.

    Returns:
        the list of grounded actions, and the constructed CSP.
    """

    optimistic_search_domain_check(executor.domain)

    csp = ConstraintSatisfactionProblem()
    if isinstance(goal_expr, str):
        goal_expr = executor.parse(goal_expr)
    else:
        assert isinstance(goal_expr, ValueOutputExpression)

    action_groundings = list()
    for a in actions:
        action_grounding = instantiate_action(csp, a)
        action_groundings.append(action_grounding)
        (succ, state), csp = apply_action(executor, state, action_grounding, csp)

        if succ:
            pass
        else:
            raise ValueError(f'Unable to perform action {action_grounding} at state {state}.')

    rv = executor.execute(goal_expr, state=state, csp=csp).item()
    if isinstance(rv, OptimisticValue):
        csp.add_constraint(EqualityConstraint.from_bool(rv, True), note='goal_test')

    if simplify:
        return action_groundings, csp_dpll_simplify(executor, csp)

    return action_groundings, csp


@jactorch.no_grad_func
def solve_optimistic_plan(
    executor: PDSketchExecutor, state: State, goal_expr: Union[str, ValueOutputExpression], actions: Sequence[OperatorApplier], *,
    csp_max_generator_trials: int = 3, verbose: bool = False
) -> Tuple[State, ConstraintSatisfactionProblem, Optional[List[OperatorApplier]]]:
    """Solve an optimistic plan using the DPLL+Sampling algorithm.

    Args:
        executor: the executor.
        state: the initial state.
        goal_expr: the goal expression.
        actions: a sequence of partially grounded actions.
        csp_max_generator_trials: the maximum number of trials for calling CSP generators.
        verbose: whether to print verbose information.

    Returns:
        a list of actions that solves the plan, or None if no solution is found.
    """

    optimistic_search_domain_check(executor.domain)
    csp = ConstraintSatisfactionProblem()
    if isinstance(goal_expr, str):
        goal_expr = executor.parse(goal_expr)
    else:
        assert isinstance(goal_expr, ValueOutputExpression)

    action_groundings = list()
    for a in actions:
        action_grounding = instantiate_action(csp, a)
        action_groundings.append(action_grounding)
        (succ, state), csp = apply_action(executor, state, action_grounding, csp)
        if succ:
            pass
        else:
            raise ValueError(f'Unable to perform action {action_grounding} at state {state}.')

    plan = goal_test(executor, state, goal_expr, csp, trajectory=action_groundings, csp_max_generator_trials=csp_max_generator_trials, verbose=verbose)
    if plan is not None:
        return state, csp, plan
    return state, csp, None


@jactorch.no_grad_func
def optimistic_search(
    executor: PDSketchExecutor, state: State, goal_expr: Union[str, ValueOutputExpression], *,
    max_depth: int = 5,
    csp_max_generator_trials: int = 3,
    actions: Optional[Sequence[OperatorApplier]] = None, action_filter: Callable[[OperatorApplier], bool] = None,
    use_tuple_desc: bool = False, use_csp_pruning: bool = True,
    forward_state_variables: bool = True, forward_derived: bool = False,
    verbose: bool = False, return_extra_info: bool = False
) -> Union[
    Optional[Sequence[OperatorApplier]],
    Tuple[Optional[Sequence[OperatorApplier]], Dict[str, Any]]
]:
    """Perform brute-force search with DPLL+Sampling algorithm for mixed discrete-continuous domains.

    Args:
        executor: the executor.
        state: the initial state.
        goal_expr: the goal expression.
        max_depth: the maximum depth of the search.
        csp_max_generator_trials: the maximum number of trials for calling CSP generators.
        actions: the actions to use. If None, use all possible actions. Partially grounded actions are allowed.
        action_filter: the action filter. If None, use all possible actions. It should be a function that takes in an action and returns a boolean.
        use_tuple_desc: whether to use tuple description to prune the search space.
        use_csp_pruning: whether to use partial CSP solver to prune the search space.
        forward_state_variables: whether to forward state variables before the search starts.
        forward_derived: whether to forward derived predicates after applying actions.
        verbose: whether to print verbose information.
        return_extra_info: whether to return extra information, such as the number of expanded nodes.

    Returns:
        the trajectory if succeeded, otherwise None.
        When `return_extra_info` is True, return a tuple of (trajectory, extra_info), where extra_info is a dictionary.
    """

    optimistic_search_domain_check(executor.domain)
    state, goal_expr, actions = prepare_optimistic_search(
        'opt', executor, state, goal_expr,
        actions=actions, action_filter=action_filter,
        verbose=verbose, forward_state_variables=forward_state_variables, forward_derived=forward_derived
    )

    assert not forward_derived, 'Not implemented.'

    states = [(state, tuple(), ConstraintSatisfactionProblem())]
    visited = set()
    if use_tuple_desc:
        visited.add(state.generate_tuple_description(executor.domain))

    nr_expanded_states = 0
    nr_tested_actions = 0

    def wrap_extra_info(trajectory):
        if return_extra_info:
            return trajectory, {'nr_expanded_states': nr_expanded_states, 'nr_tested_actions': nr_tested_actions}
        return trajectory

    for depth in range(max_depth):
        next_states = list()

        for s, traj, csp in states:
            nr_expanded_states += 1
            for a in actions:
                nr_tested_actions += 1
                ncsp = csp.clone()
                action_grounding = instantiate_action(ncsp, a)

                (succ, ns), ncsp = apply_action(executor, s, action_grounding, ncsp)
                nt = traj + (action_grounding, )

                if succ:
                    if use_csp_pruning:
                        try:
                            if not csp_dpll_sampling_solve(executor, ncsp, solvable_only=True, max_generator_trials=csp_max_generator_trials):
                                continue
                        except CSPNoGenerator:
                            pass

                    if use_tuple_desc:
                        nst = ns.generate_tuple_description(executor.domain)
                        if nst not in visited:
                            next_states.append((ns, nt, ncsp))
                            visited.add(nst)
                        else:
                            continue
                    else:  # unconditionally expand
                        next_states.append((ns, nt, ncsp))

                    plan = goal_test(executor, ns, goal_expr, ncsp, trajectory=nt, csp_max_generator_trials=csp_max_generator_trials, verbose=verbose)
                    if plan is not None:
                        if verbose:
                            print(f'opt::finished: depth={depth} nr_expanded_states={nr_expanded_states} nr_tested_actions={nr_tested_actions}.')
                        return wrap_extra_info(plan)

        states = next_states

        if verbose:
            print(f'opt::depth={depth}, this_layer_states={len(states)}')

    return wrap_extra_info(None)


@dataclass
class OptHeuristicSearchState(object):
    """The state for heuristic search."""

    state: State
    """The state."""

    strips_state: SState
    """The STRIPS state."""

    trajectory: Tuple[OperatorApplier, ...]
    """The trajectory."""

    csp: ConstraintSatisfactionProblem
    """The CSP that has been accumulated so far."""

    g: float
    """The cost so far."""


@jactorch.no_grad_func
def optimistic_search_strips(
    executor: PDSketchExecutor, state: State, goal_expr: Union[str, ValueOutputExpression],
    strips_heuristic: str = 'hff', *,
    max_expansions=100000, max_depth=100,  # search related parameters.
    csp_max_generator_trials: int = 3,
    heuristic_weight: float = float('inf'),  # heuristic related parameters.
    actions: Optional[Sequence[OperatorApplier]] = None, action_filter: Callable[[OperatorApplier], bool] = None,
    use_strips_op: bool = False,
    use_tuple_desc: bool = False, use_csp_pruning: bool = True,  # pruning related parameters.
    forward_state_variables: bool = True, forward_derived: bool = False,  # initialization related parameters.
    track_most_promising_trajectory: bool = False, prob_goal_threshold: float = 0.5,  # non-optimal trajectory tracking related parameters.
    verbose: bool = False, return_extra_info: bool = False
):
    """Perform heuristic search with DPLL+Sampling algorithm and STRIPS-based heuristics, for mixed discrete-continuous domains.

    Args:
        executor: the executor.
        state: the initial state.
        goal_expr: the goal expression.
        strips_heuristic: the heuristic to use. Should be a string.
        max_expansions: the maximum number of expanded nodes.
        max_depth: the maximum depth of the search.
        csp_max_generator_trials: the maximum number of trials for calling CSP generators.
        heuristic_weight: the weight of the heuristic. Use float('inf') to do greedy best-first search.
        actions: the actions to use. If None, use all possible actions. Partially grounded actions are allowed.
        action_filter: the action filter. If None, use all possible actions. It should be a function that takes in an action and returns a boolean.
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

    assert not forward_derived, 'Not implemented.'
    optimistic_search_domain_check(executor.domain)

    state, goal_expr, actions = prepare_optimistic_search(
        'optsstrips', executor, state, goal_expr,
        actions=actions, action_filter=action_filter,
        verbose=verbose, forward_state_variables=forward_state_variables, forward_derived=forward_derived
    )

    # TODO(Jiayuan Mao @ 2022/12/16): Relevance analysis for optimistic planning tasks.
    strips_translator = GStripsTranslatorOptimistic(executor, use_string_name=verbose)
    strips_task = strips_translator.compile_task(
        state, goal_expr, actions,
        is_relaxed=False,
        forward_relevance_analysis=False,
        backward_relevance_analysis=False
    )

    # TODO(Jiayuan Mao @ 2022/12/16): Support external heuristics.

    mpt_tracker = None
    if track_most_promising_trajectory:
        mpt_tracker = MostPromisingTrajectoryTracker(True, prob_goal_threshold)

    initial_state = OptHeuristicSearchState(state, strips_task.state, tuple(), ConstraintSatisfactionProblem(), 0)
    queue: List[QueueNode] = list()
    visited = set()

    heuristic = StripsHeuristic.from_type(strips_heuristic, strips_task, strips_translator)

    def heuristic_fn(state: OptHeuristicSearchState) -> int:
        return heuristic.compute(state.strips_state)
    priority_func = get_priority_func(heuristic_fn, heuristic_weight)

    def push_node(node: OptHeuristicSearchState):
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
            if optimistic_search_strips.DEBUG:
                print('  optsstrips::push_node:', *node.trajectory, sep='\n    ')
                print('   ', 'heuristic =', heuristic.compute(node.strips_state), 'g =', node.g)

    push_node(initial_state)

    nr_expanded_states = 0
    nr_tested_actions = 0

    def wrap_extra_info(trajectory):
        if return_extra_info:
            return trajectory, {'nr_expansions': nr_expanded_states, 'nr_tested_actions': nr_tested_actions}
        return trajectory

    is_initial_state = True
    while len(queue) > 0 and nr_expanded_states < max_expansions:
        priority, node = hq.heappop(queue)
        nr_expanded_states += 1

        s, ss, traj, csp = node.state, node.strips_state, node.trajectory, node.csp
        if optimistic_search_strips.DEBUG:
            print('optsstrips::pop_node:')
            print('  trajectory:', *traj, sep='\n  ')
            print('  priority =', priority, 'g =', node.g)
            if optimistic_search_strips.DEBUG_INTERACTIVE:
                input('  Continue?')

        if track_most_promising_trajectory and is_initial_state:
            is_initial_state = False
        else:
            plan = goal_test(
                executor, s, goal_expr, csp,
                trajectory=traj,
                csp_max_generator_trials=csp_max_generator_trials,
                verbose=verbose,
                mpt_tracker=mpt_tracker
            )
            if plan is not None:
                if verbose:
                    print('optsstrips::search succeeded.')
                    print('optsstrips::total_expansions:', nr_expanded_states)
                return wrap_extra_info(plan)

        if len(traj) >= max_depth:
            continue

        for sa in strips_task.operators:
            nr_tested_actions += 1
            a = sa.raw_operator
            ncsp = csp.clone()
            action_grounding = instantiate_action(ncsp, a)

            (succ, ns), ncsp = apply_action(executor, s, action_grounding, ncsp)
            nt = traj + (action_grounding, )

            if succ:
                if use_csp_pruning:
                    try:
                        if optimistic_search_strips.DEBUG:
                            print('  optsstrips::running CSP pruning...', *nt, sep='\n    ')
                        if not csp_dpll_sampling_solve(executor, ncsp, solvable_only=True, max_generator_trials=csp_max_generator_trials):
                            if optimistic_search_strips.DEBUG:
                                print('  optsstrips::pruned:', *nt, sep='\n    ')
                            continue
                    except CSPNoGenerator:
                        pass

                nss = sa.apply(ss) if use_strips_op else strips_translator.compile_state(ns.clone(), forward_derived=False)
                nnode = OptHeuristicSearchState(ns, nss, nt, ncsp, node.g + 1)
                push_node(nnode)

    if verbose:
        print('optsstrips::search failed.')
        print('optsstrips::total_expansions:', nr_expanded_states)

    if mpt_tracker is not None:
        return mpt_tracker.solution

    return None


optimistic_search_strips.DEBUG = False
optimistic_search_strips.set_debug = lambda x = True: setattr(optimistic_search_strips, 'DEBUG', x)

optimistic_search_strips.DEBUG_INTERACTIVE = False
optimistic_search_strips.set_debug_interactive = lambda x = True: setattr(optimistic_search_strips, 'DEBUG_INTERACTIVE', x)

