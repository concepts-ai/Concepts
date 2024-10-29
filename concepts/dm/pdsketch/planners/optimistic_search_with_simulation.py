#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : optimistic_search_with_simulation.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/26/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Planners using CSP solvers and simulators. The main difference between this file and the :mod:`concepts.dm.pdsketch.csp_solvers.dpll_sampling` is that
this file assumes that the CSP is a grounding of a planning domain problem. Therefore, it does not rely on a PDSketch definition of the
underlying physical system. Instead, it can directly simulate the physical system to check the validity of a solution."""

import collections
from typing import Any, Optional, Union, Sequence, Tuple, List, Dict

import jacinle

from concepts.dsl.dsl_types import BOOL
from concepts.dsl.expression import ValueOutputExpression
from concepts.dsl.constraint import ConstraintSatisfactionProblem, EqualityConstraint
from concepts.dm.pdsketch.executor import PDSketchExecutor, GeneratorManager
from concepts.dm.pdsketch.domain import State
from concepts.dm.pdsketch.operator import OperatorApplier

from concepts.dsl.constraint import AssignmentDict, Assignment, AssignmentType, OptimisticValue, SimulationFluentConstraintFunction
from concepts.dsl.tensor_value import TensorValue
from concepts.dm.pdsketch.predicate import Predicate
from concepts.dm.pdsketch.csp_solvers.dpll_sampling import CSPNotSolvable, CSPNoGenerator
from concepts.dm.pdsketch.csp_solvers.dpll_sampling import dpll_filter_deterministic_equal, dpll_filter_unused_rhs, dpll_filter_deterministic_clauses
from concepts.dm.pdsketch.csp_solvers.dpll_sampling import dpll_find_bool_variable, dpll_apply_assignments as _apply_assignments_old
from concepts.dm.pdsketch.csp_solvers.dpll_sampling import dpll_find_grounded_function_application, dpll_find_gen_variable_combined, dpll_find_typegen_variable
from concepts.dm.pdsketch.csp_solvers.dpll_sampling import ConstraintList

from concepts.dm.pdsketch.planners.optimistic_search import optimistic_search_domain_check, instantiate_action, ground_action, ground_actions
from concepts.dm.pdsketch.planners.optimistic_search_bilevel_utils import (
    enumerate_possible_symbolic_plans,
    enumerate_possible_symbolic_plans_regression_1,
    enumerate_possible_symbolic_plans_regression_c_v3
)
from concepts.dm.pdsketch.simulator_interface import PDSketchSimulatorInterface

__all__ = ['construct_csp_from_optimistic_plan', 'solve_optimistic_plan_with_simulation', 'optimistic_search_with_simulation', 'csp_dpll_sampling_solve_with_simulation']


def construct_csp_from_optimistic_plan(
    executor: PDSketchExecutor, state: State, goal_expr: Union[str, ValueOutputExpression], actions: Sequence[OperatorApplier]
) -> Tuple[Sequence[OperatorApplier], ConstraintSatisfactionProblem, State, State]:
    csp = ConstraintSatisfactionProblem()
    if isinstance(goal_expr, str):
        goal_expr = executor.parse(goal_expr)
    assert isinstance(goal_expr, ValueOutputExpression)

    action_index = 0
    action_groundings = list()
    initial_state, state = state, state.clone()
    for a in actions:
        action_grounding = instantiate_action(csp, a)

        for sub_action in action_grounding.iter_sub_operator_appliers():
            action_groundings.append(sub_action)
            succ, state = executor.apply(sub_action, state, csp=csp, clone=False, action_index=action_index)
            action_index += 1
            if succ:
                pass
            else:
                executor.apply_precondition_debug(sub_action, state, csp=csp)
                raise ValueError(f'Unable to perform action {sub_action} at state {state}.')

    rv = executor.execute(goal_expr, state=state, csp=csp).item()
    if isinstance(rv, OptimisticValue):
        csp.add_constraint(EqualityConstraint.from_bool(rv, True), note='goal_test')
    else:
        rv = float(rv)
        if rv < 0.5:
            raise ValueError(f'Unable to satisfy the goal {goal_expr} with the given action skeleton.')

    return action_groundings, csp, initial_state, state


def solve_optimistic_plan_with_simulation(
    executor: PDSketchExecutor, simulator: 'PDSketchSimulatorInterface', state: State, goal_expr: Union[str, ValueOutputExpression], actions: Sequence[OperatorApplier], *,
    generator_manager: Optional[GeneratorManager] = None,
    csp_max_trials: int = 1, csp_max_generator_trials: int = 3, verbose: bool = True
) -> Tuple[State, ConstraintSatisfactionProblem, Optional[List[OperatorApplier]]]:
    """Solve an optimistic plan using the DPLL+Sampling algorithm and an actual physics simulator."""

    optimistic_search_domain_check(executor.domain)
    simulator.set_init_state(state)
    action_groundings, csp, initial_state, last_state = construct_csp_from_optimistic_plan(executor, state, goal_expr, actions)

    if verbose:
        print('CSP', csp)

    for _ in range(csp_max_trials):
        if verbose:
            print('CSP trial', _)

        with executor.checkpoint_storage():
            assignments = csp_dpll_sampling_solve_with_simulation(
                executor, simulator, csp, initial_state, action_groundings,
                generator_manager=generator_manager,
                max_generator_trials=csp_max_generator_trials,
                verbose=verbose
            )

            if assignments is not None:
                return last_state, csp, ground_actions(executor, action_groundings, assignments)

    return last_state, csp, None


class OptimisticSearchSymbolicPlanner(jacinle.JacEnum):
    BRUTE_FORCE = 'brute_force'
    REGRESSION_1 = 'regression_1'
    REGRESSION_C = 'regression_c'


def optimistic_search_with_simulation(
    executor: PDSketchExecutor, simulator: 'PDSketchSimulatorInterface', state: State, goal_expr: Union[str, ValueOutputExpression], *,
    generator_manager: Optional[GeneratorManager] = None,
    symbolic_planner: Union[str, OptimisticSearchSymbolicPlanner] = OptimisticSearchSymbolicPlanner.BRUTE_FORCE,
    max_actions: int = 4, symbolic_planner_kwargs: Optional[Dict[str, Any]] = None,
    csp_max_trials: int = 1, csp_max_generator_trials: int = 3, verbose: bool = True, solver_verbose: bool = False, use_tqdm: bool = True
) -> Tuple[Optional[State], Optional[ConstraintSatisfactionProblem], Optional[List[OperatorApplier]]]:
    """Solve an optimistic plan using the DPLL+Sampling algorithm and an actual physics simulator."""

    symbolic_planner = OptimisticSearchSymbolicPlanner.from_string(symbolic_planner)
    optimistic_search_domain_check(executor.domain)

    if symbolic_planner is OptimisticSearchSymbolicPlanner.BRUTE_FORCE:
        symbolic_plan_generator = enumerate_possible_symbolic_plans(executor, state, goal_expr, max_actions=max_actions, verbose=verbose, **(symbolic_planner_kwargs or {}))
    elif symbolic_planner is OptimisticSearchSymbolicPlanner.REGRESSION_1:
        symbolic_plan_generator = enumerate_possible_symbolic_plans_regression_1(executor, state, goal_expr, max_actions=max_actions, verbose=verbose, **(symbolic_planner_kwargs or {}))
    elif symbolic_planner is OptimisticSearchSymbolicPlanner.REGRESSION_C:
        symbolic_planner_kwargs = symbolic_planner_kwargs or {}
        symbolic_planner_kwargs['enable_csp'] = True
        symbolic_plan_generator, _ = enumerate_possible_symbolic_plans_regression_c_v3(executor, state, goal_expr, max_actions=max_actions, verbose=verbose, **symbolic_planner_kwargs)
    else:
        raise ValueError(f'Unknown symbolic planner {symbolic_planner}.')

    simulator.set_init_state(state)
    # for actions, csp, initial_state, last_state in symbolic_plan_generator:
    for symbolic_plan in symbolic_plan_generator:
        actions, csp, initial_state, last_state = symbolic_plan.actions, symbolic_plan.csp, symbolic_plan.initial_state, symbolic_plan.last_state
        if verbose:
            print('Solving for: ', actions)

        if use_tqdm:
            trials = jacinle.tqdm(range(csp_max_trials))
        else:
            trials = range(csp_max_trials)
        for csp_trial_index in trials:
            if verbose and not use_tqdm:
                print('CSP trial', csp_trial_index)

            with executor.checkpoint_storage(), simulator.restore_context():
                assignments = csp_dpll_sampling_solve_with_simulation(
                    executor, simulator, csp, initial_state, actions,
                    generator_manager=generator_manager,
                    max_generator_trials=csp_max_generator_trials,
                    verbose=solver_verbose
                )

                if assignments is not None:
                    return last_state, csp, ground_actions(executor, actions, assignments)

    return None, None, None


def _filter_unused_simulation_rhs(executor: PDSketchExecutor, constraints: ConstraintList, assignments: Dict[int, Assignment]) -> ConstraintList:
    """Filter out simulation constraints that only appear once in the RHS of the constraints. In this case, the variable can be ignored and the related constraints can be removed.

    Args:
        executor: the executor.
        constraints: the list of constraints.
        assignments: the dictionary of assignments.

    Returns:
        the list of constraints that have not been satisfied, after removing all unused variables.
    """
    used = collections.defaultdict(int)
    for c in constraints:
        if c.is_group_constraint:
            continue
        for x in c.arguments:
            if isinstance(x, OptimisticValue):
                used[x.identifier] += 100  # as long as a variable has appeared in the lhs of a constraint, it is used.
        if isinstance(c.rv, OptimisticValue) and isinstance(c.function, SimulationFluentConstraintFunction):
            used[c.rv.identifier] += 1  # if the variable has only appeared in the rhs of a constraint for once, it is not used.
    for k, v in used.items():
        if v == 1:
            assignments[k] = Assignment(AssignmentType.IGNORE, None)
    return _apply_assignments_old(executor, constraints, assignments)


def _apply_assignments_with_simulation(
    executor: PDSketchExecutor,
    constraints: ConstraintList, assignments: Dict[int, Assignment],
    simulator: PDSketchSimulatorInterface, actions: Sequence[OperatorApplier],
    verbose: bool = False
) -> Tuple[ConstraintList, AssignmentDict]:
    new_constraints = _apply_assignments_old(executor, constraints, assignments)
    new_assignments = None

    while True:
        next_action_index = simulator.last_action_index + 1
        if next_action_index < len(actions):
            action = actions[next_action_index]

            try:
                grounded_action = ground_action(executor, action, assignments)
            except AssertionError:
                # The action is not applicable.
                break

            if verbose:
                jacinle.log_function.print(f'Executing grounded action: #{next_action_index}')

            action_name = grounded_action.operator.controller.name
            action_args = executor.get_controller_args(grounded_action, simulator.get_latest_pd_state())
            succ, state = simulator.run(
                next_action_index,
                action_name,
                action_args
            )
            if not succ:
                # jacinle.log_function.print(f'Action {grounded_action} failed.')
                raise CSPNotSolvable(f'Unable to perform action {action}.')
            # jacinle.log_function.print(f'Action {grounded_action} succeeded.')

            if new_assignments is None:
                new_assignments = assignments.copy()

            # Update the assignments.
            for i, c in enumerate(new_constraints):
                if isinstance(c.function, SimulationFluentConstraintFunction):
                    function: SimulationFluentConstraintFunction = c.function
                    if function.action_index == next_action_index:
                        if isinstance(c.rv, TensorValue):
                            if c.rv.dtype != BOOL:
                                raise NotImplementedError('Only bool is supported for simulation constraints.')

                            # print('Simulation constraint', c)
                            # print(state.features[function.predicate.name])
                            # print('Desired value =', c.rv.value, 'Actual value =', state.features[function.predicate.name][function.arguments].item())
                            # import pybullet
                            # pybullet.stepSimulation()
                            # import ipdb; ipdb.set_trace()

                            if state.features[function.predicate.name][function.arguments].item() != c.rv.item():
                                raise CSPNotSolvable()
                            else:
                                new_constraints[i] = None
                        else:
                            new_assignments[c.rv.identifier] = Assignment(
                                AssignmentType.VALUE,
                                state.features[function.predicate.name][function.arguments]
                            )
                            new_constraints[i] = None

            new_constraints = _apply_assignments_old(executor, new_constraints, new_assignments)
        else:
            break

    return new_constraints, new_assignments if new_assignments is not None else assignments


def csp_dpll_sampling_solve_with_simulation(
    executor: PDSketchExecutor, simulator: PDSketchSimulatorInterface, csp: ConstraintSatisfactionProblem,
    state: State, actions: Sequence[OperatorApplier], *,
    generator_manager: Optional[GeneratorManager] = None,
    max_generator_trials: int = 3,
    enable_ignore: bool = False, solvable_only: bool = False,
    verbose: bool = False
) -> Optional[Union[bool, AssignmentDict]]:
    if generator_manager is None:
        generator_manager = GeneratorManager(executor, store_history=False)

    constraints = csp.constraints.copy()

    @jacinle.log_function(verbose=False)
    def dfs(constraints, assignments):
        if len(constraints) == 0:
            return assignments

        progress = True
        while progress:
            progress, constraints = dpll_filter_deterministic_equal(executor, constraints, assignments)
        if enable_ignore:
            constraints = dpll_filter_unused_rhs(executor, constraints, assignments, csp.index2record)
        else:
            # NB(Jiayuan Mao @ 2023/03/11): for simulation constraints, we can remove them if they are not used in any other constraints.
            # TODO(Jiayuan Mao @ 2023/03/15): actually, I just noticed that this is a "bug" in the implementation of `_find_bool_variable`.
            # Basically, if the variable is a simulation variable and it only appears in the RHS of a constraint, then it will be ignored.
            # The current handling will work, but probably I need to think about a better way to handle this.
            constraints = _filter_unused_simulation_rhs(executor, constraints, assignments)

        progress = True
        while progress:
            progress, constraints = dpll_filter_deterministic_clauses(executor, constraints, assignments)

        if verbose:
            jacinle.log_function.print('Remaining constraints:', len(constraints))
            jacinle.log_function.print(*constraints, sep='\n')

        if len(constraints) == 0:
            return assignments

        if (next_bool_var := dpll_find_bool_variable(executor, constraints, assignments)) is not None:
            assignments_true = assignments.copy()
            assignments_true[next_bool_var] = Assignment(AssignmentType.VALUE, True)

            with simulator.restore_context(verbose):
                try:
                    constraints_true, assignments_true = _apply_assignments_with_simulation(executor, constraints, assignments_true, simulator, actions, verbose=verbose)
                    return dfs(constraints_true, assignments_true)
                except CSPNotSolvable:
                    pass

            assignments_false = assignments.copy()
            assignments_false[next_bool_var] = Assignment(AssignmentType.VALUE, False)
            with simulator.restore_context(verbose):
                try:
                    constraints_false, assignments_false = _apply_assignments_with_simulation(executor, constraints, assignments_false, simulator, actions, verbose=verbose)
                    return dfs(constraints_false, assignments_false)
                except CSPNotSolvable:
                    pass

            raise CSPNotSolvable()
        elif (next_fapp := dpll_find_grounded_function_application(executor, constraints)) is not None:
            function: Predicate = next_fapp.function
            arguments = next_fapp.arguments

            external_function = executor.get_function_implementation(function.name)
            output = external_function(*arguments)

            target = next_fapp.rv
            new_assignments = assignments.copy()
            new_assignments[target.identifier] = Assignment(AssignmentType.VALUE, output)
            with simulator.restore_context(verbose):
                try:
                    new_constraints = constraints.copy()
                    new_constraints[new_constraints.index(next_fapp)] = None
                    new_constraints, new_assignments = _apply_assignments_with_simulation(executor, new_constraints, new_assignments, simulator, actions, verbose=verbose)
                    return dfs(new_constraints, new_assignments)
                except CSPNotSolvable:
                    pass

            raise CSPNotSolvable()
        elif (next_gen_vars := dpll_find_gen_variable_combined(executor, csp, constraints, assignments)) is not None:
            if len(next_gen_vars) > 1:
                # jacinle.log_function.print('Generator orders', *[str(vv[1]).split('\n')[0] for vv in next_gen_vars], sep='\n  ')
                pass

            for vv in next_gen_vars:
                c, g, args, outputs_target = vv
                if g.unsolvable:
                    raise CSPNotSolvable('Hit unsolvable generator.')

                if verbose:
                    jacinle.log_function.print(f'Generator: {g}\nArgs: {args}')
                generator = generator_manager.call(g, max_generator_trials, args, c)

                # NB(Jiayuan Mao @ 2023/03/03): I didn't write for x in generator in order to make the verbose output more readable.
                generator = iter(generator)
                for j in range(max_generator_trials):
                    if verbose:
                        jacinle.log_function.print('Running generator', g.name, f'count = {j + 1} / {max_generator_trials}')

                    try:
                        output_index, outputs = next(generator)
                    except StopIteration:
                        if verbose:
                            jacinle.log_function.print('Generator', g.name, 'exhausted.')
                        break

                    new_assignments = assignments.copy()
                    for output, target in zip(outputs, outputs_target):
                        new_assignments[target.identifier] = Assignment(AssignmentType.VALUE, output, generator_index=output_index)
                        # jacinle.log_function.print('Assigned', target, output)

                    with simulator.restore_context(verbose):
                        try:
                            new_constraints = constraints.copy()
                            if isinstance(c, list):
                                for cc in c:
                                    new_constraints[new_constraints.index(cc)] = None
                            else:
                                new_constraints[new_constraints.index(c)] = None

                            new_constraints, new_assignments = _apply_assignments_with_simulation(executor, new_constraints, new_assignments, simulator, actions, verbose=verbose)
                            return dfs(new_constraints, new_assignments)
                        except CSPNotSolvable as e:
                            if verbose:
                                jacinle.log_function.print('Failed to apply assignments. Reason:', e)
                            pass

            raise CSPNotSolvable()
        else:
            # jacinle.log_function.print('Can not find a generator. Constraints:\n  ' + '\n  '.join([str(x) for x in constraints]))
            raise CSPNoGenerator('Can not find a generator. Constraints:\n  ' + '\n  '.join([str(x) for x in constraints]))

    try:
        assignments = dfs(constraints, {})
        if not solvable_only:
            for name, record in csp.index2record.items():
                dtype = record.dtype
                if name not in assignments:
                    g = dpll_find_typegen_variable(executor, dtype)
                    if g is None:
                        raise NotImplementedError('Can not find a generator for unbounded variable {}, type {}.'.format(name, dtype))
                    else:
                        try:
                            output_index, (output, ) = next(iter(generator_manager.call(g, 1, [], None)))
                            assignments[name] = Assignment(AssignmentType.VALUE, output, generator_index=output_index)
                        except StopIteration:
                            raise CSPNotSolvable

        if generator_manager.store_history:
            generator_manager.mark_success(assignments)
        if solvable_only:
            return True
        return assignments
    except CSPNotSolvable:
        return None
    except CSPNoGenerator:
        raise

