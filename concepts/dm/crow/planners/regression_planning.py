#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : regression_planning.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/17/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os
from dataclasses import dataclass
from typing import Any, Optional, Union, Iterator, Tuple, NamedTuple, List, Dict, Type

from jacinle.utils.meta import UNSET

from concepts.dsl.dsl_types import ObjectConstant
from concepts.dsl.value import ListValue
from concepts.dsl.constraint import ConstraintSatisfactionProblem, OptimisticValue
from concepts.dsl.expression import NullExpression, ValueOutputExpression, is_and_expr, ObjectOrValueOutputExpression, VariableAssignmentExpression
from concepts.dsl.tensor_value import TensorValue
from concepts.dsl.tensor_state import StateObjectReference, StateObjectList

from concepts.dm.crow.crow_function import CrowFunctionEvaluationMode
from concepts.dm.crow.crow_domain import CrowDomain, CrowProblem, CrowState
from concepts.dm.crow.controller import CrowControllerApplier, CrowControllerApplicationExpression
from concepts.dm.crow.behavior import CrowBehavior, CrowBehaviorCommit, CrowBehaviorEffectApplicationExpression
from concepts.dm.crow.behavior import CrowAssertExpression, CrowBindExpression, CrowMemQueryExpression, CrowAchieveExpression, CrowBehaviorApplicationExpression, CrowUntrackExpression
from concepts.dm.crow.behavior import CrowRuntimeAssignmentExpression, CrowFeatureAssignmentExpression
from concepts.dm.crow.behavior import CrowBehaviorOrderingSuite, CrowBehaviorCommit, CrowBehaviorConditionSuite, CrowBehaviorWhileLoopSuite, CrowBehaviorForeachLoopSuite
from concepts.dm.crow.behavior_utils import format_behavior_statement
from concepts.dm.crow.executors.crow_executor import CrowExecutor
from concepts.dm.crow.executors.generator_executor import CrowGeneratorExecutor
from concepts.dm.crow.interfaces.controller_interface import CrowSimulationControllerInterface
from concepts.dm.crow.interfaces.perception_interface import CrowPerceptionInterface

from concepts.dm.crow.planners.regression_dependency import RegressionTraceStatement

__all__ = [
    'SupportedCrowExpressionType', 'ScopedCrowExpression', 'CrowPlanningResult', 'CrowPlanningResult3',
    'CrowRegressionPlanner', 'crow_regression', 'get_crow_regression_algorithm', 'set_crow_regression_algorithm'
]


# TODO(Jiayuan Mao @ 2025/01/07): merge this using CrowBehaviorPrimitiveItem.
SupportedCrowExpressionType = Union[
    CrowBehaviorOrderingSuite, CrowBehaviorForeachLoopSuite, CrowBehaviorWhileLoopSuite, CrowBehaviorConditionSuite,
    CrowBindExpression, CrowMemQueryExpression, CrowAssertExpression, CrowRuntimeAssignmentExpression, CrowRuntimeAssignmentExpression,
    CrowControllerApplicationExpression, CrowAchieveExpression, CrowUntrackExpression,
    CrowBehaviorApplicationExpression, CrowBehaviorEffectApplicationExpression,
    CrowBehaviorCommit
]


class ScopedCrowExpression(NamedTuple):
    """A statement in the right stack of the planning state. This class is a named tuple so that it can be iterated as `for (stmt, scope_id) in ...`."""

    statement: SupportedCrowExpressionType
    """The statement."""

    scope_id: int
    """The scope id of the statement."""

    def __str__(self):
        return format_behavior_statement(self.statement, scope_id=self.scope_id)

    def __repr__(self):
        return self.__str__()


@dataclass
class CrowPlanningResult(object):
    state: CrowState
    csp: Optional[ConstraintSatisfactionProblem]
    controller_actions: Tuple[CrowControllerApplier, ...]
    scopes: Dict[int, Any]
    latest_scope: int

    @classmethod
    def make_empty(cls, state: CrowState) -> 'CrowPlanningResult':
        return cls(state, ConstraintSatisfactionProblem(), tuple(), dict(), 0)

    def clone(self, state=UNSET, csp=UNSET, controller_actions=UNSET, scopes=UNSET, latest_scope=UNSET) -> 'CrowPlanningResult':
        return CrowPlanningResult(
            state=state if state is not UNSET else self.state,
            csp=csp if csp is not UNSET else self.csp,
            controller_actions=controller_actions if controller_actions is not UNSET else self.controller_actions,
            scopes=scopes if scopes is not UNSET else self.scopes,
            latest_scope=latest_scope if latest_scope is not UNSET else self.latest_scope
        )


@dataclass
class CrowPlanningResult3(CrowPlanningResult):
    scope_constraints: Dict[int, List[ValueOutputExpression]]
    scope_constraint_evaluations: Dict[int, List[bool]]
    dependency_trace: Tuple['RegressionTraceStatement', ...]
    cost: float = 0.0
    parent_search_node: Any = None

    planner_root_node: Optional[Any] = None
    planner_current_node: Optional[Any] = None

    def get_state_index(self):
        return self.state.simulation_state_index + len(self.controller_actions)

    @classmethod
    def make_empty(cls, state: CrowState) -> 'CrowPlanningResult3':
        return CrowPlanningResult3(
            state, ConstraintSatisfactionProblem(state_timestamp=state.simulation_state_index), tuple(),
            scopes={0: dict()}, latest_scope=0,
            scope_constraints=dict(), scope_constraint_evaluations=dict(),
            dependency_trace=tuple()
        )

    def clone(
        self, state=UNSET, csp=UNSET, controller_actions=UNSET,
        scopes=UNSET, latest_scope=UNSET,
        scope_constraints=UNSET, scope_constraint_evaluations=UNSET,
        dependency_trace=UNSET
    ) -> 'CrowPlanningResult3':
        return CrowPlanningResult3(
            state=state if state is not UNSET else self.state,
            csp=csp if csp is not UNSET else self.csp,
            controller_actions=controller_actions if controller_actions is not UNSET else self.controller_actions,
            scopes=scopes if scopes is not UNSET else self.scopes,
            latest_scope=latest_scope if latest_scope is not UNSET else self.latest_scope,
            scope_constraints=scope_constraints if scope_constraints is not UNSET else self.scope_constraints,
            scope_constraint_evaluations=scope_constraint_evaluations if scope_constraint_evaluations is not UNSET else self.scope_constraint_evaluations,
            dependency_trace=dependency_trace if dependency_trace is not UNSET else self.dependency_trace,
            cost=self.cost,
            parent_search_node=self.parent_search_node,
            planner_root_node=self.planner_root_node,
            planner_current_node=self.planner_current_node
        )

    def clone_with_new_constraint(self, scope_id: int, constraint: ValueOutputExpression, evaluation: bool, do: bool) -> 'CrowPlanningResult3':
        if not do:
            return self

        scope_constraints = self.scope_constraints.copy()
        scope_constraint_evaluations = self.scope_constraint_evaluations.copy()

        if scope_id not in self.scope_constraints:
            scope_constraints[scope_id] = list()
            scope_constraint_evaluations[scope_id] = list()
        else:
            scope_constraints[scope_id] = self.scope_constraints[scope_id].copy()
            scope_constraint_evaluations[scope_id] = self.scope_constraint_evaluations[scope_id].copy()

        scope_constraints[scope_id].append(constraint)
        scope_constraint_evaluations[scope_id].append(evaluation)

        return self.clone(scope_constraints=scope_constraints, scope_constraint_evaluations=scope_constraint_evaluations)

    def clone_with_removed_constraint(self, scope_id: int, constraint: Union[ValueOutputExpression, NullExpression]) -> 'CrowPlanningResult3':
        if scope_id not in self.scope_constraints:
            return self

        if isinstance(constraint, NullExpression):
            # Remove all constraints at the scope.
            scope_constraints = self.scope_constraints.copy()
            scope_constraint_evaluations = self.scope_constraint_evaluations.copy()
            del scope_constraints[scope_id]
            del scope_constraint_evaluations[scope_id]
            return self.clone(scope_constraints=scope_constraints, scope_constraint_evaluations=scope_constraint_evaluations)

        found = None
        for i, c in enumerate(self.scope_constraints[scope_id]):
            if str(c) == str(constraint):
                found = i
                break

        if found is None:
            raise RuntimeError(f'Constraint not found: {constraint} in {self.scope_constraints[scope_id]}')

        scope_constraints = self.scope_constraints.copy()
        scope_constraint_evaluations = self.scope_constraint_evaluations.copy()
        scope_constraints[scope_id] = scope_constraints[scope_id][:found] + scope_constraints[scope_id][found + 1:]
        scope_constraint_evaluations[scope_id] = scope_constraint_evaluations[scope_id][:found] + scope_constraint_evaluations[scope_id][found + 1:]

        return self.clone(scope_constraints=scope_constraints, scope_constraint_evaluations=scope_constraint_evaluations)

    def iter_not_satisfied_constraints(self) -> Iterator[Tuple[int, int, ValueOutputExpression]]:
        for scope_id, constraints in self.scope_constraints.items():
            constraint_evaluations = self.scope_constraint_evaluations[scope_id]
            for constraint_id, constraint in enumerate(constraints):
                if len(constraint_evaluations) <= constraint_id or not constraint_evaluations[constraint_id]:
                    yield scope_id, constraint_id, constraint

    def all_scope_constraints(self) -> List[str]:
        from concepts.dm.crow.planners.regression_utils import replace_variable_with_value

        all_constraints = list()
        for scope_id, constraints in self.scope_constraints.items():
            for constraint in constraints:
                all_constraints.append(str(replace_variable_with_value(constraint, self.scopes[scope_id])) + f'@{scope_id}')
        return all_constraints


class CrowRegressionPlanner(object):
    def __init__(
        self, executor: CrowExecutor, state: CrowState, goal_expr: Union[str, ValueOutputExpression, None], *,
        perception_interface: Optional[CrowPerceptionInterface] = None,
        simulation_interface: Optional[CrowSimulationControllerInterface] = None,
        # Group 1: goal serialization and refinements.
        is_goal_ordered: bool = True,
        is_goal_serializable: bool = True,
        is_goal_refinement_compressible: bool = True,
        # Group 2: CSP solver.
        enable_csp: bool = True,
        max_csp_trials: int = 1,
        max_global_csp_trials: int = 100,
        max_csp_branching_factor: int = 5,
        use_generator_manager: bool = True,
        store_generator_manager_history: bool = False,
        # Group 3: output format.
        include_effect_appliers: bool = False,
        include_dependency_trace: bool = False,
        verbose: bool = False,
        **kwargs
    ):
        """Initialize the planner.

        Args:
            executor: the executor.
            state: the initial state.
            goal_expr: the goal expression.
            simulation_interface: the simulation interface.
            enable_reordering: whether to enable reordering.
            is_goal_ordered: whether the goal is ordered.
            is_goal_serializable: whether the goal is serializable.
            is_goal_refinement_compressible: whether the goal refinement is compressible.
            enable_csp: whether to enable the CSP solver.
            max_csp_trials: the maximum number of CSP trials.
            max_global_csp_trials: the maximum number of global CSP trials.
            max_csp_branching_factor: the maximum CSP branching factor.
            use_generator_manager: whether to use the generator manager.
            store_generator_manager_history: whether to store the generator manager history.
            include_effect_appliers: whether to include the effect appliers in the search result.
                The effect appliers are of type :class:`~concepts.dm.crow.behavior.CrowEffectApplier`.
            include_dependency_trace: whether to include the dependency graph in the search result.
            verbose: whether to output verbose information.
        """

        self.executor = executor
        self.state = state
        self.goal_expr = goal_expr

        if isinstance(self.goal_expr, str):
            self.goal_expr = executor.parse(self.goal_expr, state=state)

        self.executor.set_simulation_interface(simulation_interface)
        self.perception_interface = perception_interface
        self.simulation_interface = simulation_interface
        # TODO(Jiayuan Mao @ 2025/01/2): handle this statement in a better way...

        self.is_goal_ordered = is_goal_ordered
        self.is_goal_serializable = is_goal_serializable
        self.is_goal_refinement_compressible = is_goal_refinement_compressible

        self.enable_csp = enable_csp
        self.max_csp_trials = max_csp_trials
        self.max_global_csp_trials = max_global_csp_trials
        self.max_csp_branching_factor = max_csp_branching_factor
        self.use_generator_manager = use_generator_manager
        self.store_generator_manager_history = store_generator_manager_history

        self.include_effect_appliers = include_effect_appliers
        self.include_dependency_trace = include_dependency_trace
        self.verbose = verbose

        self.generator_manager = None
        if self.use_generator_manager:
            self.generator_manager = CrowGeneratorExecutor(executor, store_history=self.store_generator_manager_history)

        self._search_cache = dict()
        self._search_stat = {'nr_expanded_nodes': 0}
        self._results = list()
        self._post_init(**kwargs)

    @property
    def domain(self) -> CrowDomain:
        return self.executor.domain

    executor: CrowExecutor
    """The executor."""

    state: CrowState
    """The initial state."""

    goal_expr: Optional[ValueOutputExpression]
    """The goal expression. If this is None, the goal will be extracted from the behavior named '__goal__'."""

    simulation_interface: Optional[CrowSimulationControllerInterface]
    """The simulation interface."""

    is_goal_ordered: bool
    """Whether the goal is ordered. This only takes effect when the goal is a set of AND-connected expressions."""

    is_goal_serializable: bool
    """Whether the goal is serializable. This only takes effect when the goal is a set of AND-connected expressions."""

    is_goal_refinement_compressible: bool
    """Whether the refinement for each component of the goal is compressible. This only takes effect when the goal is a set of AND-connected expressions."""

    enable_csp: bool
    """Whether to enable the CSP solver."""

    max_csp_trials: int
    """The maximum number of CSP trials (for solving a single CSP)."""

    max_global_csp_trials: int
    """The maximum number of global CSP trials."""

    max_csp_branching_factor: int
    """The maximum CSP branching factor."""

    use_generator_manager: bool
    """Whether to use the generator manager."""

    store_generator_manager_history: bool
    """Whether to store the generator manager history."""

    include_effect_appliers: bool
    """Whether to include the effect appliers in the search result."""

    include_dependency_trace: bool
    """Whether to include the dependency graph in the search result."""

    verbose: bool
    """Whether to output verbose information."""

    def _post_init(self, **kwargs) -> None:
        pass

    def _make_goal_program(self) -> Tuple[CrowBehaviorOrderingSuite, Optional[ValueOutputExpression]]:
        if self.domain.has_behavior('__goal__'):
            return self.domain.get_behavior('__goal__').body, self.domain.get_behavior('__goal__').minimize

        if is_and_expr(self.goal_expr):
            if len(self.goal_expr.arguments) == 1 and self.goal_expr.arguments[0].return_type.is_list_type:
                goal_set = [self.goal_expr]
            else:
                goal_set = list(self.goal_expr.arguments)
        else:
            goal_set = [self.goal_expr]

        goal_set = [CrowAchieveExpression(x) for x in goal_set]
        assert_expr = CrowAssertExpression(self.goal_expr)
        if self.is_goal_ordered:
            goal_set = CrowBehaviorOrderingSuite.make_sequential(goal_set, variable_scope_identifier=0)
        else:
            goal_set = CrowBehaviorOrderingSuite.make_unordered(goal_set, variable_scope_identifier=0)
        if self.is_goal_serializable:
            program = CrowBehaviorOrderingSuite.make_sequential(
                goal_set,
                assert_expr,
                CrowBehaviorCommit(csp=True, sketch=True, execution=True),
                variable_scope_identifier=0
            )
        else:
            program = CrowBehaviorOrderingSuite.make_sequential(
                CrowBehaviorOrderingSuite.make_promotable(goal_set),
                assert_expr,
                CrowBehaviorCommit(csp=True, sketch=True, execution=True),
                variable_scope_identifier=0
            )
        return program, None

    @property
    def search_stat(self) -> dict:
        return self._search_stat

    @property
    def results(self) -> Union[List[CrowPlanningResult3], List[CrowPlanningResult]]:
        return self._results

    def set_results(self, results: Union[List[CrowPlanningResult3], List[CrowPlanningResult]]) -> None:
        self._results = results

    def main(self) -> Tuple[List[Tuple[CrowControllerApplier, ...]], dict]:
        program, minimize = self._make_goal_program()
        behavior_application = CrowBehaviorApplicationExpression(CrowBehavior('__goal__', [], None, program, always=True), [])
        program = CrowBehaviorOrderingSuite.make_sequential(behavior_application, variable_scope_identifier=0)

        candidate_plans = self.main_entry(program, minimize)
        if self.generator_manager is not None:
            for k, v in self.generator_manager.generator_calls_count.items():
                self._search_stat[f'gen_call/{k}'] = v
        return candidate_plans, self._search_stat

    def evaluate(
        self, expression: Union[ObjectOrValueOutputExpression, VariableAssignmentExpression], state: CrowState, csp: Optional[ConstraintSatisfactionProblem] = None,
        bounded_variables: Optional[Dict[str, Union[TensorValue, ObjectConstant]]] = None,
        clone_csp: bool = True,
        force_tensor_value: bool = False,
        state_index: Optional[int] = None
    ) -> Tuple[Union[None, StateObjectReference, StateObjectList, TensorValue, OptimisticValue], Optional[ConstraintSatisfactionProblem]]:
        """Evaluate an expression and return the result.

        Args:
            expression: the expression to evaluate.
            state: the current state.
            csp: the current CSP.
            bounded_variables: the bounded variables.
            clone_csp: whether to clone the CSP.
            force_tensor_value: whether to force the result to be a tensor value.

        Returns:
            the evaluation result and the updated CSP.
        """
        if clone_csp:
            csp = csp.clone() if csp is not None else None
        if bounded_variables is not None:
            bounded_variables = {k: v for k, v in bounded_variables.items() if not (k.startswith('__') and k.endswith('__'))}

        with self.executor.update_effect_mode(CrowFunctionEvaluationMode.SIMULATION, state_index=state_index):
            if isinstance(expression, VariableAssignmentExpression):
                self.executor.execute(expression, state=state, csp=csp, bounded_variables=bounded_variables)
                return None, csp

            rv = self.executor.execute(expression, state=state, csp=csp, bounded_variables=bounded_variables)
        if isinstance(rv, TensorValue):
            if force_tensor_value:
                return rv, csp
            if rv.is_scalar:
                return rv.item(), csp
            return rv, csp

        if isinstance(rv, ListValue) and len(rv.values) > 0:
            if isinstance(rv.values[0], StateObjectReference):
                rv = StateObjectList(rv.dtype, rv.values)

        assert isinstance(rv, StateObjectReference) or isinstance(rv, StateObjectList) or isinstance(rv, ListValue)
        return rv, csp

    def mem_query(
        self, expression: ObjectOrValueOutputExpression, state: CrowState, csp: Optional[ConstraintSatisfactionProblem] = None,
        bounded_variables: Optional[Dict[str, Union[TensorValue, ObjectConstant]]] = None,
        state_index: Optional[int] = None
    ) -> Tuple[CrowState, ConstraintSatisfactionProblem, Dict[str, Union[TensorValue, ObjectConstant]]]:
        if self.perception_interface is None:
            raise ValueError('No perception interface is provided.')
        return self.perception_interface.mem_query(expression, state, csp, bounded_variables, state_index)

    def main_entry(self, state: CrowBehaviorOrderingSuite, minimize: Optional[ValueOutputExpression]) -> List[Tuple[CrowControllerApplier, ...]]:
        raise NotImplementedError()


g_crow_regression_algorithm = None


def get_crow_regression_algorithm_mappings() -> Dict[str, Type[CrowRegressionPlanner]]:
    from concepts.dm.crow.planners.regression_planning_impl.crow_regression_planner_dfs_v1 import CrowRegressionPlannerDFSv1
    from concepts.dm.crow.planners.regression_planning_impl.crow_regression_planner_bfs_v1 import CrowRegressionPlannerBFSv1
    from concepts.dm.crow.planners.regression_planning_impl.crow_regression_planner_dfs_v2 import CrowRegressionPlannerDFSv2
    from concepts.dm.crow.planners.regression_planning_impl.crow_regression_planner_astar_v1 import CrowRegressionPlannerAStarv1
    from concepts.dm.crow.planners.regression_planning_impl.crow_regression_planner_iddfs_v1 import CrowRegressionPlannerIDDFSv1
    from concepts.dm.crow.planners.regression_planning_impl.crow_regression_planner_priority_tree_v1 import CrowRegressionPlannerPriorityTreev1

    return {
        'dfs_v1': CrowRegressionPlannerDFSv1,
        'bfs_v1': CrowRegressionPlannerBFSv1,
        'dfs_v2': CrowRegressionPlannerDFSv2,
        'astar_v1': CrowRegressionPlannerAStarv1,
        'iddfs_v1': CrowRegressionPlannerIDDFSv1,
        'priority_tree_v1': CrowRegressionPlannerPriorityTreev1
    }


def get_crow_regression_algorithm(algorithm: Optional[str]) -> Type[CrowRegressionPlanner]:
    mappings = get_crow_regression_algorithm_mappings()

    if algorithm is None:
        global g_crow_regression_algorithm
        if g_crow_regression_algorithm is None:
            g_crow_regression_algorithm = os.environ.get('CROW_REGRESSION_ALGO', 'iddfs_v1')
            print('Using default regression algorithm: {}'.format(g_crow_regression_algorithm))
        algorithm = g_crow_regression_algorithm

    if algorithm in mappings:
        return mappings[algorithm]

    raise ValueError(f'Unknown regression algorithm: {algorithm}. Available algorithms: {list(mappings.keys())}')


def set_crow_regression_algorithm(algorithm: str) -> None:
    if algorithm not in get_crow_regression_algorithm_mappings():
        raise ValueError(f'Unknown regression algorithm: {algorithm}. Available algorithms: {list(get_crow_regression_algorithm_mappings().keys())}')

    global g_crow_regression_algorithm
    g_crow_regression_algorithm = algorithm


def crow_regression(
    domain_or_executor_or_problem: Union[CrowExecutor, CrowDomain, CrowProblem], problem: Optional[CrowProblem] = None,
    goal: Optional[Union[str, ValueOutputExpression]] = None,
    return_planner: bool = False, return_results: bool = False,
    **kwargs
) -> Union[Tuple[list, dict], CrowRegressionPlanner, List[CrowPlanningResult], List[CrowPlanningResult3]]:
    if isinstance(domain_or_executor_or_problem, CrowExecutor):
        executor = domain_or_executor_or_problem
    elif isinstance(domain_or_executor_or_problem, CrowDomain):
        executor = domain_or_executor_or_problem.make_executor()
    elif isinstance(domain_or_executor_or_problem, CrowProblem):
        problem = domain_or_executor_or_problem
        executor = problem.domain.make_executor()
    else:
        raise ValueError(f'Unknown domain or executor: {domain_or_executor_or_problem}')

    merged_kwargs = problem.planner_options.copy()
    merged_kwargs.update(kwargs)
    kwargs = merged_kwargs

    algo = get_crow_regression_algorithm(kwargs.pop('algo', None))
    planner = algo(executor, problem.get_state_or_init(), goal if goal is not None else problem.goal, **kwargs)
    # planner.set_human_control_interface(True)
    if return_planner:
        return planner
    if return_results:
        planner.main()
        return planner.results
    return planner.main()

