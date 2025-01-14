#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : crow_regression_planner_dfs_v1_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/17/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Optional, Sequence, Tuple, Dict

from concepts.dsl.constraint import ConstraintSatisfactionProblem
from concepts.dm.crow import CrowBehavior, CrowExecutor, CrowState
from concepts.dm.crow.planners.regression_utils import canonicalize_bounded_variables
from concepts.dm.crow.planners.regression_planning import CrowPlanningResult


def execute_behavior_effect(executor: CrowExecutor, behavior: CrowBehavior, state: CrowState, scope: dict, csp: Optional[ConstraintSatisfactionProblem] = None, state_index: Optional[int] = None) -> CrowState:
    new_state = state.clone()
    for effect in behavior.effect_body.statements:
        with executor.update_effect_mode(effect.evaluation_mode, state_index=state_index):
            executor.execute(effect.assign_expr, state=new_state, csp=csp, bounded_variables=scope)
    return new_state


def execute_behavior_effect_batch(executor: CrowExecutor, results: Sequence[CrowPlanningResult], behavior: CrowBehavior, scope_id: int, csp: Optional[ConstraintSatisfactionProblem] = None) -> Sequence[CrowPlanningResult]:
    # print('!!!Apply behavior effect:', action.short_str(), 'with scope:', results[0].scopes[scope_id])
    new_results = list()
    for result in results:
        state = execute_behavior_effect(executor, behavior, result.state, canonicalize_bounded_variables(result.scopes, scope_id), csp=csp)
        new_results.append(CrowPlanningResult(state, result.csp, result.controller_actions, result.scopes))
    return new_results


def unique_results(results: Sequence[CrowPlanningResult]) -> Tuple[CrowPlanningResult, ...]:
    def keyfunc(x):
        return tuple(map(str, x.controller_actions))
    results: Dict[Tuple[str, ...], CrowPlanningResult] = {keyfunc(x): x for x in results}
    return tuple(results.values())
