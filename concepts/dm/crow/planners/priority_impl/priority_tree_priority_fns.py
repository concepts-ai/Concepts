#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : priority_tree_priority_fns.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/25/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import itertools
from typing import Optional, Union, Tuple

from concepts.dsl.expression import ValueOutputExpression
from concepts.dm.crow.controller import CrowControllerApplier, CrowControllerApplicationExpression
from concepts.dm.crow.behavior import CrowBehaviorOrderingSuite, CrowBehaviorCommit
from concepts.dm.crow.behavior import CrowBindExpression, CrowAssertExpression, CrowRuntimeAssignmentExpression, CrowAchieveExpression, CrowUntrackExpression, CrowBehaviorApplicationExpression, CrowBehaviorEffectApplicationExpression
from concepts.dm.crow.behavior import CrowBehaviorForeachLoopSuite, CrowBehaviorWhileLoopSuite, CrowBehaviorConditionSuite, CrowEffectApplier
from concepts.dm.crow.behavior_utils import execute_additive_heuristic_program
from concepts.dm.crow.executors.crow_executor import CrowExecutor
from concepts.dm.crow.planners.regression_planning import SupportedCrowExpressionType, ScopedCrowExpression, CrowPlanningResult3, CrowRegressionPlanner
from concepts.dm.crow.planners.regression_planning_impl.crow_regression_planner_priority_tree_v1 import MRProgram, PriorityTreeNodeData, PriorityTreeNode


class MixFunctionBase(object):
    @property
    def requires_g(self) -> bool:
        return False

    @property
    def requires_h(self) -> bool:
        return False

    def compute(self, g: Optional[float], node_h: Optional[float], edge_h: Optional[float]) -> float:
        raise NotImplementedError()


class GOnlyMixFunction(MixFunctionBase):
    @property
    def requires_g(self) -> bool:
        return True

    def compute(self, g: float, node_h: Optional[float], edge_h: Optional[float]) -> float:
        return g


class GFirstMixFunction(MixFunctionBase):
    @property
    def requires_g(self) -> bool:
        return True

    def compute(self, g: float, node_h: float, edge_h: Optional[float]) -> float:
        return -g


class BestFirstMixFunction(MixFunctionBase):
    @property
    def requires_h(self) -> bool:
        return True

    def compute(self, g: float, node_h: float, edge_h: float) -> float:
        return node_h + edge_h


class WeightedAStarMixFunction(MixFunctionBase):
    def __init__(self, w: float = 1.0):
        self.w = w

    @property
    def requires_g(self) -> bool:
        return True

    @property
    def requires_h(self) -> bool:
        return True

    def compute(self, g: float, node_h: float, edge_h: float) -> float:
        return g + self.w * (node_h + edge_h)


class PriorityFunctionBase(object):
    def __init__(self, executor: CrowExecutor):
        self.executor = executor

    def get_priority(self, node: PriorityTreeNode) -> float:
        raise NotImplementedError()


class FIFOPriorityFunction(PriorityFunctionBase):
    def get_priority(self, node: PriorityTreeNode) -> float:
        return 0


LIFO_PRIORITY_COUNTER = itertools.count()
class LIFOPriorityFunction(PriorityFunctionBase):
    def get_priority(self, node: PriorityTreeNode) -> float:
        return -next(LIFO_PRIORITY_COUNTER)


class AdditivePriorityFunctionBase(PriorityFunctionBase):
    def __init__(self, executor: CrowExecutor, mix_fn: MixFunctionBase):
        super().__init__(executor)
        self.mix_fn = mix_fn

    def get_priority(self, node: PriorityTreeNode) -> float:
        g = None if not self.mix_fn.requires_g else self.get_result_g(node.data.result, node.data.minimize)
        node_h = None if not self.mix_fn.requires_h else self.get_node_program_h(node.data.result, node.data.left, node.data.middle, node.data.right, node.data.minimize)
        edge_h = None if not self.mix_fn.requires_h else self.get_accumulated_edge_h(node.data.result, node)

        node.g = g
        node.node_h = node_h
        return self.mix_fn.compute(g, node_h, edge_h)

    def get_result_g(self, result: CrowPlanningResult3, minimize: Optional[ValueOutputExpression]) -> float:
        raise NotImplementedError()

    def get_node_program_h(self, result: CrowPlanningResult3, left: Tuple[ScopedCrowExpression, ...], middle: CrowBehaviorOrderingSuite, right: Tuple[ScopedCrowExpression, ...], minimize: Optional[ValueOutputExpression]) -> float:
        raise NotImplementedError()

    def get_accumulated_edge_h(self, result: CrowPlanningResult3, node: PriorityTreeNode) -> float:
        if node.accumulated_edge_h is not None:
            return node.accumulated_edge_h

        if len(node.parents) == 0:
            node.accumulated_edge_h = 0
            node.accumulated_edge_h_parent = None
            return 0

        best_h = None
        best_h_parent = None
        for stmt, parent in node.parents:
            this_h = self.get_edge_h(result, stmt, parent.data.minimize) + self.get_accumulated_edge_h(result, parent)
            if best_h is None or this_h < best_h:
                best_h = this_h
                best_h_parent = (stmt, parent)

        node.accumulated_edge_h = best_h
        node.accumulated_edge_h_parent = best_h_parent
        return best_h

    def get_edge_h(self, result: CrowPlanningResult3, edge: Union[ScopedCrowExpression, MRProgram, None], minimize: Optional[ValueOutputExpression]) -> float:
        raise NotImplementedError()


class UnitCostPriorityFunction(AdditivePriorityFunctionBase):
    def get_result_g(self, result: CrowPlanningResult3, minimize: Optional[ValueOutputExpression]) -> float:
        return len(result.controller_actions)

    def get_node_program_h(self, result: CrowPlanningResult3, left: Tuple[ScopedCrowExpression, ...], middle: CrowBehaviorOrderingSuite, right: Tuple[ScopedCrowExpression, ...], minimize: Optional[ValueOutputExpression]) -> float:
        args = (result, self.executor, minimize)
        return sum(_compute_unit_cost_h(x, *args) for x in left) + _compute_unit_cost_h(middle, *args) + sum(_compute_unit_cost_h(x, *args) for x in right)

    def get_edge_h(self, result: CrowPlanningResult3, edge: Union[ScopedCrowExpression, MRProgram, None], minimize: Optional[ValueOutputExpression]) -> float:
        return _compute_unit_cost_h(edge, result, self.executor, minimize)


class SimpleUnitCostPriorityFunction(AdditivePriorityFunctionBase):
    def get_result_g(self, result: CrowPlanningResult3, minimize: Optional[ValueOutputExpression]) -> float:
        return len(result.controller_actions)

    def get_node_program_h(self, result: CrowPlanningResult3, left: Tuple[ScopedCrowExpression, ...], middle: CrowBehaviorOrderingSuite, right: Tuple[ScopedCrowExpression, ...], minimize: Optional[ValueOutputExpression]) -> float:
        args = (result, self.executor, minimize)
        return sum(_compute_simple_unit_cost_h(x, *args) for x in left) + _compute_simple_unit_cost_h(middle, *args) + sum(_compute_simple_unit_cost_h(x, *args) for x in right)

    def get_edge_h(self, result: CrowPlanningResult3, edge: Union[ScopedCrowExpression, MRProgram, None], minimize: Optional[ValueOutputExpression]) -> float:
        return _compute_simple_unit_cost_h(edge, result, self.executor, minimize)


class SimpleAdditivePriorityFunction(AdditivePriorityFunctionBase):
    def get_result_g(self, result: CrowPlanningResult3, minimize: Optional[ValueOutputExpression]) -> float:
        return self.executor.execute(minimize, state=result.state).item()

    def get_node_program_h(self, result: CrowPlanningResult3, left: Tuple[ScopedCrowExpression, ...], middle: CrowBehaviorOrderingSuite, right: Tuple[ScopedCrowExpression, ...], minimize: Optional[ValueOutputExpression]) -> float:
        args = (result, self.executor, minimize)
        return sum(_compute_simple_additive_h(x, *args) for x in left) + _compute_simple_additive_h(middle, *args) + sum(_compute_simple_additive_h(x, *args) for x in right)

    def get_edge_h(self, result: CrowPlanningResult3, edge: Union[ScopedCrowExpression, MRProgram, None], minimize: Optional[ValueOutputExpression]) -> float:
        return _compute_simple_additive_h(edge, result, self.executor, minimize)


def _compute_unit_cost_h(
    data: Union[MRProgram, SupportedCrowExpressionType, ScopedCrowExpression, None],
    result: CrowPlanningResult3,
    executor: CrowExecutor,
    minimize: Optional[ValueOutputExpression]
) -> float:
    if data is None:
        return 0
    if isinstance(data, MRProgram):
        return sum(_compute_unit_cost_h(x, result, executor, minimize) for x in data.right) + _compute_unit_cost_h(data.middle, result, executor, minimize)
    if isinstance(data, ScopedCrowExpression):
        if isinstance(data.statement, CrowAchieveExpression):
            return execute_additive_heuristic_program(executor, data.statement, result.state, result.scopes[data.scope_id], minimize, is_unit_cost=True)
        else:
            return _compute_unit_cost_h(data.statement, result, executor, minimize)
    if isinstance(data, CrowControllerApplicationExpression):
        return 1
    if isinstance(data, CrowBehaviorOrderingSuite):  # Better handles while/if/foreach/alternative
        return sum(_compute_unit_cost_h(x, result, executor, minimize) for x in data.iter_statements())
    if isinstance(data, CrowBehaviorForeachLoopSuite):
        return sum(_compute_unit_cost_h(x, result, executor, minimize) for x in data.statements)
    if isinstance(data, SupportedCrowExpressionType):
        return 0
    raise ValueError(f'Invalid data type: {data} type={type(data)}')


def _compute_simple_unit_cost_h(
    data: Union[MRProgram, SupportedCrowExpressionType, ScopedCrowExpression, None],
    result: CrowPlanningResult3,
    executor: CrowExecutor,
    minimize: Optional[ValueOutputExpression]
) -> float:
    if data is None:
        return 0
    if isinstance(data, MRProgram):
        return sum(_compute_unit_cost_h(x, result, executor, minimize) for x in data.right) + _compute_unit_cost_h(data.middle, result, executor, minimize)
    if isinstance(data, ScopedCrowExpression):
        return _compute_unit_cost_h(data.statement, result, executor, minimize)
    if isinstance(data, (CrowBehaviorApplicationExpression, CrowAchieveExpression)):
        return 1
    if isinstance(data, CrowControllerApplicationExpression):
        return 0
    if isinstance(data, CrowBehaviorOrderingSuite):  # Better handles while/if/foreach/alternative
        return sum(_compute_unit_cost_h(x, result, executor, minimize) for x in data.iter_statements())
    if isinstance(data, CrowBehaviorForeachLoopSuite):
        return sum(_compute_unit_cost_h(x, result, executor, minimize) for x in data.statements)
    if isinstance(data, SupportedCrowExpressionType):
        return 0
    raise ValueError(f'Invalid data type: {data} type={type(data)}')


def _compute_simple_additive_h(
    data: Union[MRProgram, SupportedCrowExpressionType, ScopedCrowExpression, None],
    result: CrowPlanningResult3,
    executor: CrowExecutor,
    minimize: Optional[ValueOutputExpression]
) -> float:
    if data is None:
        return 0
    if isinstance(data, MRProgram):
        return sum(_compute_simple_additive_h(x, result, executor, minimize) for x in data.right) + _compute_simple_additive_h(data.middle, result, executor, minimize)
    if isinstance(data, ScopedCrowExpression):
        if isinstance(data.statement, CrowAchieveExpression):
            return execute_additive_heuristic_program(executor, data.statement, result.state, result.scopes[data.scope_id], minimize)
        elif isinstance(data.statement, CrowControllerApplicationExpression):
            return execute_additive_heuristic_program(executor, data.statement, result.state, result.scopes[data.scope_id], minimize)
        else:
            return _compute_simple_additive_h(data.statement, result, executor, minimize)
    if isinstance(data, CrowBehaviorOrderingSuite):
        return sum(_compute_simple_additive_h(x, result, executor, minimize) for x in data.iter_statements())
    if isinstance(data, CrowBehaviorForeachLoopSuite):
        return sum(_compute_simple_additive_h(x, result, executor, minimize) for x in data.statements)
    if isinstance(data, SupportedCrowExpressionType):
        return 0
    raise ValueError(f'Invalid data type: {data} type={type(data)}')

