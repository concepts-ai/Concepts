#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : execution_manager.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/13/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Any, Optional

from jacinle.logging import get_logger
from concepts.dm.crow.crow_domain import CrowProblem
from concepts.dm.crow.executors.crow_executor import CrowExecutor
from concepts.dm.crow.interfaces.perception_interface import CrowPerceptionInterface
from concepts.dm.crow.interfaces.controller_interface import CrowSimulationControllerInterface, CrowPhysicalControllerInterface

logger = get_logger(__file__)


class GoalAchieved(Exception):
    pass


class CrowExecutionManager(object):
    def __init__(
        self, executor: CrowExecutor,
        perception_interface: CrowPerceptionInterface,
        simulator_controller_interface: Optional[CrowSimulationControllerInterface],
        physical_controller_interface: CrowPhysicalControllerInterface,
    ):
        self._executor = executor
        self._perception_interface = perception_interface
        self._simulation_interface = simulator_controller_interface
        self._physical_interface = physical_controller_interface

        self._current_goal = None
        self._current_state = None

    def update_perception(self, action: Optional[Any] = None):
        self._perception_interface.step(action)

        if self._simulation_interface is not None:
            self._perception_interface.update_simulator()

        self._current_state = self._perception_interface.get_crow_state()

    def get_current_state(self):
        return self._current_state

    def run(self, goal, max_steps: int = 100):
        self.update_perception()
        self._init_planner(goal)
        for _ in range(max_steps):
            try:
                action = self._plan_next_action()
            except GoalAchieved:
                logger.critical('Goal achieved.')
                break
            self._physical_interface.step(action)
            self.update_perception(action)
        else:
            raise RuntimeError('Execution exceeds the maximum steps.')

    def _init_planner(self, goal):
        self._current_goal = goal

    def _plan_next_action(self):
        raise NotImplementedError()


class CrowDefaultOpenLoopExecutionManager(CrowExecutionManager):
    def __init__(
        self, executor: CrowExecutor,
        perception_interface: CrowPerceptionInterface,
        simulator_controller_interface: Optional[CrowSimulationControllerInterface],
        physical_controller_interface: CrowPhysicalControllerInterface,
        planner_options: Optional[dict] = None,
        planner_verbose: bool = False,
    ):
        super().__init__(executor, perception_interface, simulator_controller_interface, physical_controller_interface)

        self._plan = None
        self._current_action_idx = 0
        self._planner_options = planner_options if planner_options is not None else {}
        self._planner_verbose = planner_verbose

    def update_planner_options(self, **kwargs):
        self._planner_options.update(kwargs)

    def set_planner_verbose(self, verbose: bool = True):
        self._planner_verbose = verbose

    def run(self, goal, max_steps: int = 100, confirm_plan: bool = True):
        self.update_perception()
        self._init_planner(goal)
        if confirm_plan:
            input('Press Enter to execute the plan.')
        super().run(goal, max_steps)

    def _init_planner(self, goal):
        super()._init_planner(goal)

        if self._plan is not None:
            return

        from concepts.dm.crow.planners.regression_planning import crow_regression
        plans, _ = crow_regression(
            self._executor, CrowProblem.from_state_and_goal(self._executor.domain, self._current_state, self._current_goal),
            simulation_interface=self._simulation_interface,
            **self._planner_options,
            verbose=self._planner_verbose
        )
        if len(plans) == 0:
            raise GoalAchieved()
        self._plan = plans[0]
        self._current_action_idx = 0

        logger.critical('Plan: {}'.format([str(a) for a in self._plan]))

    def _plan_next_action(self):
        if self._current_action_idx >= len(self._plan):
            raise GoalAchieved()

        action = self._plan[self._current_action_idx]
        self._current_action_idx += 1
        return action
