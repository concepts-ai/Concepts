#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : run.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/24/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import List
import numpy as np
import torch
import concepts.dsl.all as T
import concepts.dm.crow as crow


class SimpleEnvironment(object):
    def __init__(self, objects: List[str]):
        self.objects = objects
        self.f = np.zeros(len(objects), dtype=np.float32)

    def get_state(self, executor: crow.CrowExecutor):
        state = crow.CrowState.make_empty_state(executor.domain, {name: 'Object' for name in self.objects})
        state.batch_set_value('f', torch.tensor(self.f))
        return state


class SimpleEnvironmentSimulatorInterface(crow.CrowSimulationControllerInterface):
    def __init__(self, executor: crow.CrowExecutor, environment: SimpleEnvironment):
        super().__init__(executor)
        self.environment = environment
        self.saved_states = dict()
        self.saved_states_counter = 0

    def save_state(self, **kwargs) -> int:
        this_counter = self.saved_states_counter
        self.saved_states_counter += 1
        self.saved_states[this_counter] = {
            'f': self.environment.f.copy()
        }
        return this_counter

    def restore_state(self, state_identifier: int, **kwargs):
        state = self.saved_states[state_identifier]
        self.environment.f = state['f']
        del self.saved_states[state_identifier]

    def get_crow_state(self) -> crow.CrowState:
        """Get the state of the simulation interface."""
        return self.environment.get_state(self.executor)


def register_simulation_interface_controllers(simulator: SimpleEnvironmentSimulatorInterface):
    env = simulator.environment
    def ctl(x, a):
        index = env.objects.index(x)
        env.f[index] += a.item()

    simulator.register_controller('ctl', ctl)


def register_executor_functions(executor: crow.CrowExecutor):
    def valid_goal_f(f):
        return T.TensorValue.from_scalar(f > 6, T.BOOL)

    def gen_valid_action(x):
        a = np.random.uniform(0, 10)
        return T.TensorValue.from_scalar(a, T.FLOAT32)

    executor.register_function('valid_goal_f', valid_goal_f)
    executor.register_function('gen_valid_action', gen_valid_action)


def main():
    domain = crow.load_domain_file('domain.cdl')
    executor = domain.make_executor()
    env = SimpleEnvironment(['A', 'B'])
    sci = SimpleEnvironmentSimulatorInterface(executor, env)

    register_simulation_interface_controllers(sci)
    register_executor_functions(executor)

    problem = crow.CrowProblem.from_state_and_goal(
        executor.domain,
        env.get_state(executor),
        'valid_goal_f(f(B))'
    )

    results = crow.crow_regression(
        executor, problem, simulation_interface=sci, return_results=True, algo='priority_tree_v1',
        verbose=True
    )

    for r in results:
        print(r.controller_actions)


if __name__ == '__main__':
    main()

