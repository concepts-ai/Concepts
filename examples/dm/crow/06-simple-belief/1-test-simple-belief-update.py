#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : 1-test-simple-belief-update.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/04/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from concepts.dsl.tensor_value import TensorValue
import concepts.dm.crow as crow

@crow.config_function_implementation(unwrap_values=False, use_object_names=False)
def move_to_target_belief_update(prior, target):
    # There is 0.8 probability that the new position is the target position.
    posterior = prior.tensor.clone()
    posterior.fill_(0)
    posterior[target.index] = 1.0

    posterior = prior.tensor * 0.2 + posterior * 0.8

    return TensorValue(prior.dtype, ['@0'], posterior)


@crow.config_function_implementation(unwrap_values=False, use_object_names=False, include_executor_args=True)
def get_current_room(executor: crow.CrowExecutor, prior):
    state = executor.state
    return state.get_state_object_reference('Room', index=prior.tensor.argmax().item())


@crow.config_function_implementation(unwrap_values=False, use_object_names=False, include_executor_args=True)
def get_next_room(executor: crow.CrowExecutor, room):
    state = executor.state
    index = room.index + 1
    if index == state.get_nr_objects_by_type('Room'):
        return state.get_state_object_reference('Room', index=0)
    else:
        return state.get_state_object_reference('Room', index=index)


def main():
    problem = crow.load_problem_file('belief-example-1.cdl')

    executor = problem.domain.make_executor()
    executor.register_function_implementation('move_to_target_belief_update', move_to_target_belief_update)
    executor.register_function_implementation('get_current_room', get_current_room)
    executor.register_function_implementation('get_next_room', get_next_room)

    planner = crow.crow_regression(executor, problem, return_planner=True)
    planner.main()

    if len(planner.results) > 0:
        print(planner.results[0])


if __name__ == '__main__':
    main()
