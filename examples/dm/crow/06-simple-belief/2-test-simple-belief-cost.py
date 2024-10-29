#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : 2-test-simple-belief-cost.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/5/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import torch
import concepts.dm.crow as crow
from concepts.dsl.tensor_value import TensorValue


@crow.config_function_implementation(unwrap_values=False, use_object_names=False)
def move_to_target_belief_update(prior, target):
    # There is 0.8 probability that the new position is the target position.
    posterior = prior.tensor.clone()
    posterior.fill_(0)
    posterior[target.index] = 1.0

    posterior = prior.tensor * 0.2 + posterior * 0.8

    return TensorValue(prior.dtype, ['@0'], posterior)


@crow.config_function_implementation(unwrap_values=False, use_object_names=False)
def look_to_find_belief_update(prior, target, room):
    # A simple belief update for most-likely observation.
    # Rule: if the target is in the room, we will see it with 0.9 probability.
    # Otherwise, we will see it with 0.1 probability.

    prob_in = prior.tensor
    prob_out = 1 - prob_in

    posterior_in = prob_in * 0.9
    posterior_out = prob_out * 0.1
    rv = posterior_in / (posterior_in + posterior_out)

    print(f'prio={prior.tensor}, room={room}, target={target}, posterior_in={posterior_in}, posterior_out={posterior_out}, rv={rv}')
    return TensorValue(prior.dtype, [], rv)

@crow.config_function_implementation(unwrap_values=True, support_batch=True)
def log(x: torch.Tensor):
    return x.log()

@crow.config_function_implementation(unwrap_values=True, support_batch=True)
def sqrt(x: torch.Tensor):
    return x.sqrt()


def main():
    problem = crow.load_problem_file('belief-example-2.cdl')

    executor = problem.domain.make_executor()
    executor.register_function_implementation('move_to_target_belief_update', move_to_target_belief_update)
    executor.register_function_implementation('look_to_find_belief_update', look_to_find_belief_update)
    executor.register_function_implementation('log', log)
    executor.register_function_implementation('sqrt', sqrt)

    planner = crow.crow_regression(executor, problem, return_planner=True)
    planner.main()

    if len(planner.results) > 0:
        print(planner.results[0])


if __name__ == '__main__':
    main()
