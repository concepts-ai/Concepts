#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : 3-test-simple-belief-cost-plan.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/5/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.
import jacinle
import torch
import concepts.dm.crow as crow
from concepts.dsl.tensor_value import TensorValue
from concepts.dsl.tensor_state import StateObjectReference


@crow.config_function_implementation(unwrap_values=False, use_object_names=False)
def move_to_target_belief_update(prior: TensorValue, target: StateObjectReference):
    # There is 0.8 probability that the new position is the target position.
    posterior = prior.tensor.clone()
    posterior.fill_(0)
    posterior[target.index] = 1.0

    posterior = prior.tensor * 0.2 + posterior * 0.8

    return TensorValue(prior.dtype, ['@0'], posterior)


@crow.config_function_implementation(unwrap_values=False, use_object_names=False)
def look_to_find_belief_update(prior: TensorValue, room: StateObjectReference, target: StateObjectReference):
    # A simple belief update for observation = True.
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
    problem = crow.load_problem_file('belief-example-3.cdl')

    executor = problem.domain.make_executor()
    executor.register_function_implementation('move_to_target_belief_update', move_to_target_belief_update)
    executor.register_function_implementation('look_to_find_belief_update', look_to_find_belief_update)
    executor.register_function_implementation('log', log)
    executor.register_function_implementation('sqrt', sqrt)

    # Have to turn on always_commit_skeleton so that the planner search for multiple plans.
    # Currently the planner does not support optimization for cost.
    # So currently, we basically return all the plans up to a certain depth and choose the best one post-hoc.
    planner = crow.crow_regression(executor, problem, always_commit_skeleton=False, return_planner=True, min_search_depth=8, max_search_depth=8, verbose=True)
    planner.main()

    # A known "Bug" is that currently the planner does not seen to prune the search space correctly.
    # So it generates a lot of plans that are the "same" when always_commit_skeleton=False.
    if len(planner.results) > 0:
        visualizations = list()
        for result in planner.results:
            visualizations.append((', '.join(str(x) for x in result.controller_actions), result.state.features['cost'].item()))

        # For visualization purposes, we sort the plans by cost and remove duplicates for now.
        visualizations = sorted(visualizations, key=lambda x: x[1])
        # Remove duplicates
        visualizations = [visualizations[0]] + [visualizations[i] for i in range(1, len(visualizations)) if visualizations[i][0] != visualizations[i-1][0]]

        print()
        print('Before duplicate removal: number of plans:', len(planner.results))
        print('After duplicate removal: number of plans:', len(visualizations))
        print()
        print(jacinle.tabulate(visualizations, headers=['Actions', 'Cost']))
    else:
        print('No plan found.')


if __name__ == '__main__':
    main()
