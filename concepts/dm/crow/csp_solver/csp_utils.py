#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : csp_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/23/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import List, Sequence

from concepts.dm.crow.controller import CrowControllerApplier
from concepts.dm.crow.executors.crow_executor import CrowExecutor
from concepts.dsl.constraint import AssignmentDict, OptimisticValue, ground_assignment_value
from concepts.dsl.tensor_state import StateObjectReference
from concepts.dsl.tensor_value import TensorValue

__all__ = ['csp_ground_action', 'csp_ground_action_list']


def csp_ground_action(executor: CrowExecutor, action: CrowControllerApplier, assignments: AssignmentDict) -> CrowControllerApplier:
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
        elif isinstance(arg, OptimisticValue):
            new_arguments.append(ground_assignment_value(assignments, arg.identifier))
        elif isinstance(arg, StateObjectReference):
            new_arguments.append(arg.name)
        elif isinstance(arg, str):
            new_arguments.append(arg)
        else:
            raise TypeError(f'Unsupported argument type: {type(arg)}.')
    return CrowControllerApplier(action.controller, new_arguments)


def csp_ground_action_list(executor: CrowExecutor, actions: Sequence[CrowControllerApplier], assignments: AssignmentDict) -> List[CrowControllerApplier]:
    """Ground a list of actions with a given assignment to the underlying CSP. Basically, this function looks up the
    assigned value of each optimistic variable that appear in action parameters.

    Args:
        executor: the executor.
        actions: the list of actions to be grounded.
        assignments: the solution to the underlying CSP.

    Returns:
        the list of grounded actions.
    """
    return [csp_ground_action(executor, action, assignments) for action in actions]
