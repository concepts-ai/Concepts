#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : csp_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/23/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import torch
from typing import Any, Union, Sequence, Tuple

from concepts.dm.crow.crow_domain import CrowState
from concepts.dm.crow.controller import CrowControllerApplier
from concepts.dm.crow.behavior import CrowEffectApplier
from concepts.dm.crow.executors.crow_executor import CrowExecutor
from concepts.dsl.dsl_types import ObjectConstant
from concepts.dsl.constraint import AssignmentDict, OptimisticValue, ground_assignment_value
from concepts.dsl.tensor_state import StateObjectReference
from concepts.dsl.tensor_value import TensorValue

__all__ = ['csp_ground_action', 'csp_ground_action_list']


def csp_ground_action(executor: CrowExecutor, action: Union[CrowControllerApplier, CrowEffectApplier], assignments: AssignmentDict) -> Union[CrowControllerApplier, CrowEffectApplier]:
    """Ground a single action with a given assignment to the underlying CSP. Basically, this function looks up the
    assigned value of each optimistic variable that appear in action parameters.

    Args:
        executor: the executor.
        action: the action to be grounded.
        assignments: the solution to the underlying CSP.

    Returns:
        the grounded action.
    """
    if isinstance(action, CrowControllerApplier):
        new_arguments = list()
        for arg in action.arguments:
            new_arguments.append(_ground_element(executor, assignments, arg))
        return CrowControllerApplier(action.controller, new_arguments, action.global_constraints, action.local_constraints)
    elif isinstance(action, CrowEffectApplier):
        new_bounded_variables = dict()
        for k, v in action.bounded_variables.items():
            new_bounded_variables[k] = _ground_element(executor, assignments, v)
        return CrowEffectApplier(action.statements, new_bounded_variables, global_constraints=action.global_constraints, local_constraints=action.local_constraints)
    else:
        raise TypeError(f'Unsupported action type: {type(action)}.')


def _ground_element(executor: CrowExecutor, assignments: AssignmentDict, arg: Any) -> StateObjectReference:
    if isinstance(arg, TensorValue):
        if arg.tensor_optimistic_values is not None:
            argv = arg.tensor_optimistic_values.item()
            assert isinstance(argv, int)
            argv = ground_assignment_value(assignments, argv)
            return argv
        else:
            return arg
    elif isinstance(arg, OptimisticValue):
        return ground_assignment_value(assignments, arg.identifier)
    elif isinstance(arg, StateObjectReference):
        return arg.name
    elif isinstance(arg, ObjectConstant):
        return arg.name
    elif isinstance(arg, str):
        return arg
    else:
        raise TypeError(f'Unsupported argument type: {type(arg)}.')


def csp_ground_action_list(executor: CrowExecutor, actions: Sequence[CrowControllerApplier], assignments: AssignmentDict) -> Tuple[CrowControllerApplier, ...]:
    """Ground a list of actions with a given assignment to the underlying CSP. Basically, this function looks up the
    assigned value of each optimistic variable that appear in action parameters.

    Args:
        executor: the executor.
        actions: the list of actions to be grounded.
        assignments: the solution to the underlying CSP.

    Returns:
        the list of grounded actions.
    """
    return tuple(csp_ground_action(executor, action, assignments) for action in actions)


def csp_ground_state(executor: CrowExecutor, state: CrowState, assignments: AssignmentDict) -> CrowState:
    """Map the CSP variable state to the new variable state."""

    new_state = state.clone()
    for feature_name, tensor_value in new_state.features.items():
        if tensor_value.tensor_optimistic_values is None:
            continue
        for ind in torch.nonzero(tensor_value.tensor_optimistic_values).tolist():
            ind = tuple(ind)
            identifier = tensor_value.tensor_optimistic_values[ind].item()
            if identifier in assignments:
                new_value = ground_assignment_value(assignments, identifier)
                if isinstance(new_value, OptimisticValue):
                    tensor_value.tensor_optimistic_values[ind] = new_value.identifier
                elif isinstance(new_value, TensorValue):
                    tensor_value.tensor[ind] = new_value.tensor
                    tensor_value.tensor_optimistic_values[ind] = 0
                else:
                    raise TypeError(f'Unknown value type {type(new_value)}')

    return new_state
