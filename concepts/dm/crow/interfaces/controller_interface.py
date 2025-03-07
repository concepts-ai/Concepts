#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : controller_interface.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/16/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""A controller interface connects the controller commands output by a policy or a planner to the robot simulation/physical system.
Here, we distinguish between the simulation interface and the physical interface by whether they support state save/restore.
"""

import contextlib
from typing import Optional, Union, Sequence, Tuple, List, Dict, Callable
from dataclasses import dataclass

from concepts.dsl.tensor_value import TensorValue
from concepts.dm.crow.behavior import CrowEffectApplier
from concepts.dm.crow.controller import CrowControllerApplier
from concepts.dm.crow.crow_domain import CrowState
from concepts.dm.crow.executors.crow_executor import CrowExecutor

__all__ = ['CrowControllerExecutionError', 'CrowControllerInterfaceBase', 'CrowSimulationControllerInterface', 'CrowPhysicalControllerInterface']


class CrowControllerExecutionError(Exception):
    pass


@dataclass
class CrowControllerReflex(object):
    callable: Callable
    priority: int
    uid: int


class CrowControllerReflexCaller(object):
    def __init__(self, reflex_list: List[CrowControllerReflex]):
        self._reflex_list = reflex_list

    def __call__(self, *args, **kwargs):
        for reflex in self._reflex_list:
            result = reflex.callable(*args, **kwargs)
            if result is not None:
                return result


class CrowControllerInterfaceBase(object):
    """The base class for all controller interfaces.

    The convention of the controller interface is that it takes a controller name and a list of arguments, and then
    calls the corresponding controller function with the arguments. If the execution fails, it should raise an exception.
    """
    def __init__(self, executor: Optional[CrowExecutor] = None, mock: bool = False):
        self._executor = executor
        self._controllers = dict()
        self._controller_reflexes = dict()
        self._mock = mock
        self._controller_reflexes_uid = 0

    @property
    def executor(self) -> Optional[CrowExecutor]:
        return self._executor

    @property
    def controllers(self) -> Dict[str, Callable]:
        return self._controllers

    @property
    def controller_reflexes(self) -> Dict[Tuple[str, str], CrowControllerReflex]:
        return self._controller_reflexes

    def reset(self):
        pass

    def register_controller(self, name: str, function: Callable):
        self.controllers[name] = function
        return self

    def register_reflex(self, controller_name: str, reflex_name: str, function: Callable, priority: int = 10):
        self._controller_reflexes_uid += 1
        self.controller_reflexes[(controller_name, reflex_name)] = CrowControllerReflex(function, priority, self._controller_reflexes_uid)
        return self

    def reflex(self, controller_name: str, reflex_name: Optional[Union[str, Sequence[str]]] = None):
        if reflex_name is None:
            functions = [reflex for (name, _), reflex in self.controller_reflexes.items() if name == controller_name]
        elif isinstance(reflex_name, str):
            functions = [self.controller_reflexes[(controller_name, reflex_name)]]
        else:
            functions = [self.controller_reflexes[(controller_name, name)] for name in reflex_name]
        functions.sort(key=lambda x: (x.priority, x.uid), reverse=True)
        return CrowControllerReflexCaller(functions)

    def step(self, action: CrowControllerApplier, **kwargs) -> Optional[bool]:
        if isinstance(action, CrowEffectApplier):
            return
        return self.step_internal(action.name, *action.arguments, **kwargs)

    def step_without_error(self, action: CrowControllerApplier, **kwargs) -> bool:
        try:
            self.step(action, **kwargs)
        except CrowControllerExecutionError:
            return False
        return True

    def step_internal(self, name: str, *args, **kwargs) -> Optional[bool]:
        if self._mock:
            print('Mocked controller:', name)
            args = [arg.item() if isinstance(arg, TensorValue) and arg.dtype.is_pyobj_value_type else arg for arg in args]
            for i, arg in enumerate(args):
                print(f'  Arg {i}: {arg}')
            return

        if name not in self.controllers:
            raise ValueError(f"Controller {name} not found.")
        args = [arg.item() if isinstance(arg, TensorValue) and arg.dtype.is_pyobj_value_type else arg for arg in args]
        return self.controllers[name](*args, **kwargs)


class CrowSimulationControllerInterface(CrowControllerInterfaceBase):
    def __init__(self, executor: Optional[CrowExecutor] = None):
        super().__init__(executor)
        self._action_counter = 0

    def step_with_saved_state(self, action: CrowControllerApplier, **kwargs) -> Tuple[bool, int]:
        """Step with saved state. If the execution fails, return False and the state identifier.

        Args:
            action: the action to take.

        Returns:
            bool: whether the execution is successful.
            int: the state identifier.
        """
        state_identifier = self.save_state(**kwargs)
        try:
            self.step(action, **kwargs)
        except CrowControllerExecutionError:
            return False, state_identifier
        return True, state_identifier

    def step(self, action: CrowControllerApplier, **kwargs) -> None:
        try:
            return super().step(action, **kwargs)
        finally:
            self.increment_action_counter()

    def reset_action_counter(self):
        self._action_counter = 0

    def get_action_counter(self) -> int:
        return self._action_counter

    def increment_action_counter(self):
        self._action_counter += 1

    def save_state(self, **kwargs) -> int:
        raise NotImplementedError

    def restore_state(self, state_identifier: int, **kwargs):
        raise NotImplementedError

    def restore_state_keep(self, state_identifier: int, action_counter: Optional[int] = None, **kwargs):
        raise NotImplementedError

    def get_crow_state(self) -> CrowState:
        """Get the state of the simulation interface."""
        raise NotImplementedError()

    @contextlib.contextmanager
    def restore_context(self, verbose: bool = False, **kwargs):
        state_identifier = self.save_state()
        action_counter = self._action_counter
        try:
            yield
        finally:
            self.restore_state(state_identifier)
            self._action_counter = action_counter


class CrowPhysicalControllerInterface(CrowControllerInterfaceBase):
    pass
