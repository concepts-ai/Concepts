#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : simulator_interface.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/16/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import contextlib
from typing import Any, Iterable, Tuple, Dict, Callable

import jacinle

from concepts.dm.pdsketch.operator import OperatorApplier
from concepts.dm.pdsketch.executor import PythonFunctionRef, PDSketchExecutor
from concepts.dm.pdsketch.domain import State

__all__ = ['PDSketchSimulatorInterface', 'PDSketchExecutionInterface']


class PDSketchSimulatorInterface(object):
    """The base class for interaction with a (physical simulator). This class serves two purposes:

    1. perform simulation for planners. Therefore, it requires actions to support `restore` operations.
    2. perform simulation when executing a given plan. In this case, restore operations are not required.
    """

    def __init__(self, executor: PDSketchExecutor):
        """Initialize the simulator interface.

        Args:
            executor: the :class:`~concepts.dm.pdsketch.executor.PDSketchExecutor` instance for grounding action parameters.
        """
        self._executor = executor
        self._last_action_index = -1

        self.controllers = dict()
        self.pd_states = dict()
        self.restore_functions = dict()

    controllers: Dict[str, Callable]
    """Registered controllers."""

    pd_states: Dict[int, State]
    """Mappings from action indices to PDSketch states. -1 is the initial state."""

    restore_functions: Dict[int, Any]
    """Mappings from action indices to controller states. -1 is the initial state."""

    def register_action_controller(self, name: str, controller_function: Callable):
        """Register a controller function for a controller name.

        Args:
            name: the name of the controller.
            controller_function: the controller function. It should take a PDSketch state as the first argument, together with other action arguments,
                and return a tuple of (success, new PDSketch state, restore function).
        """
        if not isinstance(controller_function, PythonFunctionRef):
            controller_function = PythonFunctionRef(controller_function, executor=self._executor)

        self.controllers[name] = controller_function.set_executor(self._executor)

    def set_init_state(self, state: State):
        """Set the initial PDSketch state of the simulator."""
        self.pd_states[-1] = state

    def get_pd_state(self, action_index: int) -> State:
        """Get the PDSketch state at a given action index. -1 is the initial state.

        Args:
            action_index: the action index.
        """
        return self.pd_states[action_index]

    def get_latest_pd_state(self) -> State:
        """Get the latest PDSketch state."""
        return self.pd_states[self._last_action_index]

    def get_restore_function(self, action_index: int) -> Any:
        """Get the restore function for a given action index.

        Args:
            action_index: the action index.
        """
        return self.restore_functions[action_index]

    def run_operator_applier(self, action_index: int, action: OperatorApplier) -> Tuple[bool, State]:
        """Execute an action, and return the success flag and the new PDSketch state. This function will not store the restore function corresponding to the action.

        Args:
            action_index: the action index.
            action: the action to execute.

        Returns:
            a tuple of (success, new PDSketch state).
        """

        action_name = action.operator.controller.name
        action_args = self._executor.get_controller_args(action, self.get_latest_pd_state())
        return self.run(action_index, action_name, action_args)

    def run(self, action_index: int, action_name: str, action_args: Tuple[Any, ...], verbose: bool = False) -> Tuple[bool, State]:
        """Run an action, and return the success flag and the new PDSketch state. This function will also store the restore function corresponding to the action.

        Args:
            action_index: the action index.
            action_name: the name of the action (the controller).
            action_args: the arguments to the controller.
            verbose: whether to print verbose information.

        Returns:
            a tuple of (success, new PDSketch state).
        """

        assert action_index == self._last_action_index + 1, f'Action index {action_index} is not the next action index.'
        if action_name not in self.controllers:
            raise ValueError(f'No controller registered for {action_name}.')

        if verbose:
            jacinle.log_function.print('Running action', action_index, action_name)

        state = self.pd_states[action_index - 1]

        # TODO (Jiayuan@2023/08/16): support wrap_rv, or handle it at the definition of the controller.
        succ, pd_state, restore_function = self.controllers[action_name](state, *action_args, wrap_rv=False)

        if not succ:
            self.restore(action_index - 1, verbose=verbose)
            return False, None

        self.pd_states[action_index] = pd_state
        self.restore_functions[action_index] = restore_function
        self._last_action_index = action_index
        return succ, pd_state

    def restore(self, target_action_index: int, verbose: bool = False) -> bool:
        """Restore the simulator to a given action index.

        Args:
            target_action_index: the target action index.
            verbose: whether to print verbose information.

        Returns:
            whether the restore operation is successful.
        """
        if target_action_index == self._last_action_index:
            return True

        for action_index in range(self._last_action_index, target_action_index, -1):
            if verbose:
                jacinle.log_function.print(f'Restoring action {action_index}')
            if action_index not in self.restore_functions:
                raise ValueError(f'No restore function for action {action_index}.')
            if self.restore_functions[action_index] is None:
                raise ValueError(f'Empty restore function for action {action_index}.')
            self.restore_functions[action_index]()
            del self.restore_functions[action_index]  # safe to delete the restore function after it is used.

        self._last_action_index = target_action_index
        return True

    @contextlib.contextmanager
    def restore_context(self, verbose: bool = False) -> Iterable[int]:
        """A context manager for restoring the simulator to the current action index."""
        action_index = self._last_action_index
        yield self._last_action_index
        self.restore(action_index, verbose=verbose)

    @property
    def last_action_index(self):
        """The last action index."""
        return self._last_action_index


class PDSketchExecutionInterface(object):
    """The base class for interaction with the actual executor."""

    def __init__(self, executor: PDSketchExecutor):
        """Initialize the executor interface.

        Args:
            executor: the :class:`~concepts.dm.pdsketch.executor.PDSketchExecutor` instance for grounding action parameters.
        """
        self._executor = executor
        self._last_action_index = -1
        self.controllers = dict()
        self.pd_states = dict()

    controllers: Dict[str, Callable]
    """Registered controllers."""

    pd_states: Dict[int, State]
    """Mappings from action indices to PDSketch states. -1 is the initial state."""

    @property
    def last_action_index(self):
        """The last action index."""
        return self._last_action_index

    def register_action_controller(self, name: str, controller_function: Callable):
        """Register a controller function for a controller name.

        Args:
            name: the name of the controller.
            controller_function: the controller function. It should take a PDSketch state as the first argument, together with other action arguments,
                and return a tuple of (success, new PDSketch state).
        """
        if not isinstance(controller_function, PythonFunctionRef):
            controller_function = PythonFunctionRef(controller_function, executor=self._executor)

        self.controllers[name] = controller_function.set_executor(self._executor)

    def set_init_state(self, state: State):
        """Set the initial PDSketch state of the simulator."""
        self.pd_states[-1] = state

    def get_pd_state(self, action_index: int) -> State:
        """Get the PDSketch state at a given action index. -1 is the initial state.

        Args:
            action_index: the action index.
        """
        return self.pd_states[action_index]

    def get_latest_pd_state(self) -> State:
        """Get the latest PDSketch state."""
        return self.pd_states[self._last_action_index]

    def run_operator_applier(self, action_index: int, action: OperatorApplier) -> Tuple[bool, State]:
        """Execute an action, and return the success flag and the new PDSketch state. This function will not store the restore function corresponding to the action.

        Args:
            action_index: the action index.
            action: the action to execute.

        Returns:
            a tuple of (success, new PDSketch state).
        """

        action_name = action.operator.controller.name
        action_args = self._executor.get_controller_args(action, self.get_latest_pd_state())
        return self.run(action_index, action_name, action_args)

    def run(self, action_index: int, action_name: str, action_args: Tuple[Any, ...], verbose: bool = False) -> Tuple[bool, State]:
        """Run an action, and return the success flag and the new PDSketch state. This function will also store the restore function corresponding to the action.

        Args:
            action_index: the action index.
            action_name: the name of the action (the controller).
            action_args: the arguments to the controller.
            verbose: whether to print verbose information.

        Returns:
            a tuple of (success, new PDSketch state).
        """

        assert action_index == self._last_action_index + 1, f'Action index {action_index} is not the next action index.'
        if action_name not in self.controllers:
            raise ValueError(f'No controller registered for {action_name}.')

        if verbose:
            jacinle.log_function.print('Executing action', action_index, action_name)

        state = self.pd_states[action_index - 1]

        # TODO (Jiayuan@2023/08/16): support wrap_rv, or handle it at the definition of the controller.
        succ, pd_state = self.controllers[action_name](state, *action_args, wrap_rv=False)

        self.pd_states[action_index] = pd_state
        self._last_action_index = action_index
        return succ, pd_state
