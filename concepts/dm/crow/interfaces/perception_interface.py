#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : perception_interface.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/04/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from dataclasses import dataclass
from typing import Any, Optional, Union, Sequence, Tuple, List, Dict

import numpy as np

import concepts.dsl.expression as E
from concepts.dsl.dsl_types import ObjectConstant
from concepts.dsl.expression import ObjectOrValueOutputExpression
from concepts.dsl.constraint import ConstraintSatisfactionProblem
from concepts.dsl.tensor_value import TensorValue
from concepts.dm.crow.crow_domain import CrowState

__all__ = ['CrowPerceptionInterface', 'CrowPerceptionResult', 'ObjectTrackingRequest', 'CrowGlobalMemory', 'CrowObjectMemoryItem']


@dataclass
class CrowGlobalMemory(object):
    """The global memory of the perception system."""

    partial_scene_pcd: Any
    """The partial point cloud of the scene."""

    partial_scene_mesh: Any
    """The partial mesh of the scene."""

    scene_pcd: Any
    """The point cloud of the scene."""

    scene_mesh: Any
    """The mesh of the scene."""


@dataclass
class CrowObjectMemoryItem(object):
    """A memory item for an object."""

    identifier: int
    """The index of the object in the memory."""

    query: str
    """The query that is used to detect the object."""

    partial_pcd: Any
    """The partial point cloud of the detected object."""

    pcd: Any
    """The completed point cloud of the detected object."""

    mesh: Any
    """The mesh of the detected object."""

    last_updated_frame: int
    """The frame number when the memory item is last updated."""

    last_updated_frame_segmentation: np.ndarray
    """The point cloud segmentation of the detected object."""

    features: Dict[str, Any]
    """Any additional features of the detected object."""


@dataclass
class CrowPerceptionResult(object):
    """The result of a perception query."""

    timestep: int
    """The timestep of the perception result."""

    global_memory: CrowGlobalMemory
    """The global memory of the perception system."""

    object_memory: Sequence[CrowObjectMemoryItem]
    """The object memory items that are detected."""


@dataclass
class ObjectTrackingRequest(object):
    identifier: int
    features: List[str]


class CrowPerceptionInterface(object):
    """The perception interface for PDSketch.

    The perception interface takes the raw sensory data and supports various types of perception queries, including

    - Occupancy point clouds. This is useful for performing collision checking.
    - Identifying of objects given particular queries, such as the name of an object.
    """

    def __init__(self):
        self._tracking_objects = dict()

    def update_simulator(self) -> None:
        """Update the simulator."""
        raise NotImplementedError()

    def get_crow_state(self) -> CrowState:
        """Get the state of the perception interface."""
        raise NotImplementedError()

    def step(self, action: Optional[Any] = None) -> None:
        """Step the perception interface."""
        raise NotImplementedError()

    def get_perception_result(self) -> CrowPerceptionResult:
        """Get the perception result."""
        raise NotImplementedError()

    def mem_query(
        self, expression: ObjectOrValueOutputExpression, state: CrowState, csp: Optional[ConstraintSatisfactionProblem] = None,
        bounded_variables: Optional[Dict[str, Union[TensorValue, ObjectConstant]]] = None,
        state_index: Optional[int] = None
    ) -> Tuple[CrowState, ConstraintSatisfactionProblem, Dict[str, Union[TensorValue, ObjectConstant]]]:
        raise NotImplementedError()

    def detect(self, name: str) -> None:
        """Detect the object with the given name."""
        raise NotImplementedError()

    def register_object_tracking(self, identifier: int) -> None:
        raise NotImplementedError()

    def unregister_object_tracking(self, identifier: int) -> None:
        raise NotImplementedError()

    def register_object_tracking_feature(self, identifier: int, feature: str) -> None:
        raise NotImplementedError()

    def unregister_object_tracking_feature(self, identifier: int, feature: str) -> None:
        raise NotImplementedError()


# class SimpleMemQueryExecutor(object):
#     def __init__(self, perception_interface: CrowPerceptionInterface):
#         self._perception_interface = perception_interface
#
#     def execute(self, expression: ObjectOrValueOutputExpression, state: CrowState, bounded_variables: Dict[str, Union[TensorValue, ObjectConstant]]) -> CrowState:
#         if isinstance(expression, E.FindAllExpression):
#             return self._execute_find_all(expression, state, bounded_variables)

