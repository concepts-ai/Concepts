#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : cogman.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/05/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os.path as osp
import numpy as np
from typing import Any, Optional, Union, Tuple, Sequence, List, Dict, Callable
from dataclasses import dataclass

import jacinle
import concepts.dm.crow as crow
from concepts.benchmark.manip_tabletop.pybullet_tabletop_base.pybullet_tabletop import TableTopEnv


class PybulletSimulationObjectNameMapping(object):
    def __init__(self, obj_identifiers: List[str], obj_ids: List[int], labels: Dict[str, Sequence[str]]):
        self.obj_identifiers = obj_identifiers
        self.obj_ids = obj_ids
        self.name2id = {name: obj_id for name, obj_id in zip(obj_identifiers, obj_ids)}
        self.id2name = {obj_id: name for name, obj_id in zip(obj_identifiers, obj_ids)}
        self.labels = labels

    obj_identifiers: List[str]
    obj_ids: List[int]
    name2id: Dict[str, int]
    id2name: Dict[int, str]
    labels = Dict[str, Sequence[str]]


@dataclass
class OBMVisionPatch(object):
    image: np.ndarray
    depth: np.ndarray
    intrinsics: Optional[np.ndarray] = None
    extrinsics: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None

    def __str__(self):
        return f'OBMVisionPatch(image={self.image.shape}, depth={self.depth.shape}, mask={self.mask.shape if self.mask is not None else None})'

    def __repr__(self):
        return str(self)


class CognitionManager(object):
    def __init__(self, domain):
        self.domain = domain

        # Scene objects related fields.
        self.object_identifiers = list()
        self.object_pybullet_ids = list()
        self.object_additional_labels = dict()
        self.object_vision_patch = dict()

        # Registered state getter functions
        self.state_getter_functions = list()

        # crow-related modules
        self.executor = None
        self.simulation_env = None
        self.sci = None
        self.pci = None
        self.vision_pipeline = None

    object_identifiers: List[str]
    object_pybullet_ids: List[Optional[int]]
    object_additional_labels: Dict[str, Tuple[str, ...]]
    object_vision_patch: Dict[str, OBMVisionPatch]
    state_getter_functions: List[Callable[[crow.CrowState], None]]

    def clear_objects(self):
        self.object_identifiers.clear()
        self.object_pybullet_ids.clear()
        self.object_additional_labels.clear()
        self.object_vision_patch.clear()

    def add_object(self, identifier: str, pybullet_id: Optional[int] = None, labels: Union[Tuple[str, ...], List[str]] = tuple(), vision_patch: Optional[OBMVisionPatch] = None):
        self.object_identifiers.append(identifier)
        self.object_pybullet_ids.append(pybullet_id)
        if len(labels) > 0:
            self.object_additional_labels[identifier] = labels
        if vision_patch is not None:
            self.object_vision_patch[identifier] = vision_patch

    # Simulation related functions
    def is_simulation_available(self) -> bool:
        return self.simulation_env is not None

    def set_simulation_env(self, env: TableTopEnv):
        self.simulation_env = env

    def get_simulation_env(self) -> TableTopEnv:
        assert self.simulation_env is not None
        return self.simulation_env

    def get_pybullet_name_mapping(self) -> PybulletSimulationObjectNameMapping:
        return PybulletSimulationObjectNameMapping(self.object_identifiers, self.object_pybullet_ids, self.object_additional_labels)

    def find_object_by_name(self, name: str, unique: bool = False) -> str | List[str]:
        name = name.replace(' ', '_')
        all_elements = list()
        for identifier in self.object_identifiers:
            if '_' not in identifier:
                continue
            l, r = identifier.split('_', maxsplit=2)
            if r == name:
                all_elements.append(identifier)
        if unique:
            assert len(all_elements) == 1
            return all_elements[0]
        return all_elements

    # Physical controller interface
    def is_physical_interface_available(self) -> bool:
        return self.pci is not None

    def get_robot_state(self) -> Any:
        raise NotImplementedError()

    def get_captures(self) -> Dict[str, Dict[str, np.ndarray]]:
        raise NotImplementedError()

    # State getter functions
    def register_state_getter_function(self, func: Callable[[crow.CrowState], None]):
        self.state_getter_functions.append(func)

    def get_state(self) -> crow.CrowState:
        state = crow.CrowState.make_empty_state(self.domain, {name: 'Object' for name in self.object_identifiers})

        for i, name in enumerate(self.object_identifiers):
            if name in self.object_additional_labels:
                for label in self.object_additional_labels[name]:
                    state.fast_set_value(label, [name], 1)

        for func in self.state_getter_functions:
            func(state)

        return state

    def plan(self, goal_string: str, state: Optional[crow.CrowState] = None, **kwargs):
        if state is None:
            state = self.get_state()
        results = crow.crow_regression(
            self.executor,
            problem=crow.CrowProblem.from_state_and_goal(self.domain, state, goal_string),
            return_results=True,
            simulation_interface=self.sci,
            algo='priority_tree_v1',
            # algo='iddfs_v1', min_search_depth=5, max_search_depth=5,
            **kwargs,
        )
        return results

    def execute_plan(self, plan: Sequence[crow.CrowControllerApplier]):
        for action in plan:
            print(f'Executing action: {action}')
            self.pci.step(action)


G_SKILL_LIB_PATH = osp.dirname(__file__)


def set_skill_lib_path(path):
    global G_SKILL_LIB_PATH
    G_SKILL_LIB_PATH = path


def load_skill_lib(cogman: CognitionManager, lib_name: str, python_only: bool = False):
    if not python_only:
        cogman.domain.incremental_define_file(f'{lib_name}.cdl')

    lib = jacinle.load_module_filename(osp.join(G_SKILL_LIB_PATH, f'{lib_name}.py'))
    if hasattr(lib, 'register_state_getter_functions'):
        print(f'Loading lib {lib_name}::register_state_getter_functions')
        lib.register_state_getter_functions(cogman)
    if hasattr(lib, 'register_function_implementations'):
        print(f'Loading lib {lib_name}::register_function_implementations')
        lib.register_function_implementations(cogman, cogman.executor)

    if cogman.is_simulation_available():
        if hasattr(lib, 'register_simulation_controllers'):
            print(f'Loading lib {lib_name}::register_simulation_controllers')
            lib.register_simulation_controllers(cogman, cogman.sci)
    if cogman.is_physical_interface_available():
        if hasattr(lib, 'register_physical_controllers'):
            print(f'Loading lib {lib_name}::register_physical_controllers')
            lib.register_physical_controllers(cogman, cogman.pci)
