#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : pybullet_planning_world_interface.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/23/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import numpy as np
from typing import Any, Iterator, Optional, Union, Tuple, List

from concepts.dm.crowhat.world.planning_world_interface import GeometricContactInfo, PlanningWorldInterface
from concepts.math.cad.mesh_utils import open3d_mesh_to_trimesh
from concepts.simulator.pybullet.client import BulletClient
from concepts.simulator.pybullet.world import WorldSaverBuiltin
from concepts.utils.typing_utils import Open3DPointCloud, Open3DTriangleMesh, Trimesh, Vec3f, Vec4f


class PyBulletPlanningWorldInterface(PlanningWorldInterface):
    def __init__(self, client: BulletClient):
        self._client = client
        self._ignored_collision_pairs = list()

    @property
    def client(self) -> BulletClient:
        return self._client

    def add_ignore_collision_pair_by_id(self, body_a, link_a, body_b, link_b):
        self._ignored_collision_pairs.append((body_a, link_a, body_b, link_b))

    def add_ignore_collision_pair_by_name(self, link_a, link_b):
        body_a, link_a = self._client.world.get_link_index(link_a)
        body_b, link_b = self._client.world.get_link_index(link_b)
        self.add_ignore_collision_pair_by_id(body_a, link_a, body_b, link_b)

    def _get_objects(self) -> List[Any]:
        return list(self._client.world.body_names.int_to_string.keys())

    def _get_object_name(self, identifier: Union[str, int]) -> str:
        return self._client.world.body_names.as_string(identifier)

    def _get_object_pose(self, identifier: Union[str, int]) -> Tuple[Vec3f, Vec4f]:
        return self._client.world.get_body_state_by_id(identifier).get_transformation()

    def _set_object_pose(self, identifier: Union[str, int], pose: Tuple[Vec3f, Vec4f]):
        self._client.world.set_body_state2_by_id(identifier, pose[0], pose[1])

    def _get_link_pose(self, body_id: int, link_id: int) -> Tuple[Vec3f, Vec4f]:
        return self._client.world.get_link_state_by_id(body_id, link_id, fk=True).get_transformation()

    def _get_object_point_cloud(self, identifier: Union[str, int], **kwargs) -> Open3DPointCloud:
        kwargs.setdefault('zero_center', False)
        return self._client.world.get_pointcloud(identifier, **kwargs)

    def _get_object_mesh(self, identifier: Union[str, int], mode: str = 'open3d', **kwargs) -> Union[Open3DTriangleMesh, Trimesh]:
        if mode == 'open3d':
            kwargs.setdefault('zero_center', False)
            return self._client.world.get_mesh(identifier, **kwargs)
        elif mode == 'trimesh':
            mesh = self._client.world.get_mesh(identifier, **kwargs)
            return open3d_mesh_to_trimesh(mesh)
        else:
            raise ValueError(f'Invalid mode: {mode}')

    def _save_world(self) -> WorldSaverBuiltin:
        return self._client.world.save_world_builtin()

    def _restore_world(self, world: WorldSaverBuiltin):
        world.restore()

    def _checkpoint_world(self) -> Iterator[Any]:
        world = self._save_world()
        # with self._client.disable_rendering(suppress_stdout=False):
        yield
        self._restore_world(world)

    def _get_contact_points(
        self,
        a: Optional[Union[str, int]] = None,
        b: Optional[Union[str, int]] = None,
        ignored_collision_bodies: Optional[List[Union[str, int]]] = None
    ) -> List[GeometricContactInfo]:
        contacts = self._client.world.get_contact(a, b, update=True)
        geometric_contacts = list()
        for x in contacts:
            if ignored_collision_bodies is not None:
                if x.body_a in ignored_collision_bodies or x.body_b in ignored_collision_bodies:
                    continue
            if (x.body_a, x.link_a, x.body_b, x.link_b) in self._ignored_collision_pairs or (x.body_b, x.link_b, x.body_a, x.link_a) in self._ignored_collision_pairs:
                continue
            geometric_contacts.append(GeometricContactInfo(
                body_a=x.body_a, body_b=x.body_b, link_a=x.link_a, link_b=x.link_b,
                position_on_a=x.position_on_a, position_on_b=x.position_on_b,
                contact_normal_on_a=tuple(-x for x in x.contact_normal_on_b), contact_normal_on_b=x.contact_normal_on_b,
                contact_distance=x.contact_distance
            ))
        return geometric_contacts

    def get_single_contact_normal(self, object_id: int, support_object_id: int, deviation_tol: float = 0.05, return_center: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        contacts = self.get_contact_points(object_id, support_object_id)
        if len(contacts) == 0:
            self._client.p.stepSimulation()
            contacts = self.get_contact_points(object_id, support_object_id)
        return self._compute_single_contact_normal_from_contacts(contacts, object_id, support_object_id, deviation_tol=deviation_tol, return_center=return_center)

