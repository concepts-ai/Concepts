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
from concepts.simulator.mplib.client import MPLibClient
from concepts.utils.typing_utils import Open3DPointCloud, Open3DTriangleMesh, Trimesh, Vec3f, Vec4f


class PyBulletPlanningWorldInterface(PlanningWorldInterface):
    def __init__(self, client: BulletClient, mplib_client: Optional[MPLibClient] = None):
        self._client = client
        self._mplib_client = mplib_client
        self._ignored_collision_pairs = list()

    @property
    def client(self) -> BulletClient:
        return self._client

    @property
    def mplib_client(self) -> Optional[MPLibClient]:
        return self._mplib_client

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
        identifier = self._client.world.get_body_index(identifier)
        return self._client.world.get_body_state_by_id(identifier).get_transformation()

    def _set_object_pose(self, identifier: Union[str, int], pose: Tuple[Vec3f, Vec4f]):
        identifier = self._client.world.get_body_index(identifier)
        self._client.world.set_body_state2_by_id(identifier, pose[0], pose[1])
        name = self._client.world.get_body_name(identifier)
        if self._mplib_client is not None:
            self._mplib_client.set_object_pose(name, pose[0], pose[1])

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
        return self._client.world.save_world()

    def _restore_world(self, world: WorldSaverBuiltin):
        world.restore()
        if self._mplib_client is not None:
            self._mplib_client.sync_object_states(self._client)

    def _checkpoint_world(self) -> Iterator[Any]:
        world = self._save_world()
        # with self._client.disable_rendering(suppress_stdout=False):
        yield
        self._restore_world(world)

    def _get_contact_points_mplib(
        self, a: Optional[Union[str, int]] = None, b: Optional[Union[str, int]] = None,
        max_distance: Optional[float] = None, ignored_collision_bodies: Optional[List[Union[str, int]]] = None
    ) -> List[GeometricContactInfo]:
        assert a is not None
        a = self._client.world.get_body_name(a)
        if b is not None:
            b = self._client.world.get_body_name(b)
            contacts = self._mplib_client.check_for_general_pair_collision(a, b)
        else:
            contacts = self._mplib_client.check_for_general_collision(a)

        # print('_get_contact_points_mplib raw_contacts', contacts)

        geometric_contacts = list()
        for x in contacts:
            body_a = self._client.world.get_body_index(x.object_name1)
            body_b = self._client.world.get_body_index(x.object_name2, default=None)
            link_a = self.client.world.get_link_index_with_body(body_a, x.link_name1)
            link_b = self.client.world.get_link_index_with_body(body_b, x.link_name2) if body_b is not None else None
            # print('geometric_contacts', x, body_a, link_a, body_b, link_b, x.max_penetration, 'ignored_collision_bodies', ignored_collision_bodies)
            if ignored_collision_bodies is not None:
                if body_a in ignored_collision_bodies or body_b in ignored_collision_bodies:
                    continue
            if (body_a, link_a, body_b, link_b) in self._ignored_collision_pairs or (body_b, link_b, body_a, link_a) in self._ignored_collision_pairs:
                continue

            if -x.max_penetration > max_distance:
                continue

            geometric_contacts.append(GeometricContactInfo(
                body_a=body_a, body_b=body_b, link_a=link_a, link_b=link_b,
                position_on_a=None, position_on_b=None, contact_normal_on_a=None, contact_normal_on_b=None,
                contact_distance=-x.max_penetration
            ))
        return geometric_contacts

    def _get_contact_points(
        self,
        a: Optional[Union[str, int]] = None,
        b: Optional[Union[str, int]] = None,
        max_distance: Optional[float] = None,
        ignored_collision_bodies: Optional[List[Union[str, int]]] = None
    ) -> List[GeometricContactInfo]:
        if self._mplib_client is not None:
            return self._get_contact_points_mplib(a, b, max_distance, ignored_collision_bodies)

        contacts = self._client.world.get_contact(a, b, max_distance=max_distance, update=True)
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

