#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : planning_world_interface.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/28/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import contextlib
from typing import Any, Optional, Union, Iterator, Tuple, List, NamedTuple

import numpy as np

from concepts.utils.typing_utils import Open3DPointCloud, Open3DTriangleMesh, Trimesh, Vec3f, Vec4f


class GeometricContactInfo(NamedTuple):
    body_a: Union[str, int]
    body_b: Union[str, int]
    link_a: Union[str, int]
    link_b: Union[str, int]
    position_on_a: Vec3f
    position_on_b: Vec3f
    contact_normal_on_a: Vec3f
    contact_normal_on_b: Vec3f
    contact_distance: float = 0


class AttachmentInfo(NamedTuple):
    body_a: Union[str, int]
    body_b: Union[str, int]
    link_a: Union[str, int]
    link_b: Union[str, int]
    a_to_b: Tuple[Vec3f, Vec4f]


class PlanningWorldInterface(object):
    def get_objects(self) -> List[Any]:
        """Get a list of objects in the world.

        Returns:
            a list of objects in the world.
        """
        return self._get_objects()

    def _get_objects(self) -> List[Any]:
        raise NotImplementedError()

    def get_object_name(self, identifier: Union[str, int]) -> str:
        """Get the name of the object with the given identifier.

        Args:
            identifier: the identifier of the object.

        Returns:
            the name of the object.
        """
        return self._get_object_name(identifier)

    def _get_object_name(self, identifier: Union[str, int]) -> str:
        raise NotImplementedError()

    def get_object_pose(self, identifier: Union[str, int]) -> Tuple[Vec3f, Vec4f]:
        """Get the pose of the object with the given identifier.

        Args:
            identifier: the identifier of the object.

        Returns:
            a tuple of the position and quaternion of the object.
        """
        return self._get_object_pose(identifier)

    def _get_object_pose(self, identifier: Union[str, int]) -> Tuple[Vec3f, Vec4f]:
        raise NotImplementedError()

    def set_object_pose(self, identifier: Union[str, int], pose: Tuple[Vec3f, Vec4f]):
        """Set the pose of the object with the given identifier.

        Args:
            identifier: the identifier of the object.
            pose: the new pose of the object.
        """
        self._set_object_pose(identifier, pose)

    def _set_object_pose(self, identifier: Union[str, int], pose: Tuple[Vec3f, Vec4f]):
        raise NotImplementedError()

    def get_link_pose(self, body_id: Union[str, int], link_id: Union[str, int]) -> Tuple[Vec3f, Vec4f]:
        """Get the pose of the link with the given body and link identifiers.

        Args:
            body_id: the identifier of the body.
            link_id: the identifier of the link.

        Returns:
            a tuple of the position and quaternion of the link.
        """
        return self._get_link_pose(body_id, link_id)

    def _get_link_pose(self, body_id: Union[str, int], link_id: Union[str, int]) -> Tuple[Vec3f, Vec4f]:
        raise NotImplementedError()

    def add_attachment(self, a: Union[str, int], a_link: Union[str, int], b: Union[str, int], b_link: Union[str, int], a_to_b: Optional[Tuple[Vec3f, Vec4f]] = None) -> Any:
        """Set the attachment between object a and object b. This is an optional functionality that can be implemented in subclasses

        Args:
            a: the identifier of the first object.
            a_link: the link index of the first object.
            b: the identifier of the second object.
            b_link: the link index of the second object.
            pose: the pose of the attachment. If None, it will use the current pose of the objects.

        Returns:
            an identifier of the attachment.
        """
        return self._add_attachment(a, a_link, b, b_link, a_to_b)

    def _add_attachment(self, a: Union[str, int], a_link: Union[str, int], b: Union[str, int], b_link: Union[str, int], a_to_b: Optional[Tuple[Vec3f, Vec4f]] = None) -> Any:
        raise NotImplementedError()

    def remove_attachment(self, a: Union[str, int], a_link: Union[str, int], b: Union[str, int], b_link: Union[str, int]):
        """Remove the attachment between object a and object b. This is an optional functionality that can be implemented in subclasses.

        Args:
            a: the identifier of the first object.
            a_link: the link index of the first object.
            b: the identifier of the second object.
            b_link: the link index of the second object.
        """
        self._remove_attachment(a, a_link, b, b_link)

    def _remove_attachment(self, a: Union[str, int], a_link: Union[str, int], b: Union[str, int], b_link: Union[str, int]):
        raise NotImplementedError()

    def get_object_mesh(self, identifier: Union[str, int], mode: str = 'open3d', **kwargs) -> Union[Open3DTriangleMesh, Trimesh]:
        """Get the mesh of the object with the given identifier.

        Args:
            identifier: the identifier of the object.
            mode: the mode of the mesh. Default is 'open3d'.

        Returns:
            the mesh of the object.
        """
        return self._get_object_mesh(identifier, mode=mode, **kwargs)

    def _get_object_mesh(self, identifier: Union[str, int], mode: str = 'open3d', **kwargs) -> Open3DTriangleMesh:
        raise NotImplementedError()

    def get_object_point_cloud(self, identifier: Union[str, int], **kwargs) -> Open3DPointCloud:
        """Get the point cloud of the object with the given identifier.

        Args:
            identifier: the identifier of the object.

        Returns:
            the point cloud of the object.
        """
        return self._get_object_point_cloud(identifier, **kwargs)

    def _get_object_point_cloud(self, identifier: Union[str, int], **kwargs) -> Open3DPointCloud:
        raise NotImplementedError()

    def get_contact_points(self, a: Optional[Union[str, int]] = None, b: Optional[Union[str, int]] = None, max_distance: Optional[float] = None, ignored_collision_bodies: Optional[List[Union[str, int]]] = None) -> List[GeometricContactInfo]:
        """Get the contact points of the object between a and b, which are the identifiers of two objects. If either a or b is None, it will return the contact
        points of the object with the given identifier. When both a and b are None, it will return all the contact points in the world.

        Args:
            a: the identifier of the first object.
            b: the identifier of the second object.
            ignored_collision_bodies: a list of identifiers of the bodies to ignore.

        Returns:
            a list of contact points.
        """
        return self._get_contact_points(a, b, max_distance=max_distance, ignored_collision_bodies=ignored_collision_bodies)

    def _get_contact_points(self, a: Optional[Union[str, int]] = None, b: Optional[Union[str, int]] = None, max_distance: Optional[float] = None, ignored_collision_bodies: Optional[List[Union[str, int]]] = None) -> List[GeometricContactInfo]:
        raise NotImplementedError()

    def check_collision(self, a: Optional[Union[str, int]] = None, b: Optional[Union[str, int]] = None, ignored_collision_bodies: Optional[List[Union[str, int]]] = None, max_distance: Optional[float] = None) -> bool:
        """Check if there is a collision between the object with the given identifiers.

        Args:
            a: the identifier of the first object.
            b: the identifier of the second object.
            ignored_collision_bodies: a list of identifiers of the bodies to ignore.

        Returns:
            True if there is a collision, False otherwise.
        """
        return len(self.get_contact_points(a, b, ignored_collision_bodies=ignored_collision_bodies, max_distance=max_distance)) > 0

    def check_collision_with_other_objects(
        self,
        object_id: Union[str, int],
        ignore_self_collision: bool = True,
        ignored_collision_bodies: Optional[List[Union[str, int]]] = None,
        max_distance: Optional[float] = None,
        return_list: bool = False, verbose: bool = False
    ) -> Union[bool, List[Union[str, int]]]:
        """Check if there is a collision between the object with the given identifier and other objects.

        Args:
            object_id: the identifier of the object.
            ignore_self_collision: whether to ignore the collision between the object and itself.
            ignored_collision_bodies: a list of identifiers of the bodies to ignore.
            max_distance: the maximum distance to consider a collision.
            return_list: whether to return the list of identifiers of the colliding objects.

        Returns:
            True if there is a collision, False otherwise. If return_list is True, it will return the list of identifiers of the colliding objects.
        """
        # print(f'check_collision_with_other_objects a={object_id}, max_distance={max_distance}')
        contacts = self.get_contact_points(a=object_id, max_distance=max_distance)
        if ignore_self_collision:
            contacts = [c for c in contacts if c.body_b != object_id]
        if ignored_collision_bodies is not None:
            contacts = [c for c in contacts if c.body_b not in ignored_collision_bodies]
        if return_list:
            return [c.body_b for c in contacts]

        if verbose:
            for c in contacts:
                print(f'Collision between {c.body_a} / {c.link_a} and {c.body_b} / {c.link_b}.')
        if False:
            import jacinle
            from concepts.simulator.pybullet.client import BulletClient
            client: BulletClient = self.client
            collisions = list()
            for c in contacts:
                name1 = client.world.get_link_name(c.body_a, c.link_a, trim_body_name=False)
                name2 = client.world.get_link_name(c.body_b, c.link_b, trim_body_name=False)
                collisions.append((name1, name2, c.contact_distance))
            if len(collisions) > 0:
                print(jacinle.tabulate(collisions, headers=['Object 1', 'Object 2', 'Distance']))

        return len(contacts) > 0

    def check_collision_pairs(self, pairs: List[Tuple[Union[str, int], Union[str, int]]], ignored_collision_bodies: Optional[List[Union[str, int]]] = None) -> bool:
        """Check if there is a collision between the pairs of objects.

        Args:
            pairs: a list of pairs of objects.
            ignored_collision_bodies: a list of identifiers of the bodies to ignore.

        Returns:
            True if there is a collision, False otherwise.
        """
        all_contacts = self.get_contact_points(ignored_collision_bodies=ignored_collision_bodies)
        for a, b in pairs:
            if any(c.body_a == a and c.body_b == b for c in all_contacts) or any(c.body_a == b and c.body_b == a for c in all_contacts):
                return True
        return False

    def get_single_contact_normal(self, object_id: Union[str, int], support_object_id: Union[str, int], deviation_tol: float = 0.05, return_center: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        contacts = self.get_contact_points(object_id, support_object_id)
        return self._compute_single_contact_normal_from_contacts(contacts, object_id, support_object_id, deviation_tol=deviation_tol, return_center=return_center)

    def _compute_single_contact_normal_from_contacts(self, contacts: List[GeometricContactInfo], object_id: Union[str, int], support_object_id: Union[str, int], deviation_tol: float = 0.05, return_center: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if len(contacts) == 0:
            raise ValueError(f'No contact between {object_id} and {support_object_id}.')

        contact_normals = np.array([c.contact_normal_on_b for c in contacts])
        contact_normal_avg = np.mean(contact_normals, axis=0)
        contact_normal_avg /= np.linalg.norm(contact_normal_avg)

        deviations = np.abs(1 - contact_normals.dot(contact_normal_avg) / np.linalg.norm(contact_normals, axis=1))
        if np.max(deviations) > deviation_tol:
            raise ValueError(
                f'Contact normals of {object_id} and {support_object_id} are not consistent. This is likely due to multiple contact points.\n'
                f'  Contact normals: {contact_normals}\n  Deviations: {deviations}.'
            )

        if return_center:
            centers = np.array([c.position_on_b for c in contacts])
            center = np.mean(centers, axis=0)
            return center, contact_normal_avg

        return contact_normal_avg

    def save_world(self) -> Any:
        """Save the current world state."""
        return self._save_world()

    def _save_world(self) -> Any:
        raise NotImplementedError()

    def restore_world(self, world: Any):
        """Restore the world state from the given world state."""
        self._restore_world(world)


    def _restore_world(self, world: Any):
        raise NotImplementedError()

    @contextlib.contextmanager
    def checkpoint_world(self) -> Iterator[Any]:
        yield from self._checkpoint_world()

    def _checkpoint_world(self) -> Iterator[Any]:
        x = self.save_world()
        try:
            yield x
        finally:
            self.restore_world(x)
