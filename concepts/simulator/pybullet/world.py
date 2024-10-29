#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : world.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/17/2020
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import time
import inspect
import collections
from typing import Any, Optional, Union, Iterable, Tuple, List, Dict, Callable

import numpy as np
import open3d as o3d
import pybullet as p

from jacinle.utils.printing import indent_text
from concepts.math.rotationlib_xyzw import quat_mul, quat_conjugate, rotate_vector_batch
from concepts.math.frame_utils_xyzw import compose_transformation
from concepts.simulator.pybullet.camera import CameraConfig

__all__ = [
    'BodyState', 'JointState', 'LinkState', 'DebugCameraState',
    'JointInfo', 'ContactInfo', 'ConstraintInfo',
    'BulletSaver', 'GroupSaver', 'BodyStateSaver', 'JointStateSaver', 'BodyFullStateSaver', 'WorldSaver',
    'WorldSaverBuiltin', 'BulletWorld'
]


class _NameToIdentifier(object):
    def __init__(self):
        self.string_to_int = dict()
        self.int_to_string = dict()

    def add_int_to_string(self, i, s):
        self.int_to_string[i] = s
        self.string_to_int[s] = i

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.string_to_int[item]
        return self.int_to_string[item]

    def as_string(self, item: Union[str, int]) -> str:
        if isinstance(item, str):
            return item
        return self.int_to_string[item]

    def as_identifier(self, item: Union[str, int, Tuple[int, int]]) -> Union[int, Tuple[int, int]]:
        if isinstance(item, str):
            return self.string_to_int[item]
        return item

    def __len__(self):
        return len(self.int_to_string)

    def __iter__(self):
        yield from self.int_to_string.items()


GlobalIdentifier = collections.namedtuple('GlobalIdentifier', ['type', 'body_id', 'joint_or_link_id'])
JointIdentifier = collections.namedtuple('JointIdentifier', ['body_id', 'joint_id'])
LinkIdentifier = collections.namedtuple('LinkIdentifier', ['body_id', 'link_id'])


class BodyState(collections.namedtuple('_BodyState', ['position', 'orientation', 'linear_velocity', 'angular_velocity'])):
    @property
    def pos(self):
        return self.position

    @property
    def quat(self):
        return self.orientation

    @property
    def quat_xyzw(self):
        return self.orientation

    @property
    def quat_wxyz(self):
        return (self.orientation[3], self.orientation[0], self.orientation[1], self.orientation[2])

    def get_transformation(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.position, self.orientation

    def get_7dpose(self) -> np.ndarray:
        return np.concatenate([self.position, self.orientation])


class JointState(collections.namedtuple('_JointState', ['position', 'velocity'])):
    pass


class LinkState(collections.namedtuple('_LinkState', ['position', 'orientation', 'linear_velocity', 'angular_velocity'])):
    @property
    def pos(self):
        return self.position

    @property
    def quat_xyzw(self):
        return self.orientation

    @property
    def quat_wxyz(self):
        return (self.orientation[3], self.orientation[0], self.orientation[1], self.orientation[2])

    def get_transformation(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(self.position), np.array(self.orientation)

    def get_7dpose(self) -> np.ndarray:
        return np.concatenate([self.position, self.orientation])


class DebugCameraState(collections.namedtuple('_DebugCameraState', [
    'width',
    'height',
    'viewMatrix',
    'projectionMatrix',
    'cameraUp',
    'cameraForward',
    'horizontal',
    'vertical',
    'yaw',
    'pitch',
    'distance',
    'target',
])):

    @property
    def dist(self) -> float:
        return self.distance


class JointInfo(collections.namedtuple('_JointInfo', [
    'joint_index',
    'joint_name',
    'joint_type',
    'qindex',
    'uindex',
    'flags',
    'joint_damping',
    'joint_friction',
    'joint_lower_limit',
    'joint_upper_limit',
    'joint_max_force',
    'joint_max_velocity',
    'link_name',
    'joint_axis',
    'parent_frame_pos',
    'parent_frame_orn',
    'parent_index',
])):
    pass


class CollisionShapeData(collections.namedtuple('_CollisionShapeInfo', [
    'object_unique_id',
    'link_index',
    'shape_type',
    'dimensions',
    'filename',
    'local_pos',
    'local_orn',
    'world_pos',  # the position relative to the world frame
    'world_orn',  # the orientation relative to the world frame
])):
    pass


class VisualShapeData(collections.namedtuple('_VisualShapeInfo', [
    'object_unique_id',
    'link_index',
    'shape_type',
    'dimensions',
    'filename',
    'local_pos',
    'local_orn',
    'rgba_color',
    'texture_unique_id',
    'world_pos',  # the position relative to the world frame
    'world_orn',  # the orientation relative to the world frame
])):
    pass


class ContactInfo(collections.namedtuple('_ContactInfo', [
    'world',
    'contact_flag',
    'body_a',
    'body_b',
    'link_a',
    'link_b',
    'position_on_a',
    'position_on_b',
    'contact_normal_on_b',
    'contact_distance',
    'contact_normal_force',
    'lateral_friction_1',
    'lateral_friction_dir1',
    'lateral_friction_2',
    'lateral_friction_dir2',
])):
    @property
    def body_a_name(self):
        return self.world.body_names[self.body_a]

    @property
    def link_a_name(self):
        return self.world.link_names[self.body_a, self.link_a]

    @property
    def body_b_name(self):
        return self.world.body_names[self.body_b]

    @property
    def link_b_name(self):
        return self.world.link_names[self.body_b, self.link_b]

    @property
    def a_name(self):
        if self.link_a != -1:
            return '@link/' + self.link_a_name
        else:
            return '@body/' + self.body_a_name

    @property
    def b_name(self):
        if self.link_b != -1:
            return '@link/' + self.link_b_name
        else:
            return '@body/' + self.body_b_name

    def hash(self):
        return (self.body_a, self.body_b, self.link_a, self.link_b)


class ConstraintInfo(collections.namedtuple('_ConstraintInfo', [
    'parent_body',
    'parent_joint',
    'child_body',
    'child_link',
    'constraint_type',
    'joint_axis',
    'parent_frame_pos',
    'child_frame_pos',
    'parent_frame_orn',
    'child_frame_orn',
    'max_force',
    'gear_ratio',
    'gear_aux_link',
    'relative_position_target',
    'erp'
])):
    pass


class BulletSaver(object):
    def __init__(self, client_id: int, world: Optional['BulletWorld'] = None):
        self.client_id = client_id
        self._world = world

    def save(self):
        pass

    def restore(self):
        raise NotImplementedError()

    def reset_client_id(self, client_id: int, world: Optional['BulletWorld'] = None):
        self.client_id = client_id
        self._world = world

    def reset_world(self, world: 'BulletWorld'):
        self._world = world

    @property
    def world(self) -> 'BulletWorld':
        if self._world is None:
            raise RuntimeError('The world object is not set.')
        return self._world

    # Before pickling, delete self.world to avoid pickling the entire world object
    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != '_world'}

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __enter__(self):
        self.save()
        return self

    def __exit__(self, type, value, traceback):
        self.restore()


class GroupSaver(BulletSaver):
    def __init__(self, client_id: int, savers: Iterable[BulletSaver]):
        super().__init__(client_id)
        savers = tuple(savers)
        assert len(savers) > 0, 'savers should not be empty'
        self.savers = savers

    def save(self):
        for saver in self.savers:
            saver.save()

    def restore(self):
        for saver in self.savers:
            saver.restore()

    def reset_client_id(self, client_id: int):
        super().reset_client_id(client_id)
        for saver in self.savers:
            saver.reset_client_id(client_id)

    def __str__(self):
        return 'GroupSaver({}){{\n'.format(len(self.savers)) + '\n'.join(indent_text(str(saver)) for saver in self.savers) + '\n}'


class BodyStateSaver(BulletSaver):
    def __init__(self, world: 'BulletWorld', body_id: int, state: Optional[BodyState] = None, save: bool = True):
        super().__init__(world.client_id)
        self.body_id = body_id
        self.body_name = world.body_names.int_to_string.get(body_id, None)
        self.state = None

        if save:
            self.save(state=state)

    def save(self, state: Optional[BodyState] = None):
        if state is not None:
            self.state = ((state.position, state.orientation), (state.linear_velocity, state.angular_velocity))
        else:
            self.state = (p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.client_id), p.getBaseVelocity(self.body_id, physicsClientId=self.client_id))

    def restore(self):
        p.resetBasePositionAndOrientation(self.body_id, self.state[0][0], self.state[0][1], physicsClientId=self.client_id)
        p.resetBaseVelocity(self.body_id, self.state[1][0], self.state[1][1], physicsClientId=self.client_id)

    def __str__(self):
        return 'BodyState({}, state={})'.format(self.body_name or self.body_id, self.state)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.body_name or self.body_id)


class JointStateSaver(BulletSaver):
    def __init__(self, world: 'BulletWorld', body_id: int, joint_ids: Optional[List[int]] = None, state: Optional[List[JointState]] = None, save: bool = True):
        super().__init__(world.client_id)
        self.body_id = body_id
        self.body_name = world.body_names.int_to_string.get(body_id, None)
        self.joint_ids = joint_ids if joint_ids is not None else world.get_free_joints(body_id)
        self.states = None

        if save:
            self.save(state=state)

    def save(self, state: Optional[List[JointState]] = None):
        if state is not None:
            self.states = [(s.position, s.velocity) for s in state]
        else:
            self.states = [
                p.getJointState(self.body_id, joint_id, physicsClientId=self.client_id)
                for joint_id in self.joint_ids
            ]

    def restore(self):
        for joint_id, state in zip(self.joint_ids, self.states):
            p.resetJointState(self.body_id, joint_id, state[0], state[1], physicsClientId=self.client_id)

    def __str__(self):
        return 'JointState({}, states={})'.format(self.body_name or self.body_id, {joint_id: state for joint_id, state in zip(self.joint_ids, self.states)})

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.body_name or self.body_id)


class BodyFullStateSaver(BulletSaver):
    """Save and restore the full state of a body. That is, including both the state of the base link and all joints."""

    def __init__(self, world: 'BulletWorld', body_id: int, save: bool = True):
        super().__init__(world.client_id)
        self.body_id = body_id
        self.body_name = world.body_names.int_to_string.get(body_id, None)

        self.pose_saver = BodyStateSaver(world, body_id, save=save)
        self.joint_saver = JointStateSaver(world, body_id, save=save)

    def save(self, body_state: Optional[BodyState] = None, joint_states: Optional[List[JointState]] = None):
        self.pose_saver.save(body_state)
        self.joint_saver.save(joint_states)

    def restore(self):
        self.pose_saver.restore()
        self.joint_saver.restore()

    def reset_client_id(self, client_id: int, world: Optional['BulletWorld'] = None):
        super().reset_client_id(client_id, world)
        self.pose_saver.reset_client_id(client_id, world)
        self.joint_saver.reset_client_id(client_id, world)

    def __str__(self):
        return 'BodyFullState({}\n  pose={}\n  joints={})'.format(self.body_name or self.body_id, self.pose_saver.state, {joint_id: state for joint_id, state in zip(self.joint_saver.joint_ids, self.joint_saver.states)})

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.body_name or self.body_id)


class WorldSaver(BulletSaver):
    def __init__(self, world: 'BulletWorld', body_ids: Optional[List[int]] = None, save: bool = True):
        super().__init__(world.client_id)

        self.body_ids = body_ids or list(world.body_names.int_to_string.keys())
        self.body_savers = [BodyFullStateSaver(world, body_id, save=False) for body_id in self.body_ids]
        self.additional_savers = [saver() for saver in world.additional_state_savers.values()]

        if save:
            self.save()

    def save(self):
        for body_saver in self.body_savers:
            body_saver.save()
        for saver in self.additional_savers:
            saver.save()

    def restore(self):
        for body_saver in self.body_savers:
            body_saver.restore()
        for saver in self.additional_savers:
            saver.restore()

    def reset_client_id(self, client_id: int, world: Optional['BulletWorld'] = None):
        super().reset_client_id(client_id)
        for body_saver in self.body_savers:
            body_saver.reset_client_id(client_id, world)
        for saver in self.additional_savers:
            saver.reset_client_id(client_id, world)

    def __str__(self):
        return 'WorldSaver({}){{\n'.format(self.body_ids) + '\n'.join(indent_text(str(saver)) for saver in self.body_savers) + '\n}'

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.body_ids)


class WorldSaverBuiltin(BulletSaver):
    def __init__(self, world: 'BulletWorld', save: bool = True):
        super().__init__(world.client_id)

        self.saved_id = None
        self.additional_savers = [saver() for saver in world.additional_state_savers.values()]
        if save:
            self.save()

    def save(self):
        self.saved_id = p.saveState(physicsClientId=self.client_id)
        for saver in self.additional_savers:
            saver.save()

    def clear(self):
        if self.saved_id is not None:
            p.removeState(self.saved_id, physicsClientId=self.client_id)
            self.saved_id = None

    def restore(self):
        if self.saved_id is None:
            raise RuntimeError('No state has been saved.')
        p.restoreState(self.saved_id, physicsClientId=self.client_id)
        p.removeState(self.saved_id, physicsClientId=self.client_id)
        self.saved_id = None

        for saver in self.additional_savers:
            saver.restore()


def _geometry_cache(type_id: int, geom_name_template: str):
    def wrapper(func):
        sig = inspect.signature(func)

        def wrapped_func(self: 'BulletWorld', *args, **kwargs):
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()
            geom_name = geom_name_template.format(**bound_args.arguments)

            if (type_id, geom_name) in self.cached_geometries:
                return self.cached_geometries[type_id, geom_name]
            else:
                value = func(self, *args, **kwargs)
                self.cached_geometries[(type_id, geom_name)] = value
                return value
        return wrapped_func
    return wrapper


class BulletWorld(object):
    def __init__(self, client_id=None):
        self.client_id = client_id

        self.body_names = _NameToIdentifier()
        self.link_names = _NameToIdentifier()
        self.joint_names = _NameToIdentifier()
        self.global_names = _NameToIdentifier()
        self.body_base_link: Dict[str, str] = dict()
        self.body_groups: Dict[str, List[int]] = dict()
        self.managed_interfaces: Dict[str, Any] = dict()
        self.additional_state_savers: Dict[str, Callable[[], BulletSaver]] = dict()

        self.cached_geometries: Dict[Tuple[int, str], Any] = dict()

        if self.client_id is not None:
            self.refresh_names()

    body_names: _NameToIdentifier
    """The mapping from body name to body index."""

    link_names: _NameToIdentifier
    """The mapping from link name to link index."""

    joint_names: _NameToIdentifier
    """The mapping from joint name to joint index."""

    global_names: _NameToIdentifier
    """The mapping from global name to global identifier."""

    body_base_link: Dict[str, str]
    """The mapping from body name to base link name."""

    body_groups: Dict[str, List[int]]
    """The mapping from group name to body indices."""

    managed_interfaces: Dict[str, Any]
    """The mapping from identifiers to interface objects."""

    additional_state_savers: Dict[str, Callable[[], BulletSaver]]
    """The mapping from identifiers to additional state savers."""

    def set_client_id(self, client_id):
        self.client_id = client_id
        self.refresh_names()

    @property
    def nr_bodies(self):
        return len(self.body_names)

    @property
    def nr_joints(self):
        return len(self.joint_names)

    @property
    def nr_links(self):
        return len(self.link_names)

    def notify_update(self, body_id, body_name=None, group=None):
        body_info = p.getBodyInfo(body_id, physicsClientId=self.client_id)
        base_link_name = body_info[0].decode('utf-8')
        if body_name is None:
            body_name = body_info[1].decode('utf-8')
        self.body_base_link[body_name] = base_link_name

        self.body_names.add_int_to_string(body_id, body_name)
        self.link_names.add_int_to_string(LinkIdentifier(body_id, -1), body_name + '/' + base_link_name)
        self.global_names.add_int_to_string(GlobalIdentifier('body', body_id, -1), body_name)
        self.global_names.add_int_to_string(GlobalIdentifier('link', body_id, -1), body_name + '/' + base_link_name)

        for j in range(p.getNumJoints(body_id, physicsClientId=self.client_id)):
            joint_info = JointInfo(*p.getJointInfo(body_id, j, physicsClientId=self.client_id))
            joint_name = joint_info[1].decode('utf-8')
            link_name = joint_info[12].decode('utf-8')
            self.joint_names.add_int_to_string(JointIdentifier(body_id, j), joint_name)
            self.joint_names.add_int_to_string(JointIdentifier(body_id, j), body_name + '/' + joint_name)
            self.global_names.add_int_to_string(GlobalIdentifier('joint', body_id, j), joint_name)
            self.global_names.add_int_to_string(GlobalIdentifier('joint', body_id, j), body_name + '/' + joint_name)

            self.link_names.add_int_to_string(LinkIdentifier(body_id, j), link_name)
            self.link_names.add_int_to_string(LinkIdentifier(body_id, j), body_name + '/' + link_name)
            self.global_names.add_int_to_string(GlobalIdentifier('link', body_id, j), link_name)
            self.global_names.add_int_to_string(GlobalIdentifier('link', body_id, j), body_name + '/' + link_name)

        if group is not None:
            if group not in self.body_groups:
                self.body_groups[group] = list()
            self.body_groups[group].append(body_id)

    def refresh_names(self):
        self.global_names = _NameToIdentifier()
        self.body_names = _NameToIdentifier()
        self.joint_names = _NameToIdentifier()
        self.link_names = _NameToIdentifier()
        self.body_base_link = dict()

        for i in range(p.getNumBodies(physicsClientId=self.client_id)):
            body_info = p.getBodyInfo(i, physicsClientId=self.client_id)
            base_link_name, body_name = body_info[0].decode('utf-8'), body_info[1].decode('utf-8')
            self.body_base_link[body_name] = base_link_name

            self.body_names.add_int_to_string(i, body_name)
            self.global_names.add_int_to_string(('body', i), body_name)
            self.link_names.add_int_to_string((i, -1), body_name + '/' + base_link_name)
            self.global_names.add_int_to_string(('link', i, -1), body_name + '/' + base_link_name)

        for body_id in range(self.nr_bodies):
            body_name = self.body_names[body_id]
            for j in range(p.getNumJoints(body_id, physicsClientId=self.client_id)):
                joint_info = JointInfo(*p.getJointInfo(body_id, j, physicsClientId=self.client_id))
                joint_name = joint_info[1].decode('utf-8')
                link_name = joint_info[12].decode('utf-8')
                self.joint_names.add_int_to_string((body_id, j), joint_name)
                self.joint_names.add_int_to_string((body_id, j), body_name + '/' + joint_name)
                self.global_names.add_int_to_string(('joint', body_id, j), joint_name)
                self.global_names.add_int_to_string(('joint', body_id, j), body_name + '/' + joint_name)

                self.link_names.add_int_to_string((body_id, j), link_name)
                self.link_names.add_int_to_string((body_id, j), body_name + '/' + link_name)
                self.global_names.add_int_to_string(('link', body_id, j), link_name)
                self.global_names.add_int_to_string(('link', body_id, j), body_name + '/' + link_name)

    def get_body_name(self, body_id: int) -> str:
        return self.body_names.int_to_string[body_id]

    def get_link_name(self, body_id: int, link_id: int) -> str:
        return self.link_names.int_to_string[(body_id, link_id)]

    def get_body_index(self, body_name: Union[str, int]) -> int:
        if isinstance(body_name, str):
            return self.body_names.string_to_int[body_name]
        return body_name

    def get_link_index(self, link_name: Union[str, LinkIdentifier, Tuple[int, int]]) -> LinkIdentifier:
        if isinstance(link_name, str):
            return self.link_names.string_to_int[link_name]
        return link_name

    def get_link_index_with_body(self, body_id: int, link_name: str) -> int:
        body_name = self.body_names.int_to_string[body_id]
        return self.link_names.string_to_int[body_name + '/' + link_name].link_id

    def get_xpos(self, name, type=None):
        info = self.get_state(name, type)
        assert isinstance(info, (BodyState, LinkState))
        return info.position

    def get_xquat(self, name, type=None):
        info = self.get_state(name, type)
        assert isinstance(info, (BodyState, LinkState))
        return info.orientation

    def get_xmat(self, name, type=None):
        return p.getMatrixFromQuaternion(self.get_xquat(name, type=type), physicsClientId=self.client_id).reshape((3, 3))

    def get_qpos(self, name):
        info = self.get_joint_state(name)
        return info.position

    def get_qpos_by_id(self, body_id, joint_id):
        info = self.get_joint_state_by_id(body_id, joint_id)
        return info.position

    def set_qpos(self, name, qpos):
        return p.resetJointState(*self.joint_names[name], qpos, physicsClientId=self.client_id)

    def set_qpos_by_id(self, body_id, joint_id, qpos):
        return p.resetJointState(body_id, joint_id, qpos, physicsClientId=self.client_id)

    def get_batched_qpos(self, names, numpy=True):
        rv = [self.get_qpos(name) for name in names]
        if numpy:
            return np.array(rv)
        return rv

    def get_batched_qpos_by_id(self, body_id, joint_ids, numpy=True):
        rv = [self.get_qpos_by_id(body_id, joint_id) for joint_id in joint_ids]
        if numpy:
            return np.array(rv)
        return rv

    def set_batched_qpos(self, names, qpos):
        for name, q in zip(names, qpos):
            self.set_qpos(name, q)

    def set_batched_qpos_by_id(self, body_id, joint_ids, qpos):
        for index, q in zip(joint_ids, qpos):
            p.resetJointState(body_id, index, q, physicsClientId=self.client_id)

    def get_qvel(self, name):
        info = self.get_joint_state(name)
        return info.velocity

    def get_qvel_by_id(self, body_id, joint_id):
        info = self.get_joint_state_by_id(body_id, joint_id)
        return info.velocity

    def get_batched_qvel(self, names, numpy=True):
        rv = [self.get_qvel(name) for name in names]
        if numpy:
            return np.array(rv)
        return rv

    def get_batched_qvel_by_id(self, body_id, joint_ids, numpy=True):
        rv = [self.get_qvel_by_id(body_id, joint_id) for joint_id in joint_ids]
        if numpy:
            return np.array(rv)
        return rv

    def get_state(self, name: str, type: Optional[str] = None) -> Union[BodyState, LinkState, JointState]:
        if ':' in name:
            type, name = name.split(':')

        try:
            if type is None:
                record = self.global_names[name]
                type, args = record[0], record[1:]
            elif type == 'body':
                args = (self.body_names[type],)
            elif type == 'joint':
                args = self.joint_names[type]
            elif type == 'link':
                args = self.link_names[type]
            else:
                raise ValueError('Unknown object type: {}.'.format(type))
        except KeyError:
            raise ValueError('Unknown name: {}.'.format(name))

        if type == 'body':
            return self.get_body_state_by_id(*args)
        elif type == 'joint':
            return self.get_joint_state_by_id(*args)
        elif type == 'link':
            return self.get_link_state_by_id(*args)
        else:
            raise ValueError('Unknown object type: {}.'.format(type))

    def __getitem__(self, item):
        return self.get_state(item)

    def register_additional_state_saver(self, name: str, saver_factory: Callable[[], BulletSaver]):
        self.additional_state_savers[name] = saver_factory

    def unregister_additional_state_saver(self, name: str):
        del self.additional_state_savers[name]

    def register_managed_interface(self, name: str, interface: Any):
        self.managed_interfaces[name] = interface

    def unregister_managed_interface(self, name: str):
        del self.managed_interfaces[name]

    def get_debug_camera(self) -> DebugCameraState:
        return DebugCameraState(*p.getDebugVisualizerCamera(physicsClientId=self.client_id))

    def set_debug_camera(self, distance: float, yaw: float, pitch: float, target: Tuple[float, float, float]):
        p.resetDebugVisualizerCamera(distance, yaw, pitch, target, physicsClientId=self.client_id)

    def change_visual_color(self, body_id, rgba, link_id=None):
        if link_id is None:
            for i_body_id, link_id in self.link_names.int_to_string:
                if i_body_id == body_id:
                    p.changeVisualShape(body_id, link_id, rgbaColor=rgba, physicsClientId=self.client_id)
        else:
            p.changeVisualShape(body_id, link_id, rgbaColor=rgba, physicsClientId=self.client_id)

    def change_dynamics(self, body_id: int, mass: float, lateral_friction: float, link_id: int = -1):
        p.changeDynamics(body_id, link_id, mass=mass, lateralFriction=lateral_friction, physicsClientId=self.client_id)

    def get_body_state_by_id(self, body_id: int) -> BodyState:
        state = p.getBasePositionAndOrientation(body_id, physicsClientId=self.client_id)
        vel = p.getBaseVelocity(body_id, physicsClientId=self.client_id)
        return BodyState(np.array(state[0]), np.array(state[1]), np.array(vel[0]), np.array(vel[1]))

    def get_body_state(self, body_name: str) -> BodyState:
        return self.get_body_state_by_id(self.body_names[body_name])

    def set_body_state_by_id(self, body_id: int, state: BodyState):
        p.resetBasePositionAndOrientation(body_id, tuple(state.position), tuple(state.orientation), physicsClientId=self.client_id)
        p.resetBaseVelocity(body_id, tuple(state.linear_velocity), tuple(state.angular_velocity), physicsClientId=self.client_id)

    def set_body_state(self, body_name: str, state: BodyState):
        self.set_body_state_by_id(self.body_names[body_name], state)

    def set_body_state2_by_id(self, body_id: int, position: np.ndarray, orientation: np.ndarray):
        p.resetBasePositionAndOrientation(body_id, tuple(position), tuple(orientation), physicsClientId=self.client_id)

    def set_body_state2(self, body_name: str, position: np.ndarray, orientation: np.ndarray):
        self.set_body_state2_by_id(self.body_names[body_name], position, orientation)

    def get_joint_info_by_id(self, body_id: int, joint_id: int) -> JointInfo:
        info = p.getJointInfo(body_id, joint_id, physicsClientId=self.client_id)
        return JointInfo(*info)

    def get_joint_info_by_body(self, body_id: int) -> List[JointInfo]:
        return [self.get_joint_info_by_id(body_id, i) for i in range(p.getNumJoints(body_id, physicsClientId=self.client_id))]

    def get_joint_info(self, joint_name: str) -> JointInfo:
        return self.get_joint_info_by_id(*self.joint_names[joint_name])

    def get_joint_state_by_id(self, body_id: int, joint_id: int) -> JointState:
        state = p.getJointState(body_id, joint_id, physicsClientId=self.client_id)
        return JointState(state[0], state[1])

    def get_joint_state(self, joint_name: str) -> JointState:
        return self.get_joint_state_by_id(*self.joint_names[joint_name])

    def set_joint_state_by_id(self, body_id: int, joint_id: int, state: JointState):
        p.resetJointState(body_id, joint_id, state.position, state.velocity, physicsClientId=self.client_id)

    def set_joint_state(self, joint_name: str, state: JointState):
        self.set_joint_state_by_id(*self.joint_names[joint_name], state)

    def set_joint_state2_by_id(self, body_id: int, joint_id: int, position: float, velocity: float = 0.0):
        p.resetJointState(body_id, joint_id, position, velocity, physicsClientId=self.client_id)

    def set_joint_state2(self, joint_name: str, position: float, velocity: float = 0.0):
        self.set_joint_state2_by_id(*self.joint_names[joint_name], position, velocity)

    def get_link_state_by_id(self, body_id: int, link_id: int, fk=False) -> LinkState:
        if link_id == -1:
            state = p.getBasePositionAndOrientation(body_id, physicsClientId=self.client_id)
            vel = p.getBaseVelocity(body_id, physicsClientId=self.client_id)
            return LinkState(np.array(state[0]), np.array(state[1]), np.array(vel[0]), np.array(vel[1]))
        state = p.getLinkState(body_id, link_id, computeForwardKinematics=fk, computeLinkVelocity=fk, physicsClientId=self.client_id)
        return LinkState(np.array(state[0]), np.array(state[1]), np.array(state[6]), np.array(state[7]))

    def get_link_state(self, link_name: str, fk=False) -> LinkState:
        return self.get_link_state_by_id(*self.link_names[link_name], fk=fk)

    def get_collision_shape_data(self, body_name: str) -> List[CollisionShapeData]:
        body_id = self.body_names.as_identifier(body_name)
        return self.get_collision_shape_data_by_id(body_id)

    def get_collision_shape_data_by_id(self, body_id: int) -> List[CollisionShapeData]:
        output_data = list()
        for i in range(-1, p.getNumJoints(body_id, physicsClientId=self.client_id)):
            link_state = self.get_link_state_by_id(body_id, i, fk=True)

            data_list = p.getCollisionShapeData(body_id, i, physicsClientId=self.client_id)
            for shape_data in data_list:
                world_pos, world_orn = compose_transformation(link_state.position, link_state.orientation, shape_data[5], shape_data[6])
                # world_pos, world_orn = p.multiplyTransforms(link_state.position, link_state.orientation, shape_data[5], shape_data[6], physicsClientId=self.client_id)
                output_data.append(CollisionShapeData(*shape_data, world_pos, world_orn))
        return output_data

    def get_visual_shape_data(self, body_name: str) -> List[VisualShapeData]:
        body_id = self.body_names.as_identifier(body_name)
        return self.get_visual_shape_data_by_id(body_id)

    def get_visual_shape_data_by_id(self, body_id: int) -> List[VisualShapeData]:
        output_data = list()
        for i in range(-1, p.getNumJoints(body_id, physicsClientId=self.client_id)):
            link_state = self.get_link_state_by_id(body_id, i, fk=True)

            data_list = p.getVisualShapeData(body_id, i, physicsClientId=self.client_id)
            for shape_data in data_list:
                world_pos, world_orn = compose_transformation(link_state.position, link_state.orientation, shape_data[5], shape_data[6])
                # world_pos, world_orn = p.multiplyTransforms(link_state.position, link_state.orientation, shape_data[5], shape_data[6], physicsClientId=self.client_id)
                output_data.append(VisualShapeData(*shape_data, world_pos, world_orn))
        return output_data

    def get_free_joints(self, body_id: int) -> List[int]:
        """Get a list of indices corresponding to all non-fixed joints in the body."""
        all_joints = list()
        for joint_id in range(p.getNumJoints(body_id, physicsClientId=self.client_id)):
            if self.get_joint_info_by_id(body_id, joint_id).joint_type != p.JOINT_FIXED:
                all_joints.append(joint_id)
        return all_joints

    def get_constraint(self, constraint_id: int) -> ConstraintInfo:
        return ConstraintInfo(*p.getConstraintInfo(constraint_id, physicsClientId=self.client_id))

    def perform_collision_detection(self):
        p.performCollisionDetection(physicsClientId=self.client_id)

    def get_contact(self, a: Optional[Union[int, LinkIdentifier, str]] = None, b: Optional[Union[int, LinkIdentifier, str]] = None, update: bool = False) -> List[ContactInfo]:
        if update:
            p.performCollisionDetection(physicsClientId=self.client_id)

        kwargs = dict(physicsClientId=self.client_id)

        def update_kwargs(name, value):
            if value is not None:
                if isinstance(value, int):
                    kwargs['body' + name] = value
                elif isinstance(value, (tuple, list)):
                    kwargs['body' + name], kwargs['linkIndex' + name] = value
                elif isinstance(value, str):
                    value = self.global_names[value]
                    if value[0] == 'body':
                        kwargs['body' + name] = value[1]
                    elif value[0] == 'link':
                        kwargs['body' + name], kwargs['linkIndex' + name] = value[1:]
                    else:
                        raise ValueError('get_contact API only allows the specification of body or link.')
                else:
                    raise TypeError(f'get_contact API only allows the specification of body or link: got {value}')

        update_kwargs('A', a)
        update_kwargs('B', b)
        while True:
            contacts = p.getContactPoints(**kwargs)
            if contacts is not None:
                break
            # TODO(Jiayuan Mao @ 2023/04/13): sometimes PyBullet returns None, not sure why.
            time.sleep(0.001)
        return [ContactInfo(self, *c) for c in contacts]

    def check_collision(self, a: Union[int, LinkIdentifier, str], b: Union[int, LinkIdentifier, str], update: bool = False) -> bool:
        return len(self.get_contact(a, b, update=update)) > 0

    def check_collision_single(self, a: Union[int, LinkIdentifier, str], ignored_objects: List[int], ignore_self_collision: bool = True, update: bool = False) -> bool:
        """Check if the object is in collision with any object other than the ignored objects."""
        contacts = self.get_contact(a, update=update)
        for contact in contacts:
            if isinstance(a, int) and ignore_self_collision and contact.body_b == a:
                continue
            if contact.body_b not in ignored_objects:
                return True
        return False

    def get_single_contact_normal(self, object_id: int, support_object_id: int, deviation_tol: float = 0.05, return_center: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        body_names = self.body_names.int_to_string
        object_name, support_name = body_names[object_id], body_names[support_object_id]
        contacts = self.get_contact(object_id, support_object_id)

        if len(contacts) == 0:
            # TODO(Jiayuan Mao @ 2023/03/15): find a better way to configure this.
            # self.client.step(1)
            p.stepSimulation(physicsClientId=self.client_id)
            contacts = self.get_contact(object_id, support_object_id)

        if len(contacts) == 0:
            raise ValueError(f'No contact between {object_name} and {support_name}.')

        contact_normals = np.array([c.contact_normal_on_b for c in contacts])
        contact_normal_avg = np.mean(contact_normals, axis=0)
        contact_normal_avg /= np.linalg.norm(contact_normal_avg)

        deviations = np.abs(1 - contact_normals.dot(contact_normal_avg) / np.linalg.norm(contact_normals, axis=1))
        if np.max(deviations) > deviation_tol:
            raise ValueError(
                f'Contact normals of {object_name} and {support_name} are not consistent. This is likely due to multiple contact points.\n'
                f'  Contact normals: {contact_normals}\n  Deviations: {deviations}.'
            )

        if return_center:
            centers = np.array([c.position_on_b for c in contacts])
            center = np.mean(centers, axis=0)
            return center, contact_normal_avg

        return contact_normal_avg

    def get_supporting_objects_by_id(self, body_id: int, return_name: bool = True) -> List[Union[str, int]]:
        """Get the bodies that are supporting the given body."""
        all_contact = self.get_contact(body_id)
        supported_by_list = set()
        for contact in all_contact:
            body_name = contact.body_b_name
            if body_name == 'robot':
                continue

            normal = contact.contact_normal_on_b
            if normal[2] > np.cos(np.deg2rad(45)):
                if return_name:
                    supported_by_list.add(body_name)
                else:
                    supported_by_list.add(contact.body_b)
        return list(supported_by_list)

    def update_contact(self):
        p.performCollisionDetection(physicsClientId=self.client_id)

    def is_collision_free(self, a: Union[int, LinkIdentifier, str], b: Union[int, LinkIdentifier, str]) -> bool:
        return len(self.get_contact(a, b)) == 0

    def save_world(self) -> WorldSaver:
        return WorldSaver(self)

    def save_world_builtin(self) -> WorldSaverBuiltin:
        return WorldSaverBuiltin(self)

    def save_body(self, body_identifier: Union[str, int]) -> BodyFullStateSaver:
        if isinstance(body_identifier, int):
            return BodyFullStateSaver(self, body_identifier)
        else:
            return BodyFullStateSaver(self, self.body_names[body_identifier])

    def save_bodies(self, body_identifiers: List[Union[str, int]]) -> GroupSaver:
        return GroupSaver([self.save_body(b) for b in body_identifiers])

    def render_image(self, config: CameraConfig, image_size: Optional[Tuple[int, int]] = None, normalize_depth: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        view_matrix, proj_matrix = config.get_view_and_projection_matricies(image_size)
        image_size = image_size if image_size is not None else config.image_size
        znear, zfar = config.zrange

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = p.getCameraImage(
            width=image_size[1],
            height=image_size[0],
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            shadow=1,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # Get color image.
        color_image_size = (image_size[0], image_size[1], 4)
        color = np.asarray(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel

        # Get depth image.
        depth_image_size = (image_size[0], image_size[1])
        zbuffer = np.asarray(depth).reshape(depth_image_size)

        if normalize_depth:
            depth = (zfar + znear - (2. * zbuffer - 1.) * (zfar - znear))
            depth = (2. * znear * zfar) / depth
        else:
            depth = zbuffer

        # Get segmentation image.
        segm = np.uint8(segm).reshape(depth_image_size)

        return color, depth, segm

    def get_pointcloud(self, body_id: int, points_per_geom: int = 1000, zero_center: bool = True) -> np.ndarray:
        """Get the point cloud of a body."""

        all_pcds = list()

        body_state = self.get_body_state_by_id(body_id)
        for shape_info in self.get_collision_shape_data_by_id(body_id):
            if shape_info.shape_type == p.GEOM_BOX:
                pcd = self.get_pointcloud_box(shape_info.dimensions, points_per_geom)
            # elif shape_info.shape_type == client.p.GEOM_SPHERE:
            #     return get_point_cloud_sphere(shape_info.dimensions, points_per_geom)
            elif shape_info.shape_type == p.GEOM_MESH:
                pcd = self.get_pointcloud_mesh(shape_info.filename, shape_info.dimensions[0], points_per_geom)
            else:
                raise ValueError(f'Unsupported shape type: {shape_info.shape_type}.')

            if zero_center:
                pos = shape_info.world_pos - body_state.pos
                orn = quat_mul(quat_conjugate(body_state.quat_xyzw), shape_info.world_orn)
            else:
                pos, orn = shape_info.world_pos, shape_info.world_orn

            pcd = self.transform_pcd(pcd, pos, orn)
            all_pcds.append(pcd)

        if len(all_pcds) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return np.concatenate(all_pcds, axis=0)

    @_geometry_cache(type_id=0, geom_name_template='box_{dimensions[0]}_{dimensions[1]}_{dimensions[2]}_{points_per_geom}')
    def get_pointcloud_box(self, dimensions, points_per_geom=1000) -> np.ndarray:
        """Get a point cloud for a box."""

        total_volume = dimensions[0] * dimensions[1] * dimensions[2]
        density = points_per_geom / total_volume
        linear_density = density ** (1. / 3.)

        x, y, z = np.meshgrid(
            np.linspace(-dimensions[0] / 2, dimensions[0] / 2, int(np.ceil(dimensions[0] * linear_density))),
            np.linspace(-dimensions[1] / 2, dimensions[1] / 2, int(np.ceil(dimensions[1] * linear_density))),
            np.linspace(-dimensions[2] / 2, dimensions[2] / 2, int(np.ceil(dimensions[2] * linear_density))),
        )
        return np.stack([x, y, z], axis=-1).reshape(-1, 3)

    @_geometry_cache(type_id=0, geom_name_template='mesh_{mesh_file}_{mesh_scale}_{points_per_geom}')
    def get_pointcloud_mesh(self, mesh_file, mesh_scale, points_per_geom=1000) -> np.ndarray:
        """Get a point cloud for a mesh."""
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        pcd = mesh.sample_points_poisson_disk(points_per_geom)
        return np.asarray(pcd.points, dtype=np.float32) * mesh_scale

    def transform_pcd(self, raw_pcd, pos, quat_xyzw):
        return rotate_vector_batch(raw_pcd, quat_xyzw) + pos

    def get_mesh(self, body_id: int, zero_center: bool = True, verbose: bool = False, mesh_filename: Optional[str] = None, mesh_scale: float = 1.0) -> o3d.geometry.TriangleMesh:
        """Get the point cloud of a body.

        Args:
            body_id: the ID of the body.
            zero_center: whether to zero-center the mesh (i.e., move the center of the mesh to the origin).
            verbose: whether to print debug information.
            mesh_filename: the filename of the mesh. This should be provided if the body has a mesh shape but we can't get it from the collision shape data.
            mesh_scale: the scale of the mesh. This should be provided if the body has a mesh shape but we can't get it from the collision shape data.
        """

        base_mesh = o3d.geometry.TriangleMesh()
        body_state = self.get_body_state_by_id(body_id)
        for shape_info in self.get_collision_shape_data_by_id(body_id):
            if shape_info.shape_type == p.GEOM_BOX:
                mesh = self.get_mesh_box(shape_info.dimensions)
            # elif shape_info.shape_type == client.p.GEOM_SPHERE:
            #     return get_point_cloud_sphere(shape_info.dimensions, points_per_geom)
            elif shape_info.shape_type == p.GEOM_MESH:
                if mesh_filename is not None:
                    mesh = self.get_mesh_mesh(mesh_filename, mesh_scale)
                else:
                    mesh = self.get_mesh_mesh(shape_info.filename, shape_info.dimensions[0])
            else:
                raise ValueError(f'Unsupported shape type: {shape_info.shape_type}.')

            if zero_center:
                pos = shape_info.world_pos - body_state.pos
                orn = quat_mul(quat_conjugate(body_state.quat_xyzw), shape_info.world_orn)
            else:
                pos, orn = shape_info.world_pos, shape_info.world_orn

            if verbose:
                print(shape_info)
                print('Zero-center:', zero_center)
                print(f'Transforming the mesh: pos: {pos}, orn: {orn}')

            mesh = self.transform_mesh(mesh, pos, orn)
            base_mesh += mesh

        return base_mesh

    @_geometry_cache(type_id=1, geom_name_template='box_{dimensions[0]}_{dimensions[1]}_{dimensions[2]}')
    def get_mesh_box(self, dimensions) -> o3d.geometry.TriangleMesh:
        """Get a triangle mesh for a box primitive."""
        mesh = o3d.geometry.TriangleMesh.create_box(dimensions[0], dimensions[1], dimensions[2])
        mesh = mesh.translate([-dimensions[0] / 2, -dimensions[1] / 2, -dimensions[2] / 2])
        return mesh

    @_geometry_cache(type_id=1, geom_name_template='mesh_{mesh_file}_{mesh_scale}')
    def get_mesh_mesh(self, mesh_file, mesh_scale) -> o3d.geometry.TriangleMesh:
        """Get a triangle mesh for a mesh primitive."""
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        mesh = mesh.scale(mesh_scale, np.array([0, 0, 0], dtype=np.float64))
        return mesh

    def transform_mesh(self, mesh: o3d.geometry.TriangleMesh, pos, quat_xyzw):
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        mesh = o3d.geometry.TriangleMesh(mesh)
        mesh = mesh.rotate(rotation_matrix, center=(0, 0, 0))
        mesh = mesh.translate(pos)
        return mesh

    def transform_mesh2(self, mesh: o3d.geometry.TriangleMesh, pos: np.ndarray, quat_xyzw: np.ndarray, current_pos: np.ndarray, current_quat_xyzw: np.ndarray):
        rotation_matrix1 = o3d.geometry.get_rotation_matrix_from_quaternion([current_quat_xyzw[3], -current_quat_xyzw[0], -current_quat_xyzw[1], -current_quat_xyzw[2]])
        rotation_matrix2 = o3d.geometry.get_rotation_matrix_from_quaternion([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        mesh = o3d.geometry.TriangleMesh(mesh)
        mesh.translate(-current_pos)
        mesh.rotate(rotation_matrix1, center=np.array([0, 0, 0]))
        mesh.rotate(rotation_matrix2, center=np.array([0, 0, 0]))
        mesh.translate(pos)
        return mesh

    def get_mesh_info(self, body_id: int, link_id: int = -1) -> Tuple[str, float, np.ndarray, np.ndarray]:
        """Get the mesh filename and transform of a body or link."""
        visual_data = self.get_visual_shape_data_by_id(body_id)[link_id + 1]
        mesh_filename = visual_data.filename.decode('utf-8')
        scale = visual_data.dimensions[0]
        pos = visual_data.world_pos
        orn = visual_data.world_orn
        return mesh_filename, scale, pos, orn

    def get_all_mesh_info(self, body_id: int) -> List[Tuple[str, float, np.ndarray, np.ndarray]]:
        """Get the mesh filenames and transforms of all links of a body."""
        visual_data = self.get_visual_shape_data_by_id(body_id)
        filenames_and_transforms = []
        for data in visual_data:
            mesh_filename = data.filename.decode('utf-8')
            scale = data.dimensions[0]
            pos = data.world_pos
            orn = data.world_orn
            filenames_and_transforms.append((mesh_filename, scale, pos, orn))
        return filenames_and_transforms
