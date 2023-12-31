#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : pybullet_tabletop.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/01/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os.path as osp
from typing import Optional, Union, List

import numpy as np
import jacinle.io as io

from concepts.simulator.pybullet.client import BulletClient
from concepts.simulator.pybullet.world import WorldSaver
from concepts.simulator.pybullet.components.ur5.ur5_robot import UR5Robot
from concepts.simulator.pybullet.components.panda.panda_robot import PandaRobot
from concepts.simulator.pybullet.rotation_utils import rpy
from concepts.pdsketch.domain import State
from concepts.pdsketch.executor import PDSketchExecutor
from concepts.utils.typing_utils import Vec2f, Vec3f, Vec4f, BroadcastableVec3f

__all__ = ['TableTopEnv', 'SimpleTableTopEnv', 'get_tabletop_base_domain_filename']


class TableTopEnv(object):
    """TableTopEnv is a environment for manipulating tools in a 3D but table-top-only environment.
    So there is only minimal amount of 3D information involved.

    The environment will be simulated by pybullet.
    """

    def __init__(self, client: Optional[BulletClient] = None, executor: Optional[PDSketchExecutor] = None, seed: int = 1234):
        if client is None:
            client = BulletClient()

        self.client = client
        self.robot = None
        self.saver = WorldSaver(client.w, save=False)
        self.metainfo = dict()

        self._executor = executor
        self.np_random = np.random.RandomState(seed)

    @property
    def executor(self) -> PDSketchExecutor:
        if self._executor is None:
            raise RuntimeError('Executor is not initialized yet.')
        return self._executor

    def set_executor(self, executor: PDSketchExecutor):
        self._executor = executor

    def seed(self, seed: int):
        self.np_random.seed(seed)

    @property
    def world(self):
        return self.client.w

    @property
    def w(self):
        return self.client.w

    @property
    def p(self):
        return self.client.p

    def reset(self) -> None:
        self.saver.restore()

    def restore(self) -> None:
        """Restore the environment to the last saved state."""
        self.saver.restore()

    def set_default_debug_camera(self, distance: float = 1.0):
        """Set the default debug camera of the environment."""
        target = self.world.get_debug_camera().target
        self.world.set_debug_camera(distance, 90, -25, target=target)

    PLANE_FILE = 'assets://basic/plane/plane.urdf'
    WORKSPACE_FILE = 'assets://basic/plane/workspace.urdf'

    PLANE_LARGE_FILE = 'assets://basic/plane/plane_large.urdf'
    WORKSPACE_LARGE_FILE = 'assets://basic/plane/workspace_large.urdf'

    def add_workspace(self, large: bool = False) -> int:
        """Add a table-top workspace to the environment.

        Returns:
            the body id of the collision shape of the body.
        """

        if not large:
            # The actual table with collision shape.
            plane_id = self.client.load_urdf(type(self).PLANE_FILE, (0, 0, -0.001), static=True, body_name='table')
            # Just a visual shape for the table.
            self.client.load_urdf(type(self).WORKSPACE_FILE, (1, 0, 0), static=True, body_name='workspace')
        else:
            plane_id = self.client.load_urdf(type(self).PLANE_LARGE_FILE, (0, 0, -0.001), static=True, body_name='table')
            self.client.load_urdf(type(self).WORKSPACE_LARGE_FILE, (1, 0, 0), static=True, body_name='workspace')
        return plane_id

    def add_robot(self, robot: str = 'panda', pos: Optional[Vec3f] = None) -> int:
        """Add a robot to the environment.

        Args:
            robot: the type of the robot. Currently only ``['ur5', 'panda']`` are supported.
            pos: the initial position of the robot. If not given, the robot will be placed at the origin.

        Returns:
            the body id of the robot.
        """
        if robot == 'ur5':
            self.robot = UR5Robot(self.client, pos=pos)
        elif robot == 'panda':
            self.robot = PandaRobot(self.client, pos=pos)
        else:
            raise ValueError(f'Unknown robot type: {robot}.')

        return self.robot.get_robot_body_id()

    def add_region(
        self, size_2d: Vec2f, location_2d: Vec2f, name: str = 'region', *,
        color_rgba: Vec4f = (0.5, 0.5, 0.5, 1)
    ) -> int:
        """Add a visual-only region indicator to the environment.

        Args:
            size_2d: the size of the region.
            location_2d: the location of the region, asssumed to be on the table (z = 0).
            name: the name of the region.
            color_rgba: the color of the region.

        Returns:
            the body id of the region.
        """
        visual_shape = self.p.createVisualShape(self.p.GEOM_BOX, halfExtents=[size_2d[0] / 2, size_2d[1] / 2, 0.0001], rgbaColor=color_rgba)
        shape = self.p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=[location_2d[0], location_2d[1], 0.001],
            baseOrientation=(0, 0, 0, 1)
        )
        self.world.notify_update(shape, name, 'region')
        return shape

    def add_box(
        self, size: BroadcastableVec3f, location_2d: Vec2f, name: str = 'box', *,
        static: bool = False, z_height: float = 0,
        mass: float = 0.2, lateral_friction: float = 1.0,
        color_rgba: Vec4f = (1, 0.34, 0.34, 1.),
        quat: Optional[Vec4f] = None
    ) -> int:
        if isinstance(size, float):
            size = (size, size, size)
        return self.client.load_urdf_template(
            'assets://basic/box/box-template.urdf',
            {'DIM': size, 'MASS': mass, 'COLOR': color_rgba, 'LATERAL_FRICTION': lateral_friction},
            pos=(location_2d[0], location_2d[1], size[2] / 2 + z_height),
            quat=quat,
            body_name=name,
            static=static,
        )

    # TODO(Jiayuan Mao @ 08/12): use better typing like Vector2f.
    def add_container(
        self, size_2d: Vec2f, depth: float, location_2d: Vec2f, name: str = 'container', *,
        static=True,
        color_rgba: Optional[Vec4f] = None,
        quat: Optional[Vec4f] = None,
    ) -> int:
        container_id = self.client.load_urdf_template(
            'assets://basic/container/container-template.urdf', {
                'DIM': (size_2d[0], size_2d[1], depth),
                'HALF': (size_2d[0] / 2, size_2d[1] / 2, depth / 2),
            },
            (location_2d[0], location_2d[1], depth / 2),
            quat=quat,
            rgba=(0.5, 1.0, 0.5, 1.0),
            body_name=name,
            static=static,
        )

        if color_rgba is not None:
            self.p.changeVisualShape(container_id, -1, rgbaColor=color_rgba)
        return container_id

    def add_bar(
        self, size_2d: Vec2f, thickness: float, location_2d: Vec2f, name: str = 'bar-shape', *,
        static: bool = False,
        quat: Vec4f = (0, 0, 0, 1)
    ) -> int:
        return self.client.load_urdf_template(
            'assets://basic/simple-tools/bar-shape-template.urdf',
            {'DIM': (size_2d[0], size_2d[1], thickness)},
            (location_2d[0], location_2d[1], thickness / 2),
            quat,
            body_name=name,
            static=static,
        )

    def add_l_shape(
        self, size1_2d: Vec2f, size2_2d: Vec2f, thickness: float, location_2d: Vec2f, name: str = 'l-shape', *,
        static: bool = False,
        quat: Vec4f = (0, 0, 0, 1)
    ) -> int:
        return self.client.load_urdf_template(
            'assets://basic/simple-tools/l-shape-template.urdf', {
                'DIMX': (size1_2d[0], size1_2d[1], thickness),
                'HALFX': (size1_2d[0] / 2, size1_2d[1] / 2, thickness / 2),
                'DIMY': (size2_2d[0], size2_2d[1], thickness),
                'HALFY': (size2_2d[0] / 2, size2_2d[1] / 2, thickness / 2),
                'DISP': ((size1_2d[0] + size2_2d[0]) / 2, (size2_2d[1] - size1_2d[1]) / 2, 0.0)
            },
            (location_2d[0], location_2d[1], thickness / 2),
            quat,
            body_name=name,
            static=static,
        )

    def add_l_shape_with_tip(
        self, size1_2d: Vec2f, size2_2d: Vec2f, size3_2d: Vec2f, thickness: float, location_2d: Vec2f, name: str = 'l-shape-with-tip', *,
        static=False,
        quat: Vec4f = (0, 0, 0, 1)
    ) -> int:
        return self.client.load_urdf_template(
            'assets://basic/simple-tools/l-shape-with-tip-template.urdf', {
                'DIMX': (size1_2d[0], size1_2d[1], thickness),
                'HALFX': (size1_2d[0] / 2, size1_2d[1] / 2, thickness / 2),
                'DIMY': (size2_2d[0], size2_2d[1], thickness),
                'HALFY': (size2_2d[0] / 2, size2_2d[1] / 2, thickness / 2),
                'DIMZ': (size3_2d[0], size3_2d[1], thickness),
                'HALFZ': (size3_2d[0] / 2, size3_2d[1] / 2, thickness / 2),
                'DISPY': ((size1_2d[0] + size2_2d[0]) / 2, (size2_2d[1] - size1_2d[1]) / 2, 0.0),
                'DISPZ': ((-size2_2d[0] - size3_2d[0]) / 2, (size2_2d[1] - size3_2d[1]) / 2, 0.0)
            },
            (location_2d[0], location_2d[1], thickness / 2),
            quat,
            body_name=name,
            static=static,
        )

    def add_t_shape(
        self, size1_2d: Vec2f, size2_2d: Vec2f, thickness: float, location_2d: Vec2f, name: str = 't-shape', *,
        static: bool = False,
        quat: Vec4f = (0, 0, 0, 1)
    ) -> int:
        return self.client.load_urdf_template(
            'assets://basic/simple-tools/t-shape-template.urdf', {
                'DIMX': (size1_2d[0], size1_2d[1], thickness),
                'HALFX': (size1_2d[0] / 2, size1_2d[1] / 2, thickness / 2),
                'DIMY': (size2_2d[0], size2_2d[1], thickness),
                'HALFY': (size2_2d[0] / 2, size2_2d[1] / 2, thickness / 2),
                'DISP': ((size1_2d[0] + size2_2d[0]) / 2, 0, 0.0)
            },
            (location_2d[0], location_2d[1], thickness / 2),
            quat,
            body_name=name,
            static=static,
        )

    def add_t_shape_with_tip(
        self, size1_2d: Vec2f, size2_2d: Vec2f, size3_2d: Vec2f, thickness: float, location_2d: Vec2f, name: str = 't-shape-with-tip', *,
        static: bool = False,
        quat: Vec4f = (0, 0, 0, 1)
    ) -> int:
        return self.client.load_urdf_template(
            'assets://basic/simple-tools/t-shape-with-tip-template.urdf', {
                'DIMX': (size1_2d[0], size1_2d[1], thickness),
                'HALFX': (size1_2d[0] / 2, size1_2d[1] / 2, thickness / 2),
                'DIMY': (size2_2d[0], size2_2d[1], thickness),
                'HALFY': (size2_2d[0] / 2, size2_2d[1] / 2, thickness / 2),
                'DIMZ': (size3_2d[0], size3_2d[1], thickness),
                'HALFZ': (size3_2d[0] / 2, size3_2d[1] / 2, thickness / 2),
                'DISPY': ((size1_2d[0] + size2_2d[0]) / 2, 0.0, 0.0),
                'DISPZ1': ((-size2_2d[0] - size3_2d[0]) / 2, (size2_2d[1] - size3_2d[1]) / 2, 0.0),
                'DISPZ2': ((-size2_2d[0] - size3_2d[0]) / 2, (-size2_2d[1] + size3_2d[1]) / 2, 0.0)
            },
            (location_2d[0], location_2d[1], thickness / 2),
            quat,
            body_name=name,
            static=static,
        )

    def add_plate(
        self, scale: float, location_2d: Vec2f, name: str = 'plate', *,
        static: bool = False, z_height: float = 0.0
    ) -> int:
        return self.client.load_urdf(
            'assets://objects/kitchenware/plate1/model_normalized.obj.urdf',
            (location_2d[0], location_2d[1], 0.063 * scale + z_height),
            rpy(90, 0, 0),
            scale=scale,
            body_name=name,
            static=static,
        )

    def add_thin_plate(
        self, scale: float, location_2d: Vec2f, name: str = 'plate', *,
        static: bool = False, z_height: float = 0.0
    ) -> int:
        return self.client.load_urdf(
            'assets://objects/ycb/029_plate/textured.obj.urdf',
            (location_2d[0], location_2d[1], 0.063 * scale + z_height),
            scale=scale,
            body_name=name,
            static=static,
        )

    @property
    def igibson_root(self) -> str:
        return osp.join(self.client.assets_root, 'igibson')

    def find_igibson_object_by_category(self, category: str) -> List[str]:
        path = osp.join(self.client.assets_root, 'igibson', 'objects', category)
        all_files = io.lsdir(path, '*/*.urdf')
        assert len(all_files) > 0, f'No urdf files found in {path}.'
        return all_files

    def add_igibson_object_by_category(
        self, category: str, location_3d: Vec3f, scale: float = 1.0, name: Optional[str] = None, *,
        urdf_file: Optional[str] = None,
        static: bool = False, quat: Vec4f = (0, 0, 0, 1),
    ):
        if name is None:
            name = category

        if urdf_file is None:
            all_files = self.find_igibson_object_by_category(category)
            urdf_file = all_files[0]

        return self.client.load_urdf(
            urdf_file,
            location_3d,
            quat,
            body_name=name,
            scale=scale,
            static=static,
        )

    @property
    def ocrtoc_root(self) -> str:
        return osp.join(self.client.assets_root, 'ocrtoc')

    def list_ocrtoc_objects(self) -> List[str]:
        all_model_names = list()
        for dirname in io.lsdir(self.ocrtoc_root, '*'):
            if osp.isdir(dirname) and osp.exists(osp.join(dirname, 'model.urdf')):
                all_model_names.append(osp.basename(dirname))
        return all_model_names

    def add_ocrtoc_object(
        self, identifier: str, location_3d: Vec3f, scale: float = 1.0, name: Optional[str] = None, *,
        static: bool = False, quat: Vec4f = (0, 0, 0, 1)
    ):
        if name is None:
            name = identifier
        return self.client.load_urdf(
            osp.join(self.ocrtoc_root, identifier, 'model.urdf'),
            location_3d,
            quat,
            body_name=name,
            scale=scale,
            static=static,
        )

    def get_support(self, body_id: int, return_name: bool = True) -> List[Union[str, int]]:
        return _get_support(self, body_id, return_name=return_name)



class SimpleTableTopEnv(TableTopEnv):
    def reset(self):
        with self.client.disable_rendering(disable_rendering=True):
            metainfo = self._reset_scene()

        self.metainfo = metainfo
        self.saver.save()

    def _reset_scene(self):
        raise NotImplementedError

    def get_pds_state(self) -> State:
        objects = dict()
        for name, info in self.metainfo.items():
            object_type = self.executor.domain.types['robot'] if name == 'robot' else self.executor.domain.types['item']
            objects[name] = object_type

        state, ctx = self.executor.new_state(objects, create_context=True)

        for name, info in self.metainfo.items():
            index = info['id']
            if name == 'robot':
                ctx.set_value('robot-qpos', [name], self.robot.get_qpos())
                ctx.set_value('robot-identifier', [name], index)
            else:
                ctx.set_value('item-pose', [name], self.world.get_body_state_by_id(index).get_7dpose())
                ctx.set_value('item-identifier', [name], index)

        for name, info in self.metainfo.items():
            if name not in ('robot', 'table', 'panda'):
                for name2 in _get_support(self, info['id']):
                    if name2 not in ('robot', 'panda'):
                        ctx.set_value('support', [name, name2], True)

        ctx.init_feature('moveable')
        for name, info in self.metainfo.items():
            if 'moveable' in info and info['moveable']:
                ctx.set_value('moveable', [name], True)

        if hasattr(self.robot, 'gripper_constraint'):
            if self.robot.gripper_constraint is None:
                ctx.define_predicates([ctx.robot_hands_free('robot')])
            else:
                constraint = self.world.get_constraint(self.robot.gripper_constraint)
                name = self.world.body_names[constraint.child_body]
                ctx.define_predicates([ctx.robot_holding_item('robot', name)])

        return state


def get_tabletop_base_domain_filename() -> str:
    return osp.join(osp.dirname(__file__), 'pybullet_tabletop_base.pdsketch')


def _get_support(env: TableTopEnv, body_id: int, return_name: bool = True) -> List[Union[str, int]]:
    all_contact = env.world.get_contact(body_id)
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
