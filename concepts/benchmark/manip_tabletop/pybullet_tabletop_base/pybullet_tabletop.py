#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : pybullet_tabletop.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/01/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Optional, Union, Tuple, List

import numpy as np

from concepts.math.rotationlib_wxyz import rpy
from concepts.simulator.pybullet.client import BulletClient
from concepts.simulator.pybullet.default_env import BulletEnvBase
from concepts.simulator.pybullet.world import WorldSaver
from concepts.simulator.pybullet.components.ur5.ur5_robot import UR5Robot
from concepts.simulator.pybullet.components.panda.panda_robot import PandaRobot
from concepts.utils.typing_utils import Vec2f, Vec3f, Vec4f, BroadcastableVec3f

__all__ = ['TableTopEnv', 'SimpleTableTopEnv']


class TableTopEnv(BulletEnvBase):
    """TableTopEnv is a environment for manipulating tools in a 3D but table-top-only environment.
    So there is only minimal amount of 3D information involved.

    The environment will be simulated by pybullet.
    """

    def __init__(self, client: Optional[BulletClient] = None, is_gui: bool = True, seed: int = 1234):
        super().__init__(client, seed=seed, is_gui=is_gui)
        self.saver = WorldSaver(self.world, save=False)
        self.metainfo = dict()

    def reset(self) -> None:
        raise NotImplementedError('The reset method should be implemented by the subclass.')

    def restore(self) -> None:
        """Restore the environment to the last saved state."""
        self.saver.restore()

    def set_default_debug_camera(self, distance: float = 1.5):
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

    CLIPORT_PLANE_FILE = 'assets://cliport/plane/plane.urdf'
    CLIPORT_WORKSPACE_FILE = 'assets://cliport/workspace/workspace.urdf'

    def add_cliport_workspace(self):
        plane_id = self.client.load_urdf(type(self).CLIPORT_PLANE_FILE, (0, 0, -0.001), static=True, body_name='table')
        self.client.load_urdf(type(self).CLIPORT_WORKSPACE_FILE, (0.5, 0, 0), static=True, body_name='workspace')
        return plane_id

    def add_workspace_boundary(self, x_range: Tuple[float, float], y_range: Tuple[float, float], z_range: Tuple[float, float], name: str = 'workspace-boundary') -> int:
        """Add a workspace boundary to the environment."""

        parameters = {
            'X_PLANE_SIZE_Y': y_range[1] - y_range[0],
            'X_PLANE_SIZE_Z': z_range[1] - z_range[0],
            'Y_PLANE_SIZE_X': x_range[1] - x_range[0],
            'Y_PLANE_SIZE_Z': z_range[1] - z_range[0],
            'Z_PLANE_SIZE_X': x_range[1] - x_range[0],
            'Z_PLANE_SIZE_Y': y_range[1] - y_range[0],
            'X_PLANE_MIN_POS': (x_range[0] - 0.05 - 0.005, (y_range[0] + y_range[1]) / 2, (z_range[0] + z_range[1]) / 2),
            'X_PLANE_MAX_POS': (x_range[1] + 0.05 + 0.005, (y_range[0] + y_range[1]) / 2, (z_range[0] + z_range[1]) / 2),
            'Y_PLANE_MIN_POS': ((x_range[0] + x_range[1]) / 2, y_range[0] - 0.05 - 0.005, (z_range[0] + z_range[1]) / 2),
            'Y_PLANE_MAX_POS': ((x_range[0] + x_range[1]) / 2, y_range[1] + 0.05 + 0.005, (z_range[0] + z_range[1]) / 2),
            'Z_PLANE_MIN_POS': ((x_range[0] + x_range[1]) / 2, (y_range[0] + y_range[1]) / 2, z_range[0] - 0.05 - 0.005),
            'Z_PLANE_MAX_POS': ((x_range[0] + x_range[1]) / 2, (y_range[0] + y_range[1]) / 2, z_range[1] + 0.05 + 0.005),
        }

        return self.client.load_urdf_template(
            'assets://basic/workspace/workspace_boundary.urdf',
            parameters,
            (0, 0, 0),
            body_name=name,
            static=True,
        )

    def add_table(self, x_range: Tuple[float, float], y_range: Tuple[float, float], surface_z: float, name: str = 'table') -> int:
        box_size = (x_range[1] - x_range[0], y_range[1] - y_range[0], 1)
        TABLE_URDF_STRING = f"""
<?xml version="1.0" ?>
<robot name="plane">
  <link name="planeLink">
    <contact>
      <lateral_friction value="1"/>
    </contact>
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value=".0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 -0.5"/>
      <geometry>
        <box size="{box_size[0]} {box_size[1]} {box_size[2]}"/>
      </geometry>
    </collision>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 -0.5"/>
      <geometry>
        <box size="{box_size[0]} {box_size[1]} {box_size[2]}"/>
      </geometry>
      <material name="LightGrey">
        <color rgba="0.9 0.9 0.9 1"/>
      </material>
    </visual>
  </link>
</robot>
"""
        x_center = (x_range[0] + x_range[1]) / 2
        y_center = (y_range[0] + y_range[1]) / 2
        return self.client.load_urdf_string(TABLE_URDF_STRING, (x_center, y_center, surface_z), static=True, body_name=name)

    def add_robot(self, robot: str = 'panda', pos: Optional[Vec3f] = None, quat: Optional[Vec4f] = None, name: Optional[str] = None, robot_kwargs: Optional[dict] = None) -> int:
        """Add a robot to the environment.

        Args:
            robot: the type of the robot. Currently only ``['ur5', 'panda']`` are supported.
            pos: the initial position of the robot. If not given, the robot will be placed at the origin.
            quat: the initial orientation of the robot. If not given, the robot will be placed in the default orientation.
            name: the name of the robot.
            robot_kwargs: the additional keyword arguments for the robot.

        Returns:
            the body id of the robot.
        """

        robot_kwargs = robot_kwargs or dict()
        if robot == 'ur5':
            self.robot = UR5Robot(self.client, pos=pos, quat=quat, body_name=name if name is not None else 'ur5', **robot_kwargs)
        elif robot == 'panda':
            self.robot = PandaRobot(self.client, pos=pos, quat=quat, body_name=name if name is not None else 'panda', **robot_kwargs)
        else:
            raise ValueError(f'Unknown robot type: {robot}.')

        self.robots.append(self.robot)
        return self.robot.get_body_id()

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
        self, size1_2d: Vec2f, size2_2d: Vec2f, thickness: float, location_2d: Optional[Vec2f] = None, name: str = 't-shape', *,
        static: bool = False,
        pos: Optional[Vec3f] = None,
        quat: Vec4f = (0, 0, 0, 1),
    ) -> int:
        """Add a T-shape object to the environment.

        Args:
            size1_2d: the size of the "handle" part of the T-shape.
            size2_2d: the size of the "top bar" part of the T-shape.
            thickness: the thickness of the T-shape.
        """
        if pos is not None:
            assert location_2d is None, 'Cannot specify both pos and location_2d.'
        else:
            assert location_2d is not None, 'Either pos or location_2d should be specified.'
            pos = (location_2d[0], location_2d[1], thickness / 2)

        return self.client.load_urdf_template(
            'assets://basic/simple-tools/t-shape-template.urdf', {
                'DIMX': (size1_2d[0], size1_2d[1], thickness),
                'HALFX': (size1_2d[0] / 2, size1_2d[1] / 2, thickness / 2),
                'DIMY': (size2_2d[0], size2_2d[1], thickness),
                'HALFY': (size2_2d[0] / 2, size2_2d[1] / 2, thickness / 2),
                'DISP': ((size1_2d[0] + size2_2d[0]) / 2, 0, 0.0)
            },
            pos=pos, quat=quat, body_name=name, static=static
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

    def get_support(self, body_id: int, return_name: bool = True) -> List[Union[str, int]]:
        return _get_support(self, body_id, return_name=return_name)


class SimpleTableTopEnv(TableTopEnv):
    """A simple table-top environment that overrides two methods.

    - :meth:`_reset_scene` is the method to be implemented by the subclass. This function will be called by :meth:`reset`.
      This method should return a dictionary of metainfo of the objects in the environment.
      After the metainfo is returned, the environment will be saved by the :meth:`WorldSaver.save` method, so that the scene can be restored using :meth:`restore`.
    """
    def reset(self):
        with self.client.disable_rendering(disable_rendering=True):
            metainfo = self._reset_scene()

        self.metainfo = metainfo
        self.saver.save()

    def _reset_scene(self) -> dict:
        raise NotImplementedError


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

