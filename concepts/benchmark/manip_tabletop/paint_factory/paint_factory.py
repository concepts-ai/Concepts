#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : paint_factory.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/22/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os
import cv2
import torch
import numpy as np
import os.path as osp
import itertools
from typing import Optional, Iterator, Tuple

from concepts.benchmark.manip_tabletop.pybullet_tabletop import TableTopEnv
from concepts.simulator.pybullet.client import BulletClient
from concepts.simulator.pybullet.camera import TopDownOracle, get_point_cloud_image, get_orthographic_heightmap
from concepts.simulator.pybullet.rotation_utils import rpy, quat_mul
from concepts.pdsketch.domain import State
from concepts.pdsketch.executor import PDSketchExecutor
from concepts.utils.typing_utils import Vec2f, Vec4f


__all__ = ['PaintFactoryEnv', 'get_paint_factory_domain_filename']


class PaintFactoryEnv(TableTopEnv):
    def __init__(self, client: Optional[BulletClient] = None, executor: Optional[PDSketchExecutor] = None, seed: int = 1234):
        super().__init__(client, executor, seed)

        self.max_steps = 10
        self.pos_eps = 0.05

        self._topdown_camera = TopDownOracle.get_configs()[0]
        self._pix_size = 0.003125
        self._workspace_bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.3]])

        self._container_info = dict()
        self._block_info = dict()
        self._bowl_info = dict()

        self._target_box = None
        self._clean_box = None
        self._dry_box = None

        self.color_goals, self.relation_goals, self.lang_goals = list(), list(), list()

    def add_bowl(
        self, location_2d: Vec2f, name: str = 'bowl', *,
        static=True, z: float = 0,
        color_rgba: Optional[Vec4f] = None,
        quat: Optional[Vec4f] = None,
    ) -> int:
        if color_rgba is None:
            color_rgba = (1.0, 1.0, 1.0, 1.0)
        return self.client.load_urdf('assets://cliport/bowl/bowl.urdf', (location_2d[0], location_2d[1], z), quat=quat, rgba=color_rgba, body_name=name, static=static)

    def add_cliport_block(
        self, location_2d: Vec2f, name: str = 'box', *,
        static=False, z: float = 0.02,
        color_rgba: Optional[Vec4f] = None,
        quat: Optional[Vec4f] = None,
    ):
        if color_rgba is None:
            color_rgba = (1.0, 1.0, 1.0, 1.0)
        return self.client.load_urdf('assets://cliport/stacking/block.urdf', (location_2d[0], location_2d[1], z), quat=quat, rgba=color_rgba, body_name=name, static=static)

    def iter_objects(self) -> Iterator[Tuple[int, dict]]:
        yield from itertools.chain(self._container_info.items(), self._block_info.items(), self._bowl_info.items())

    def reset(self):
        plane_id = self.client.load_urdf('assets://cliport/plane/plane.urdf', (0, 0, -0.001), static=True, body_name='table')
        self.client.load_urdf('assets://cliport/workspace/workspace.urdf', (0.5, 0, 0), static=True, body_name='workspace')

        robot_id = self.add_robot('ur5')

        self._container_info = dict()
        self._block_info = dict()
        self._bowl_info = dict()

        def add_container(name: str, color=None, zone_size=None):
            if zone_size is None:
                zone_size = self._get_random_size(0.1, 0.2, 0.1, 0.2, 0.05, 0.05)
            pos, quat = self._get_random_pose(zone_size)
            if color is not None:
                color = g_colors[color]
                color = (color[0], color[1], color[2], 1)
            print('adding container', zone_size, pos, quat, color)
            box_id = self.add_container(size_2d=zone_size[:2], depth=zone_size[2], location_2d=pos[:2], quat=quat, color_rgba=color, name=name)
            self._container_info[box_id] = {'pos': pos, 'quat': quat, 'size': zone_size, 'name': name}
            return box_id

        self._target_box = add_container('target-box', 'brown', zone_size=(0.3, 0.3, 0.02))
        self._clean_box = add_container('clean-box', 'blue', zone_size=(0.1, 0.1, 0.02))
        self._dry_box = add_container('dry-box', 'red', zone_size=(0.1, 0.1, 0.02))

        nr_bowls, nr_blocks = 3, 3
        all_color_names = ['red', 'green', 'orange', 'yellow', 'purple', 'pink', 'cyan', 'brown']

        # Add bowls.
        bowl_size = (0.12, 0.12, 0)
        bowl_color_names = self.np_random.choice(all_color_names, nr_bowls, replace=False)
        for i in range(nr_bowls):
            pos, quat = self._get_random_pose(bowl_size)
            bowl_id = self.add_bowl((pos[0], pos[1]), name=f'bowl{i}', color_rgba=g_colors[bowl_color_names[i]] + [1], quat=quat)
            self._bowl_info[bowl_id] = {'pos': pos[0], 'quat': quat, 'size': bowl_size, 'color': bowl_color_names[i], 'name': f'bowl{i}'}

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        for i in range(nr_blocks):
            pos, quat = self._get_random_pose(block_size)

            # Randomly select a color.
            color_id = self.np_random.randint(0, 4)  # clean, wet, dirty, dirty+wet
            # color_id = 0  # all things white
            if color_id == 0:
                color_name = 'white'
                color = g_colors['white']
            elif color_id == 1:
                color_name = 'wet'
                color = g_colors['blue']
            elif color_id == 2:
                color_name = 'dirty'
                color = g_colors['gray']
            elif color_id == 3:
                color_name = 'dirty+wet'
                color = _mixing_colors('blue', 'gray')
            else:
                raise ValueError('Unknown color id: {}'.format(color_id))

            block_id = self.add_cliport_block((pos[0], pos[1]), name=f'block{i}', color_rgba=list(color) + [1], quat=quat)
            self._block_info[block_id] = {'pos': pos, 'quat': quat, 'size': block_size, 'color': color_name, 'name': f'block{i}'}

        # Goal: put each block in a different bowl.
        bowl_colors = list(set(self._bowl_info[bowl_id]['color'] for bowl_id in self._bowl_info))
        self.color_goals = list(self.np_random.choice(bowl_colors, size=nr_blocks, replace=False))

        all_relations = ['left of', 'right of', 'on']
        relations = self.np_random.choice(all_relations, size=2, replace=False)
        self.relation_goals = [ (1, 0, relations[0]), (2, 0, relations[1]) ]

        block_names = [self.color_goals[i] + ' block' for i in range(nr_blocks)]
        self.lang_goals = list()
        self.lang_goals.append('; '.join([
            f'put a {block_names[x]} in the brown box' for x in [0]
        ] + [
            f'put a {block_names[x]} {rel} a {block_names[y]}' for x, y, rel in self.relation_goals
        ]))

        # Only one mistake allowed.
        self.max_steps = nr_blocks * 5

        if self.client.is_gui:
            target = self.p.getDebugVisualizerCamera()[11]
            self.p.resetDebugVisualizerCamera(cameraDistance=1.1, cameraYaw=90, cameraPitch=-25, cameraTargetPosition=target)

        self.client.step(10)

    def get_pds_state(self) -> State:
        t_item = self.executor.domain.types['item']
        objects = dict()

        objects['robot'] = self.executor.domain.types['robot']
        for _, item_info in self.iter_objects():
            objects[item_info['name']] = t_item

        state, ctx = self.executor.new_state(objects, create_context=True)

        ctx.define_feature('robot-qpos', torch.tensor([self.robot.get_qpos()], dtype=torch.float32))
        ctx.set_value('robot-identifier', ['robot'], self.robot.get_robot_body_id())

        for item_id, item_info in self.iter_objects():
            ctx.set_value('item-pose', [item_info['name']], self.world.get_body_state_by_id(item_id).get_7dpose())
            ctx.set_value('item-identifier', [item_info['name']], item_id)

            if item_info['name'].startswith('block'):
                ctx.set_value('moveable', [item_info['name']], True)

            # for name2 in _get_support(env, item_id):
            #     ctx.set_value('support', [item_info['name'], name2], True)

        # TODO(Jiayuan): add gripper constraint.
        # if env.robot.gripper_constraint is None:
        #     ctx.define_predicates([ctx.robot_hands_free('robot')])
        # else:
        #     constraint = env.world.get_constraint(env.robot.gripper_constraint)
        #     name = env.world.body_names[constraint.child_body]
        #     ctx.define_predicates([ctx.robot_holding_item('robot', name)])

        return state

    def _get_random_pose(self, obj_size):
        """Get random collision-free object pose within workspace bounds."""

        # Get erosion size of the object in pixels.
        max_size = np.sqrt(obj_size[0] ** 2 + obj_size[1] ** 2)
        erode_size = int(np.round(max_size / self._pix_size))

        color, depth, segm = self.world.render_image(self._topdown_camera, normalize_depth=True)
        pcd_image = get_point_cloud_image(self._topdown_camera, depth)
        ortho_height, ortho_color, ortho_segm = get_orthographic_heightmap(pcd_image, color, self._workspace_bounds, self._pix_size, segmentation=segm)

        # Randomly sample an object pose within free-space pixels.
        free = np.ones(ortho_segm.shape, dtype=np.uint8)
        free[ortho_segm > 0] = 0
        free[0, :], free[:, 0], free[-1, :], free[:, -1] = 0, 0, 0, 0
        free = cv2.erode(free, np.ones((erode_size, erode_size), np.uint8))
        if np.sum(free) == 0:
            return None, None

        pix = self._sample_distribution(np.float32(free))
        pos = _pix_to_xyz(pix, ortho_height, self._workspace_bounds, self._pix_size)
        pos = (pos[0], pos[1], obj_size[2] / 2)
        theta = self.np_random.rand() * 360
        rot = rpy(0, 0, theta, degree=True)
        return pos, rot

    def _get_random_size(self, min_x, max_x, min_y, max_y, min_z, max_z):
        """Get random box size."""
        size = self.np_random.rand(3)
        size[0] = size[0] * (max_x - min_x) + min_x
        size[1] = size[1] * (max_y - min_y) + min_y
        size[2] = size[2] * (max_z - min_z) + min_z
        return tuple(size)

    def _sample_distribution(self, prob, n_samples=1):
        """Sample data point from a custom distribution."""
        flat_prob = prob.flatten() / np.sum(prob)
        rand_ind = self.np_random.choice(
            np.arange(len(flat_prob)), n_samples, p=flat_prob, replace=False
        )
        rand_ind_coords = np.array(np.unravel_index(rand_ind, prob.shape)).T
        return np.int32(rand_ind_coords.squeeze())

    def step(self, pick_location: Vec2f, place_location: Vec2f, place_orientation: float):
        from concepts.simulator.pybullet.components.ur5.ur5_robot import UR5Robot
        robot: UR5Robot = self.robot

        current_ee_quat = np.array((0, 0, 0, 1), dtype=np.float32)
        pick_location = (pick_location[0], pick_location[1], 0)
        robot.reach_and_pick(pick_location, current_ee_quat)
        place_location = (place_location[0], place_location[1], 0)
        place_quat = quat_mul(current_ee_quat, rpy(0, 0, place_orientation, degree=False))
        robot.reach_and_place(place_location, place_quat)

        self.scene_step()
        return True

    def step_object_name(self, pick_object, place_object, additional_pick_offset=None):
        pos1 = self.world.get_body_state(pick_object).pos
        pos2 = self.world.get_body_state(place_object).pos
        if additional_pick_offset is not None:
            return self.step((pos1[0] + additional_pick_offset[0], pos1[1] + additional_pick_offset[1]), (pos2[0], pos2[1]), 0)
        return self.step((pos1[0], pos1[1]), (pos2[0], pos2[1]), 0)

    def scene_step(self):
        for block_id, info in self._block_info.items():
            support_objects = self.get_support(block_id)
            for support_object_name in support_objects:
                if support_object_name == 'clean-box':
                    self._block_info[block_id]['color'] = 'wet'
                    self.world.change_visual_color(block_id, g_colors['blue'] + [1])
                elif support_object_name == 'dry-box':
                    if info['color'] == 'wet':
                        self._block_info[block_id]['color'] = 'white'
                        self.world.change_visual_color(block_id, g_colors['white'] + [1])
                    elif info['color'] == 'dirty+wet':
                        self._block_info[block_id]['color'] = 'dirty'
                        self.world.change_visual_color(block_id, g_colors['gray'] + [1])
                elif support_object_name.startswith('bowl'):
                    if info['color'] not in ('dirty', 'wet', 'dirty+wet'):
                        bowl_id = self.world.get_body_index(support_object_name)
                        self._block_info[block_id]['color'] = self._bowl_info[bowl_id]['color']
                        self.world.change_visual_color(block_id, g_colors[self._bowl_info[bowl_id]['color']] + [1])


def get_paint_factory_domain_filename() -> str:
    """Get the domain filename of the crafting world."""
    return osp.join(osp.dirname(__file__), f'paint-factory-domain-v20231225.pdsketch')


g_colors = {
    'blue': [078.0 / 255.0, 121.0 / 255.0, 167.0 / 255.0],
    'red': [255.0 / 255.0, 087.0 / 255.0, 089.0 / 255.0],
    'green': [089.0 / 255.0, 169.0 / 255.0, 079.0 / 255.0],
    'orange': [242.0 / 255.0, 142.0 / 255.0, 043.0 / 255.0],
    'yellow': [237.0 / 255.0, 201.0 / 255.0, 072.0 / 255.0],
    'purple': [176.0 / 255.0, 122.0 / 255.0, 161.0 / 255.0],
    'pink': [255.0 / 255.0, 157.0 / 255.0, 167.0 / 255.0],
    'cyan': [118.0 / 255.0, 183.0 / 255.0, 178.0 / 255.0],
    'brown': [156.0 / 255.0, 117.0 / 255.0, 095.0 / 255.0],
    'white': [255.0 / 255.0, 255.0 / 255.0, 255.0 / 255.0],
    'gray': [186.0 / 255.0, 176.0 / 255.0, 172.0 / 255.0],
}


def _mixing_colors(*colors: str):
    return np.mean(np.array([g_colors[c] for c in colors]), axis=0)


def _pix_to_xyz(pixel, height, bounds, pixel_size, skip_height=False):
    """Convert from pixel location on heightmap to 3D position."""
    u, v = pixel
    x = bounds[0, 0] + v * pixel_size
    y = bounds[1, 0] + u * pixel_size
    z = 0.0 if skip_height else bounds[2, 0] + height[u, v]
    return x, y, z

