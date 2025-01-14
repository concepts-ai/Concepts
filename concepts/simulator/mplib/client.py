#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : client.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/09/2025
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os.path as osp
import contextlib

import numpy as np
from typing import Optional, Union, Literal, Sequence, NamedTuple
from functools import partial

from concepts.math.rotationlib_xyzw import xyzw2wxyz
from mplib.collision_detection import WorldCollisionResult
from mplib.collision_detection.fcl import CollisionGeometry, FCLObject
from mplib.planning.ompl import FixedJoint, OMPLPlanner
from mplib.pymp import ArticulatedModel, PlanningWorld, Pose
from mplib.urdf_utils import generate_srdf, replace_urdf_package_keyword


class RobotSpec(NamedTuple):
    urdf_filename: str
    ee_link_name: str
    srdf_filename: Optional[str] = None
    new_package_keyword: str = ""
    use_convex: bool = False
    joint_vel_limits: Optional[Sequence[float]] = None
    joint_acc_limits: Optional[Sequence[float]] = None


class MPLibRobot(object):
    def __init__(self, name: str, urdf_filename: str, ee_link_name: str, srdf_filename: Optional[str] = None, new_package_keyword: str = "", use_convex: bool = False, joint_vel_limits: Optional[Sequence[float]] = None, joint_acc_limits: Optional[Sequence[float]] = None, verbose: bool = False):
        if srdf_filename is None:
            if osp.exists(srdf_filename := urdf_filename.replace(".urdf", ".srdf")):
                print(f"No SRDF file provided but found {srdf_filename}")
            elif osp.exists(srdf_filename := urdf_filename.replace(".urdf", "_mplib.srdf")):
                print(f"No SRDF file provided but found {srdf_filename}")
            else:
                srdf_filename = generate_srdf(urdf_filename, new_package_keyword, verbose=True)

        self.name = name
        self.urdf_filename = urdf_filename
        self.ee_link_name = ee_link_name
        self.srdf_filename = srdf_filename
        self.new_package_keyword = new_package_keyword
        self.use_convex = use_convex
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits
        self.verbose = verbose

        if new_package_keyword != '':
            raise NotImplementedError()

        self.robot = ArticulatedModel(
            str(self.urdf_filename),
            str(self.srdf_filename if self.srdf_filename is not None else ''),
            name=name,
            convex=use_convex,
            verbose=self.verbose,
        )

        self.pinocchio_model = self.robot.get_pinocchio_model()
        self.user_link_names = self.pinocchio_model.get_link_names()
        self.user_joint_names = self.pinocchio_model.get_joint_names()

        self.joint_name2idx = {joint: i for i, joint in enumerate(self.user_joint_names)}
        self.link_name2idx = {link: i for i, link in enumerate(self.user_link_names)}

        assert self.ee_link_name in self.user_link_names, f"end-effector not found as one of the links in {self.user_link_names}"
        self.ee_link_index = self.link_name2idx[self.ee_link_name]

        self.robot.set_move_group(self.ee_link_name)
        self.ee_joint_indices = self.robot.get_move_group_joint_indices()

        self._planning_world = None

    @classmethod
    def from_pb_robot(cls, robot):
        rv = cls(robot.get_body_name(), robot.get_urdf_filename(), robot.get_ee_link_name())
        rv.set_base_pose(*robot.get_body_pose())
        return rv

    def set_planning_world(self, planning_world):
        self._planning_world = planning_world

    def pad_move_group_qpos(self, qpos):
        """
        If qpos contains only the move_group joints, return qpos padded with
        current values of the remaining joints of articulation.
        Otherwise, verify number of joints and return.

        :param qpos: joint positions
        :param articulation: the articulation to get qpos from. If None, use self.robot
        :return: joint positions with full dof
        """
        if (ndim := len(qpos)) == self.robot.get_move_group_qpos_dim():
            tmp = self.robot.get_qpos().copy()
            tmp[:ndim] = qpos
            qpos = tmp
        return qpos

    def set_base_pose(self, pos, quat):
        quat = quat[[3, 0, 1, 2]]
        self.robot.set_base_pose(Pose(p=pos, q=quat))

    def get_qpos(self):
        return self.robot.get_qpos()[:self.robot.get_move_group_qpos_dim()]

    def set_qpos(self, qpos):
        self.robot.set_qpos(qpos)

    def set_full_qpos(self, qpos):
        self.robot.set_qpos(qpos, True)

    @contextlib.contextmanager
    def set_qpos_context(self, qpos):
        old_qpos = self.get_qpos()
        self.set_qpos(qpos)
        yield
        self.set_qpos(old_qpos)


class MPLibClient(object):
    def __init__(self, robots: Sequence[MPLibRobot], objects: Sequence[FCLObject] = tuple(), verbose: bool = False):
        self.robots = robots

        self.planning_world = PlanningWorld([r.robot for r in robots], list(objects))
        self.acm = self.planning_world.get_allowed_collision_matrix()
        self.verbose = verbose

        for robot in self.robots:
            robot.set_planning_world(self.planning_world)

    def get_robot(self, name: str) -> MPLibRobot:
        for robot in self.robots:
            if robot.name == name:
                return robot
        raise ValueError(f"Robot {name} not found in the client.")

    def get_object(self, name: str) -> FCLObject:
        return self.planning_world.get_object(name)

    def get_articulation(self, name: str) -> ArticulatedModel:
        return self.planning_world.get_articulation(name)

    def has_object(self, name: str, allow_articulation: bool = True):
        if allow_articulation:
            return self.planning_world.has_object(name) or self.planning_world.has_articulation(name)
        return self.planning_world.has_object(name)

    def has_articulation(self, name: str):
        return self.planning_world.has_articulation(name)

    def set_object_pose(self, name, pos, quat):
        if self.planning_world.has_object(name):
            fcl_object = self.planning_world.get_object(name)
            fcl_object.set_pose(Pose(p=pos, q=xyzw2wxyz(quat)))
        elif self.planning_world.has_articulation(name):
            robot = self.planning_world.get_articulation(name)
            robot.set_base_pose(Pose(p=pos, q=xyzw2wxyz(quat)))
        else:
            raise ValueError(f"Object {name} not found in the world.")

    @contextlib.contextmanager
    def set_qpos_context(self, qposes: dict[str, np.ndarray]):
        old_qposes = {robot.name: robot.get_qpos() for robot in self.robots}
        for robot in self.robots:
            robot.set_qpos(qposes[robot.name])
        yield
        for robot in self.robots:
            robot.set_qpos(old_qposes[robot.name])

    def check_for_collision(self, collision_function, states: Optional[dict[str, np.ndarray]] = None) -> list[WorldCollisionResult]:
        """
        Helper function to check for collision

        :param state: all planned articulations qpos state. If None, use current qpos.
        :return: A list of collisions.
        """
        if states is None:
            return collision_function()

        with self.set_qpos_context(states):
            rv = collision_function()
            return rv

    def check_for_self_collision(self, state: Optional[dict[str, np.ndarray]] = None) -> list[WorldCollisionResult]:
        """
        Check if the robot is in self-collision.

        :param state: all planned articulations qpos state. If None, use current qpos.
        :return: A list of collisions.
        """
        return self.check_for_collision(self.planning_world.check_self_collision, state)

    def check_for_env_collision(self, state: Optional[dict[str, np.ndarray]] = None) -> list[WorldCollisionResult]:
        """
        Check if the robot is in collision with the environment

        :param state: all planned articulations qpos state. If None, use current qpos.
        :return: A list of collisions.
        """
        return self.check_for_collision(self.planning_world.check_robot_collision, state)

    def check_for_general_collision(self, name: str, state: Optional[dict[str, np.ndarray]] = None) -> list[WorldCollisionResult]:
        """
        Check if the robot is in collision with the environment

        :param state: all planned articulations qpos state. If None, use current qpos.
        :return: A list of collisions.
        """
        # print(f'Checking for general collision {name=}')
        # for robot in self.robots:
        #     print(f'Robot {robot.name=} qpos={robot.get_qpos()}')
        # print(f'Object name=object_3 pose={self.get_object("object_3").pose}')
        return self.check_for_collision(partial(self.planning_world.check_general_object_collision, name), state)

    def check_for_general_pair_collision(self, a: str, b: str, state: Optional[dict[str, np.ndarray]] = None) -> list[WorldCollisionResult]:
        """
        Check if the robot is in collision with the environment

        :param state: all planned articulations qpos state. If None, use current qpos.
        :return: A list of collisions.
        """
        return self.check_for_collision(partial(self.planning_world.check_general_object_pair_collision, a, b), state)

    def update_point_cloud(self, points, pos = (0, 0, 0), quat = (0, 0, 0, 1), resolution=1e-3, name="scene_pcd"):
        """
        Adds a point cloud as a collision object with given name to world.
        If the ``name`` is the same, the point cloud is simply updated.

        :param points: points, numpy array of shape (n, 3)
        :param resolution: resolution of the point OcTree
        :param name: name of the point cloud collision object
        """
        self.planning_world.add_point_cloud(name, points, resolution, Pose(p=pos, q=xyzw2wxyz(quat)))

    def remove_point_cloud(self, name="scene_pcd") -> bool:
        """
        Removes the point cloud collision object with given name

        :param name: name of the point cloud collision object
        :return: ``True`` if success, ``False`` if the non-articulation object
            with given name does not exist
        """
        return self.planning_world.remove_object(name)

    def add_object(self, fcl_object):
        """adds an object to the world"""
        return self.planning_world.add_object(fcl_object)

    def remove_object(self, name) -> bool:
        """returns true if the object was removed, false if it was not found"""
        return self.planning_world.remove_object(name)

    def sync_object_states(self, pb_client):
        for index, name in pb_client.world.get_body_id_and_names():
            if self.has_object(name):
                pos, quat = pb_client.world.get_body_state_by_id(index).get_transformation()
                self.set_object_pose(name, pos, quat)

            if self.has_articulation(name):
                qpos_indices = self.get_articulation(name).get_move_group_joint_indices()
                qpos = pb_client.world.get_batched_qpos_by_id(index, qpos_indices)
                self.get_articulation(name).set_qpos(qpos)
