#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : tracik_wrapper.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/19/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import math
import numpy as np
from tracikpy import TracIKSolver, MultiTracIKSolver
from concepts.math.rotationlib_xyzw import pos_quat2mat_xyzw
from concepts.utils.typing_utils import Vec3f, Vec4f


class TracIKWrapper(object):
    def __init__(self, ik_solver):
        self.ik_solver = ik_solver
        self.joint_names = self.ik_solver.joint_names

    def solve(self, tool_pose_mat4, seed_conf=None, pos_tolerance: float = 1e-4, ori_tolerance: float = math.radians(5e-2)):
        # will use random seed_conf if seed_conf is None
        bx, by, bz = pos_tolerance * np.ones(3)
        brx, bry, brz = ori_tolerance * np.ones(3)
        conf = self.ik_solver.ik(tool_pose_mat4, qinit=seed_conf, bx=bx, by=by, bz=bz, brx=brx, bry=bry, brz=brz)
        name2conf = {name: j for name, j in zip(self.joint_names, conf)}
        return name2conf

    def solve_closest(self, tool_pose_mat4, seed_conf, **kwargs):
        return self.solve(tool_pose_mat4, seed_conf=seed_conf, **kwargs)

    def solve_multi(self, tool_pose_mat4, seed_conf=None, pos_tolerance: float = 1e-4, ori_tolerance: float = math.radians(5e-2), n_result: int = 20):
        """Compute multiple IK solutions for the given tool pose.

        Args:
            tool_pose_mat4: the target pose of the tool in the base frame.
            seed_conf: the initial joint configuration for the IK solver.
            pos_tolerance: the tolerance for the position of the tool.
            ori_tolerance: the tolerance for the orientation of the tool.
            n_result: the number of results to return.
        """
        bx, by, bz = pos_tolerance * np.ones(3)
        brx, bry, brz = ori_tolerance * np.ones(3)
        is_valid_conf, confs = self.ik_solver.iks(
            np.repeat(tool_pose_mat4[np.newaxis, ...], n_result, axis=0),
            qinits=np.repeat(seed_conf[np.newaxis, ...], n_result, axis=0) if seed_conf is not None else None,
            bx=bx, by=by, bz=bz, brx=brx, bry=bry, brz=brz
        )
        all_name2conf = [{name: j for name, j in zip(self.joint_names, conf)} for (is_valid, conf) in zip(is_valid_conf, confs) if is_valid]
        return all_name2conf


class URDFTracIKWrapper(object):
    def __init__(self, urdf_path, base_link_name: str):
        self.urdf_path = urdf_path
        self.base_link_name = base_link_name
        self.ik_solvers = {}

    def create_tracik_solver(self, tool_link_name, max_time=0.025, error=1e-3):
        tracik_solver = MultiTracIKSolver(
            urdf_file=self.urdf_path,
            base_link=self.base_link_name,
            tip_link=tool_link_name,
            timeout=max_time,
            epsilon=error,
            solve_type='Distance'  # Speed | Distance | Manipulation1 | Manipulation2
        )
        assert tracik_solver.joint_names
        tracik_solver.urdf_file = self.urdf_path
        return tracik_solver

    def create_ik_solver(self, tool_link_name):
        tracik_solver = self.create_tracik_solver(tool_link_name)
        ik_solver = TracIKWrapper(tracik_solver)
        return ik_solver

    def get_ik_solver(self, link_name):
        if link_name not in self.ik_solvers:
            self.ik_solvers[link_name] = self.create_ik_solver(link_name)
        return self.ik_solvers[link_name]

    def gen_ik_mat(self, link_name, link_target_pose, seed_conf=None, return_all=False, n_proposals=20):
        solver = self.get_ik_solver(link_name)
        if seed_conf is not None:
            # TODO(Jiayuan Mao @ 2024/12/20): implement joint name mapping
            raise NotImplementedError

        ik_solutions = solver.solve_multi(link_target_pose, n_result=n_proposals)
        valid_solutions = [ik_solution for ik_solution in ik_solutions if not self.is_self_collision(ik_solution)]
        if len(valid_solutions) == 0:
            return [] if return_all else None

        # sort according to torso changes
        valid_solutions_sorted = sorted(valid_solutions, key=lambda x: np.abs(np.array([qval for (joint_name, qval) in x.items() if 'torso' in joint_name])).sum())

        if return_all:
            return valid_solutions_sorted
        return valid_solutions_sorted[0]

    def gen_ik(self, link_name: str, pos: Vec3f, quat: Vec4f, seed_conf=None, **kwargs):
        tool_pose_mat4 = pos_quat2mat_xyzw(pos, quat)
        return self.gen_ik_mat(link_name, tool_pose_mat4, seed_conf=seed_conf, **kwargs)

    def is_self_collision(self, name2conf):
        # TODO(Jiayuan Mao @ 2024/12/19): implement this function
        return False
