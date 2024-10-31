#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test-panda-ikfast-fk.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/09/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import numpy as np
import jacinle

import concepts.simulator.ikfast.franka_panda.ikfast_panda_arm as ikfast_module
from concepts.simulator.ikfast.ikfast_common import IKFastWrapperBase
from concepts.math.rotationlib_xyzw import quat_mul, quat_conjugate, quat_diff
from concepts.math.frame_utils_xyzw import compose_transformation
from concepts.simulator.pybullet.client import BulletClient
from concepts.simulator.pybullet.components.panda.panda_robot import PandaRobot


ikfast_wrapper = IKFastWrapperBase(
    ikfast_module,
    joint_ids=[0, 1, 2, 3, 4, 5, 6], free_joint_ids=[6], use_xyzw=True,
    joints_lower=[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175,-2.8973],
    joints_upper=[2.8963, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
)

def fk(q):
    pos, quat = ikfast_wrapper.fk(q)
    pos, quat = np.array(pos), np.array(quat)
    dq = [0.00000000, 0.00000000, 0.38268343, -0.92387953]
    quat = quat_mul(quat, dq)
    if quat[3] < 0:
        quat = -quat
    return pos, quat


def fk11(q):
    pos, quat = ikfast_wrapper.fk(q)
    pos, quat = np.array(pos), np.array(quat)
    dp = [0, 0, 0.1]
    dq = [0.0, 0.0, 0.9238795325108381, 0.38268343236617297]
    return compose_transformation(pos, quat, dp, dq)


client = BulletClient(is_gui=False)
robot = PandaRobot(client)


for i in range(100):
    q = np.random.uniform(ikfast_wrapper.joints_lower, ikfast_wrapper.joints_upper)

    x, xq = fk(q)
    y, yq = robot.fk(q, link_name_or_id=8)

    assert np.allclose(x, y, atol=1e-5), (x, y)
    assert quat_diff(xq, yq) < 1e-5, (xq, yq)

print(jacinle.colored('All tests on Link 8 passed.', 'green'))


for i in range(100):
    q = np.random.uniform(ikfast_wrapper.joints_lower, ikfast_wrapper.joints_upper)

    x, xq = fk11(q)
    y, yq = robot.fk(q, link_name_or_id=11)

    assert np.allclose(x, y, atol=1e-5), (x, y)
    assert quat_diff(xq, yq) < 1e-5, (xq, yq)

print(jacinle.colored('All tests on Link 11 passed.', 'green'))
