#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : quickaccess.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/11/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import numpy as np
import concepts.simulator.ikfast.franka_panda.ikfast_panda_arm as ikfast_module
from concepts.simulator.ikfast.ikfast_common import IKFastWrapperBase
from concepts.math.rotationlib_xyzw import quat_mul
from concepts.math.frame_utils_xyzw import compose_transformation


def get_franka_panda_ikfast():
    ikfast_wrapper = IKFastWrapperBase(
        ikfast_module,
        joint_ids=[0, 1, 2, 3, 4, 5, 6], free_joint_ids=[6], use_xyzw=True,
        joints_lower=[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175,-2.8973],
        joints_upper=[2.8963, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
    )
    return ikfast_wrapper


def franka_panda_fk(wrapper: IKFastWrapperBase, q: np.ndarray, link_id: int = 8):
    pos, quat = wrapper.fk(q)
    if link_id == 8:
        pos, quat = np.array(pos), np.array(quat)
        dq = [0.00000000, 0.00000000, 0.38268343, -0.92387953]
        quat = quat_mul(quat, dq)
        return pos, quat
    elif link_id == 11:
        pos, quat = np.array(pos), np.array(quat)
        dp = [0, 0, 0.1]
        dq = [0.0, 0.0, 0.9238795325108381, 0.38268343236617297]
        return compose_transformation(pos, quat, dp, dq)
    else:
        raise ValueError(f'Unsupported link id: {link_id}')
