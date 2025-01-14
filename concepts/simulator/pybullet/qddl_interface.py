#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : qddl_interface.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/29/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import tempfile
import os.path as osp
from typing import Any, Optional, Tuple, List, Dict
from dataclasses import dataclass, field

import numpy as np
import pybullet as p
import lisdf.components as C
from lisdf.parsing.qddl import load_qddl, load_qddl_string

from concepts.simulator.pybullet.client import BulletClient

__all__ = ['PyBulletQDDLInterface', 'QDDLSceneMetainfo', 'QDDLSceneObjectMetainfo']



@dataclass
class QDDLSceneObjectMetainfo(object):
    id: int
    color: Optional[Tuple[float, float, float, float]] = None
    moveable: bool = True
    additional_attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QDDLSceneMetainfo(object):
    objects: Dict[str, QDDLSceneObjectMetainfo] = field(default_factory=dict)
    robots: List[Any] = field(default_factory=list)

    def get_object_identifier(self, object_name: str) -> int:
        return self.objects[object_name].id


@dataclass
class QDDLProblemMetainfo(object):
    goal: str


class PyBulletQDDLInterface(object):
    """Load a scene from a QDDL problem file."""

    def __init__(self, client: BulletClient, package_map: Optional[Dict[str, str]] = None):
        self.client = client
        self.package_map = package_map or dict()
        self.package_map['concepts'] = self.client.assets_root

    def load(self, domain_file: str, problem_file: str) -> Tuple[QDDLSceneMetainfo, QDDLProblemMetainfo]:
        _, problem = self.load_qddl(domain_file, problem_file)
        return self._load_scene(problem, self.package_map), self._load_problem_metainfo(problem)

    def load_qddl(self, domain_file: str, problem_file: str) -> Tuple[C.PDDLDomain, C.PDDLProblem]:
        return load_qddl(domain_file, problem_file)

    def load_scene(self, domain_file, problem_file) -> QDDLSceneMetainfo:
        _, problem = load_qddl(domain_file, problem_file)
        return self._load_scene(problem, self.package_map)

    def load_scene_string(self, domain_string, problem_string) -> QDDLSceneMetainfo:
        _, problem = load_qddl_string(domain_string, problem_string)
        return self._load_scene(problem, self.package_map)

    def _load_problem_metainfo(self, problem) -> QDDLProblemMetainfo:
        goals = list()
        for x in problem.conjunctive_goal:
            name = x.predicate.name
            args = [y.name for y in x.arguments]
            goals.append(f"{name}({', '.join(args)})")
        return QDDLProblemMetainfo(goal=" and ".join(goals))

    def _load_scene(self, problem, package_map: Dict[str, str]) -> QDDLSceneMetainfo:
        objects = dict()
        boxes = list()
        metainfo = QDDLSceneMetainfo()

        for name, obj in problem.objects.items():
            # Ignore "purely symbolic" entities and the special world-type.
            if obj.type is None:
                continue

            if obj.type.identifier == "world-type":
                continue

            if obj.type.identifier == "box-type" and obj.type.scope == "qrgeom":
                boxes.append(name)
                continue

            url = obj.type.url.value

            if url.startswith("package://"):
                for package, path in package_map.items():
                    if url.startswith(f"package://{package}/"):
                        url = url.replace(f"package://{package}", path)

            objects[name] = url

        static = dict()
        poses = dict()
        scales = dict()
        urdf_props = dict()
        urdf_load_args = dict()
        joint_configs = dict()
        box_shapes = dict()
        box_colors = dict()
        debug_camera_poses = {
            'distance': 10,
            'yaw': 0,
            'pitch': -45,
            'target': [0, 0, 0],
        }

        for v in problem.init:
            if v.predicate.name == "body-pose":
                body_name = v.arguments[0].name
                pose = _get_pose_from_value(v.arguments[1].value)
                poses[body_name] = pose
            elif v.predicate.name == "body-scale":
                body_name = v.arguments[0].name
                scale = v.arguments[1].value
                scales[body_name] = scale
            elif v.predicate.name == "weld":
                parent = v.arguments[0].name
                child = v.arguments[1].name
                assert parent == "world::world"

                if "::" in child:
                    print(f"WARNING: Welding to a non-base-link is not supported. Currently assuming {child} is the base link for {child.split()[0]}.")
                    child = child.split("::")[0]

                static[child] = True
                pose = _get_pose_from_value(v.arguments[2].value)
                poses[child] = pose
            elif v.predicate.name == "joint-conf":
                joint_name = v.arguments[0].name
                joint_conf = v.arguments[1].value

                body_name, joint_name = joint_name.split("::")
                if body_name not in joint_configs:
                    joint_configs[body_name] = dict()
                joint_configs[body_name][joint_name] = joint_conf
            elif v.predicate.pddl_name == "qrgeom::box-shape":
                box_name = v.arguments[0].name
                size = v.arguments[1].value
                box_shapes[box_name] = size
            elif v.predicate.pddl_name == "qrgeom::box-color":
                box_name = v.arguments[0].name
                color = v.arguments[1].value
                box_colors[box_name] = color
            elif v.predicate.pddl_name == 'urdf::prop':
                body_name = v.arguments[0].name
                prop_name = v.arguments[1].value
                prop_value = v.arguments[2].value
                if isinstance(prop_value, np.ndarray):
                    prop_value = prop_value.tolist()
                if body_name not in urdf_props:
                    urdf_props[body_name] = dict()
                urdf_props[body_name][prop_name] = prop_value
            elif v.predicate.pddl_name == 'urdf::load-arg':
                body_name = v.arguments[0].name
                arg_name = v.arguments[1].value
                arg_value = v.arguments[2].value
                if isinstance(arg_value, np.ndarray):
                    arg_value = arg_value.tolist()
                if body_name not in urdf_load_args:
                    urdf_load_args[body_name] = dict()
                urdf_load_args[body_name][arg_name] = arg_value
            elif v.predicate.pddl_name == 'sim::camera-distance':
                debug_camera_poses['distance'] = float(v.arguments[0].value)
            elif v.predicate.pddl_name == 'sim::camera-yaw':
                debug_camera_poses['yaw'] = float(v.arguments[0].value)
            elif v.predicate.pddl_name == 'sim::camera-pitch':
                debug_camera_poses['pitch'] = float(v.arguments[0].value)
            elif v.predicate.pddl_name == 'sim::camera-look-at':
                debug_camera_poses['target'] = list(v.arguments[0].value)
            else:
                print(f"WARNING: Unknown predicate {v.predicate.pddl_name}. Ignoring.")

        with tempfile.TemporaryDirectory() as tmp_dirname:
            with self.client.disable_stdout():
                for box_name in boxes:
                    filename = osp.join(tmp_dirname, f"{box_name}.sdf")
                    _export_box_urdf(filename, box_shapes[box_name], box_colors.get(box_name, (1, 0, 0, 1)))
                    box_id = self.client.load_urdf(filename, pos=poses[box_name][0], quat=poses[box_name][1], static=static.get(box_name, False))

                    metainfo.objects[box_name] = QDDLSceneObjectMetainfo(box_id, color=box_colors.get(box_name, (1, 0, 0, 1)), moveable=not static.get(box_name, False))

            for object_name, object_url in objects.items():
                with open(object_url, "r") as f:
                    xml = f.read()

                if "package://" in xml:
                    for name, path in package_map.items():
                        xml = xml.replace(f"package://{name}", path)
                    tmp_url = osp.join(tmp_dirname, osp.basename(object_url))
                    with open(tmp_url, "w") as f:
                        f.write(xml)
                else:
                    tmp_url = object_url

                if object_url.endswith(".urdf"):
                    print(object_url)
                    # with suppress_stdout():
                    urdf_load_kwargs = urdf_load_args.get(object_name, dict())
                    for protected_field in ['pos', 'quat', 'scale', 'static']:
                        if protected_field in urdf_load_kwargs:
                            print(f"WARNING: {protected_field} is a protected field for URDF loading. Use the corresponding QDDL predicate instead. Ignoring.")
                            del urdf_load_kwargs[protected_field]
                    if object_name in urdf_props:
                        object_id = self.client.load_urdf_template(
                            tmp_url, urdf_props[object_name],
                            body_name=object_name,
                            pos=poses[object_name][0] if object_name in poses else (0, 0, 0), quat=poses[object_name][1] if object_name in poses else (0, 0, 0, 1),
                            scale=scales.get(object_name, 1.0), static=static.get(object_name, False),
                            **urdf_load_kwargs
                        )
                    else:
                        object_id = self.client.load_urdf(
                            tmp_url, pos=poses[object_name][0] if object_name in poses else (0, 0, 0), quat=poses[object_name][1] if object_name in poses else (0, 0, 0, 1),
                            body_name=object_name,
                            scale=scales.get(object_name, 1.0), static=static.get(object_name, False),
                            **urdf_load_kwargs
                        )
                elif object_url.endswith(".sdf"):
                    object_ids = self.client.load_sdf(tmp_url, scale=scales.get(object_name, 1.0))

                    if object_name in static:
                        print("WARNING: Static SDFs are not supported. Ignoring static flag. You should specify this in the SDF file.")

                    object_id = object_ids[0]
                    if object_name in poses:
                        p.resetBasePositionAndOrientation(object_id, poses[object_name][0], poses[object_name][1], physicsClientId=self.client.client_id)
                else:
                    raise ValueError(f"Unknown file extension for {object_url}")

                if object_name in joint_configs:
                    for joint_name, joint_conf in joint_configs[object_name].items():
                        # NB(Jiayuan Mao @ 2024/04/1): the simulator in concepts use "/" as the delimiter for joint names.
                        self.client.world.set_joint_state2(f'{object_name}/{joint_name}', joint_conf)

                metainfo.objects[object_name] = QDDLSceneObjectMetainfo(object_id, moveable=not static.get(object_name, False))

        self.client.world.set_debug_camera(**debug_camera_poses)
        return metainfo


def _export_box_urdf(filename, size, color):
    color = tuple(str(float(x)) for x in color)
    size = tuple(str(float(x)) for x in size)
    with open(filename, "w") as f:
        f.write(
            f"""<?xml version="1.0"?>
<robot name="box">
    <!-- Colors -->
    <material name="boxcolor">
        <color rgba="{' '.join(color)}"/>
    </material>

    <!-- Plane -->
    <link name="box">
        <visual>
            <geometry>
                <box size="{' '.join(size)}"/>
                <origin rpy="0 0 0" xyz="0 0 0"/>
            </geometry>
            <material name="boxcolor"/>
        </visual>
        <collision>
             <geometry>
                <box size="{' '.join(size)}"/>
                <origin rpy="0 0 0" xyz="0 0 0"/>
            </geometry>
        </collision>
        <inertial>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <mass value="0.0"/>
            <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
        </inertial>
    </link>
</robot>""")



def _get_pose_from_value(pose):
    pos, rpy = pose[:3], pose[3:]
    return tuple(pos), p.getQuaternionFromEuler(rpy)
