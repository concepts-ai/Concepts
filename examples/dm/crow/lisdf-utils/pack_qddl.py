#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : pack_qddl.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/15/2025
#
# This file is part of LISDF.
# Distributed under terms of the MIT license.

"""Tar a QDDL file and a list of URDF files into a single directory and removes all template URDF stuff.

Usage:

```bash
python pack_qddl.py domain.pddl problem.pddl --output <output_dir> --packages ~/Projects/Concepts/concepts/assets
```
"""

import os
import os.path as osp
import contextlib
import numpy as np
import argparse
import lisdf.components.pddl as C
from lisdf.parsing.qddl import load_qddl, get_default_qddl_domain


@contextlib.contextmanager
def wrapped_open_w(filename):
    print('Writing to: "{}" ...'.format(filename))
    with open(filename, 'w') as f:
        yield f


def _transform_tree(problem, package_map: dict[str, str], output_dir: str):
    objects = dict()
    boxes = list()

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

    urdf_props = dict()
    urdf_load_args = dict()
    box_shapes = dict()
    box_colors = dict()

    new_init = list()
    for v in problem.init:
        if v.predicate.name == "body-pose":
            new_init.append(v)
        elif v.predicate.name == "body-scale":
            new_init.append(v)
        elif v.predicate.name == "weld":
            new_init.append(v)
        elif v.predicate.name == "joint-conf":
            new_init.append(v)
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
            print(f"WARNING: URDF load arguments are not supported. Ignoring.")
        elif v.predicate.pddl_name == 'sim::camera-distance':
            new_init.append(v)
        elif v.predicate.pddl_name == 'sim::camera-yaw':
            new_init.append(v)
        elif v.predicate.pddl_name == 'sim::camera-pitch':
            new_init.append(v)
        elif v.predicate.pddl_name == 'sim::camera-look-at':
            new_init.append(v)
        else:
            print(f"WARNING: Unknown predicate {v.predicate.pddl_name}. Ignoring.")

    for box_name in boxes:
        filename = osp.join(output_dir, f"{box_name}-type.sdf")
        _export_box_urdf(filename, box_shapes[box_name], box_colors.get(box_name, (1, 0, 0, 1)))

        object_type = C.PDDLObjectType(f"{box_name}-type", C.PDDLLiteral(f"{box_name}-type.sdf"))
        problem.domain.object_types[object_type.identifier] = object_type
        problem.objects[box_name].type = object_type

    for object_name, object_url in objects.items():
        if object_url.endswith(".urdf"):
            urdf_load_kwargs = urdf_load_args.get(object_name, dict())
            for protected_field in ['pos', 'quat', 'scale', 'static']:
                if protected_field in urdf_load_kwargs:
                    print(f"WARNING: {protected_field} is a protected field for URDF loading. Use the corresponding QDDL predicate instead. Ignoring.")
                    del urdf_load_kwargs[protected_field]

            if object_name in urdf_props:
                replaces = urdf_props[object_name]
                with open(object_url) as f:
                    xml_content = f.read()

                for k, v in sorted(replaces.items(), key=lambda x: len(x[0]), reverse=True):
                    if isinstance(v, (tuple, list)):
                        for i in range(len(v)):
                            xml_content = xml_content.replace(k + str(i), str(v[i]))
                    else:
                        xml_content = xml_content.replace(k, str(v))

                new_url = osp.join(output_dir, f'{object_name}-type.urdf')
                with wrapped_open_w(new_url) as f:
                    f.write(xml_content)

                object_type = C.PDDLObjectType(f'{object_name}-type', C.PDDLLiteral(f'{object_name}-type.urdf'))
                problem.domain.object_types[object_type.identifier] = object_type
                problem.objects[object_name].type = object_type
        elif object_url.endswith(".sdf"):
            pass
        else:
            raise ValueError(f"Unknown file extension for {object_url}")

    problem.init = new_init


def _export_box_urdf(filename, size, color):
    color = tuple(str(float(x)) for x in color)
    size = tuple(str(float(x)) for x in size)
    with wrapped_open_w(filename) as f:
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


def _truncate_domain(domain):
    reference = get_default_qddl_domain()
    for x in reference.types:
        del domain.types[x]
    for x in reference.object_types:
        del domain.object_types[x]
    for x in reference.constants:
        del domain.constants[x]
    for x in reference.predicates:
        del domain.predicates[x]
    for x in reference.operators:
        del domain.operators[x]


def read_packages(package_paths):
    packages = dict()
    for path in package_paths:
        assert osp.isdir(path), f"Package path {path} does not exist."
        assert osp.isfile(osp.join(path, 'package.xml')), f"Package path {path} does not contain a package.xml file."

        # Read the package.xml file.
        with open(osp.join(path, 'package.xml')) as f:
            lines = f.readlines()

        package_name = None
        for line in lines:
            if '<name>' in line:
                package_name = line.strip().replace('<name>', '').replace('</name>', '').strip()

        assert package_name is not None, f"Package path {path} does not contain a package name."

        packages[package_name] = path
    return packages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('domain', type=str, help='The domain file.')
    parser.add_argument('problem', type=str, help='The problem file.')
    parser.add_argument('--output', type=str, help='The output directory.')
    parser.add_argument('--packages', type=str, nargs='+', help='The packages to be included.')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    package_map = read_packages(args.packages)
    domain, problem = load_qddl(args.domain, args.problem)
    _transform_tree(problem, package_map, args.output)
    _truncate_domain(problem.domain)

    domain_url = osp.join(args.output, 'domain.pddl')
    problem_url = osp.join(args.output, 'problem.pddl')
    with wrapped_open_w(domain_url) as f:
        f.write(problem.domain.to_pddl())
    with wrapped_open_w(problem_url) as f:
        f.write(problem.to_pddl())


if __name__ == '__main__':
    main()

