#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : cliport-ocrtoc-blender.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/25/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""
This file expects to be run from Blender like this:

blender --background --python render_images.py -- [arguments to this script]
"""

import sys, argparse, json
import os.path as osp

BASE_DIR = osp.join(osp.dirname(__file__), '..', '..')
SCENE_BLENDER_FILE = osp.join(BASE_DIR, 'concepts/assets/blender/cliport_tabletop_scene.blend')

sys.path.insert(0, osp.join(BASE_DIR, 'concepts/simulator/blender'))

INSIDE_BLENDER = True
try:
    from simple_render import render_scene_simple, render_scene_incremental_with_new_camera
except ImportError as e:
    print(e)
    INSIDE_BLENDER = False

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--draft', action='store_true', help='set draft settings for quick rendering (for debugging)')


def extract_args(input_argv=None):
    """Pull out command-line arguments after "--". Blender ignores command-line flags after --,
    so this lets us forward command line arguments from the blender invocation to our own script."""
    if input_argv is None:
        input_argv = sys.argv
    output_argv = []
    if '--' in input_argv:
        idx = input_argv.index('--')
        output_argv = input_argv[(idx + 1):]
    return output_argv


def main(args):
    with open('./generated/scene_example1.json') as f:
        spec = json.load(f)

    rendering_kwargs = {
        'use_eevee': False,
        'use_gpu': True,
        'render_timeout': 30
    }
    if args.draft:
        rendering_kwargs['use_eevee'] = True

    render_scene_simple(
        spec,
        scene_blend=SCENE_BLENDER_FILE,
        output_image='./generated/scene_example1.png',
        **rendering_kwargs
    )
    print('Image saved to', './generated/scene_example1.png')
    render_scene_incremental_with_new_camera(
        output_image='./generated/scene_example1_topdown.png',
        camera_pos=(0, 0, 2.0),
        camera_angle=(0, 0, 1.5708),
        **rendering_kwargs
    )
    print('Image saved to', './generated/scene_example1_topdown.png')


if __name__ == '__main__':
    if INSIDE_BLENDER:
        # Run normally
        argv = extract_args()
        args = parser.parse_args(argv)
        main(args)
    else:
        print('This script is intended to be called from blender like this:')
        print()
        print('blender --background --python render_images.py -- [args]')
        print()
        print('You can also run as a standalone python script to view all')
        print('arguments like this:')
        print()
        print('python render_images.py --help')

