# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

"""This file expects to be run from Blender like this: blender --background --python render_images.py -- [arguments to this script]"""

import math, sys, random, argparse, json, os, tempfile, itertools
sys.path.insert(0, '../../concepts/benchmark/clevr/scene_generator')

INSIDE_BLENDER = True
try:
    import clevr_blender_utils as utils
    from render import render_scene
except ImportError:
    INSIDE_BLENDER = False

parser = argparse.ArgumentParser()
parser.add_argument('--use_gpu', default=0, type=int, help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. You must have an NVIDIA GPU with the CUDA toolkit installed for it to work.")
parser.add_argument('--width', default=320, type=int, help="The width (in pixels) for the rendered images")
parser.add_argument('--height', default=240, type=int, help="The height (in pixels) for the rendered images")

parser.add_argument('--render_num_samples', default=512, type=int, help="The number of samples to use when rendering. Larger values will result in nicer images but will cause rendering to take longer.")
parser.add_argument('--render_min_bounces', default=8, type=int, help="The minimum number of bounces to use for rendering.")
parser.add_argument('--render_max_bounces', default=8, type=int, help="The maximum number of bounces to use for rendering.")
parser.add_argument('--render_tile_size', default=256, type=int, help=(
    "The tile size to use for rendering. This should not affect the " +
    "quality of the rendered image but may affect the speed; CPU-based " +
    "rendering may achieve better performance using smaller tile sizes " +
    "while larger tile sizes may be optimal for GPU-based rendering."
))

parser.add_argument('--key_light_jitter', default=1.0, type=float, help="The magnitude of random jitter to add to the key light position.")
parser.add_argument('--fill_light_jitter', default=1.0, type=float, help="The magnitude of random jitter to add to the fill light position.")
parser.add_argument('--back_light_jitter', default=1.0, type=float, help="The magnitude of random jitter to add to the back light position.")
parser.add_argument('--camera_jitter', default=0.5, type=float, help="The magnitude of random jitter to add to the camera position")

RSA_SCENE_SPEC_SIZE = {1: 'small', 2: 'middle1', 3: 'middle2', 4: 'large'}
RSA_SCENE_SPEC_SIZE2DIS = {1: 0.5, 2: 0.7, 3: 1.0, 4: 1.5}


def gen_scene1():
    rv = []

    dis = 0
    k = 0
    rv.append({
        'color': 'red',
        'shape': 'sphere',
        'material': 'metal',
        'size': RSA_SCENE_SPEC_SIZE[2],
        'pos': (-3 + 3 * (k // 3) - dis, -5 + 4.5 * (k % 3) - dis)
    })
    k = 2
    rv.append({
        'color': 'red',
        'shape': 'sphere',
        'material': 'metal',
        'size': RSA_SCENE_SPEC_SIZE[2],
        'pos': (-3 + 3 * (k // 3) - dis, -5 + 4.5 * (k % 3) - dis)
    })
    k = 7
    rv.append({
        'color': 'red',
        'shape': 'sphere',
        'material': 'metal',
        'size': RSA_SCENE_SPEC_SIZE[4],
        'pos': (-3 + 3 * (k // 3) - dis, -5 + 4.5 * (k % 3) - dis)
    })
    return {'objects': rv}


def main(args):
    os.makedirs('./generated', exist_ok=True)

    spec = gen_scene1()
    print(spec)
    render_scene(
        args,
        spec,
        output_image='./generated/scene_example1.png',
        output_json='./generated/scene_example1.render.json',
        # output_blendfile='./dumps/rsa_vagueness/scene_example1.blend',
        output_shadeless='./generated/scene_example1.shadeless.png'
    )


if __name__ == '__main__':
    if INSIDE_BLENDER:
        # Run normally
        argv = utils.extract_args()
        args = parser.parse_args(argv)
        main(args)
    elif '--help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
    else:
        print('This script is intended to be called from blender like this:')
        print()
        print('blender --background --python render_images.py -- [args]')
        print()
        print('You can also run as a standalone python script to view all')
        print('arguments like this:')
        print()
        print('python render_images.py --help')

