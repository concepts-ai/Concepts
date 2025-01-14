#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : client.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/17/2020
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import time
import os.path as osp
import tempfile
import threading
import collections
import functools
import contextlib
import warnings
from typing import Any, Optional, Union, Tuple, List, Dict

import numpy as np
import pybullet as p
import pybullet_data
import jacinle
import jacinle.io as io

from concepts.math.rotationlib_xyzw import quat2mat
from concepts.simulator.pybullet.world import BulletWorld
from concepts.utils.typing_utils import Vec3f, Vec4f

__all__ = ['BulletClient']


class BulletP(object):
    def __init__(self, client_id=None):
        self.client_id = client_id

    def set_client_id(self, client_id):
        self.client_id = client_id

    def __getattr__(self, item):
        assert self.client_id is not None
        func = getattr(p, item)
        if callable(func):
            return functools.partial(func, physicsClientId=self.client_id)
        return func


class MouseEvent(collections.namedtuple('_MouseEvent', ['eventType', 'mousePosX', 'mousePosY', 'buttonIndex', 'buttonState'])):
    pass


class BulletClient(object):
    """A wrapper for the pybullet client."""

    DEFAULT_ENGINE_PARAMETERS = {'numSolverIterations': 10}
    DEFAULT_FPS = 240
    DEFAULT_GRAVITY = (0, 0, -9.8)
    DEFAULT_ASSETS_ROOT = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))), 'assets')

    def __init__(
        self,
        assets_root: Optional[str] = None,
        is_gui: bool = False,
        *,
        fps: Optional[int] = None,
        render_fps: Optional[int] = None,
        gravity: Optional[Union[Tuple[float], float]] = None,
        connect: bool = True,
        client_id: int = -1,
        width: Optional[int] = 960,
        height: Optional[int] = 960,
        additional_title: Optional[str] = None,
        save_video: Optional[str] = None,
        enable_realtime_rendering: Optional[bool] = None,
        enable_debug_gui: bool = False,
        engine_parameters: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the BulletClient.

        Args:
            assets_root: the root directory of the assets (by default it is the `assets` directory in the `concepts` package).
            is_gui: whether to enable the GUI.
            fps: the physics simulation FPS (default: 120).
            render_fps: the rendering FPS (default: 120).
            gravity: the gravity vector (default: (0, 0, -9.8)).
            connect: whether to connect to the server immediately.
            client_id: the client id to connect to. If this is set to -1, a new client id will be created.
            save_video: the path to save the video.
            width: the width of the window.
            height: the height of the window.
            additional_title: the additional title of the window.
            enable_debug_gui: whether to enable the debug GUI.
            enable_realtime_rendering: whether to enable realtime rendering (default: True if render_fps is set, otherwise False).
            engine_parameters: additional engine parameters.
        """
        if not is_gui:
            render_fps = 0

        self.is_gui = is_gui
        self.fps = fps if fps is not None else type(self).DEFAULT_FPS
        self.render_fps = render_fps if render_fps is not None else self.fps
        if not self.is_gui:
            self.render_fps = 0
        self.gravity = canonicalize_gravity(gravity if gravity is not None else type(self).DEFAULT_GRAVITY)
        self.engine_parameters = engine_parameters
        self.client_id = None
        self.assets_root = assets_root if assets_root is not None else type(self).DEFAULT_ASSETS_ROOT
        self.save_video = save_video
        self.additional_title = additional_title

        self.enable_realtime_rendering = enable_realtime_rendering if enable_realtime_rendering is not None else self.render_fps > 0
        self.enable_debug_gui = enable_debug_gui

        self.w = BulletWorld()
        self.p = BulletP()
        self.width = width
        self.height = height
        self.debug_items = dict()

        if client_id == -1:
            if connect:
                self.connect()
        else:
            self.client_id = client_id
            self.w.set_client_id(self.client_id)
            self.p.set_client_id(self.client_id)

        self._nonphysics_step_callbacks = []

    debug_items: Dict[str, Union[int, Tuple[int, ...]]]
    """The debug items that are added to the world. The key is the name of the item, and the value is the item id."""

    @property
    def world(self):
        """Alias for `self.w`."""
        return self.w

    @contextlib.contextmanager
    def with_fps(self, fps: Optional[int] = None, render_fps: Optional[int] = None, realtime_rendering: Optional[bool] = None):
        current_fps, current_render_fps, current_realtime_rendering = self.fps, self.render_fps, self.enable_realtime_rendering
        if realtime_rendering is None and render_fps is not None:  # if render_fps is set, we assume realtime rendering is enabled.
            realtime_rendering = True
        if fps is not None:
            self.fps = fps
        if render_fps is not None:
            self.render_fps = render_fps
        elif fps is not None:
            self.render_fps = fps
        if realtime_rendering is not None:
            self.enable_realtime_rendering = realtime_rendering
        yield
        self.fps, self.render_fps, self.enable_realtime_rendering = current_fps, current_render_fps, current_realtime_rendering

    def set_rendering_fps(self, render_fps: Optional[int] = None):
        if render_fps is None:
            self.render_fps = self.fps
        else:
            self.render_fps = render_fps
            self.enable_realtime_rendering = True

    def set_enable_realtime_rendering(self, enable_realtime_rendering: Optional[bool] = None):
        if enable_realtime_rendering is None:
            self.enable_realtime_rendering = self.render_fps > 0
        else:
            self.enable_realtime_rendering = enable_realtime_rendering

    def connect(self, suppress_warnings: bool = True):
        if suppress_warnings:
            with jacinle.suppress_stdout():
                self._connect()
        else:
            self._connect()

    def _connect(self):
        options = ''
        if self.save_video:
            options += f'--mp4="{self.save_video}" --mp4fps=60'
        if self.width is not None:
            options += ' --width={}'.format(self.width)
        if self.height is not None:
            options += ' --height={}'.format(self.height)
        self.client_id = p.connect(p.GUI if self.is_gui else p.DIRECT, options=options)

        # p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 0, physicsClientId=self.client_id)
        if self.save_video:
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1, physicsClientId=self.client_id)
        if self.is_gui and self.enable_realtime_rendering:
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1, physicsClientId=self.client_id)

        # Disable the cache of the URDF files. This would allow us to load JIT URDF files.
        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=self.client_id)

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        if self.engine_parameters is not None:
            p.setPhysicsEngineParameter(physicsClientId=self.client_id, **self.engine_parameters)
        else:
            p.setPhysicsEngineParameter(physicsClientId=self.client_id, **type(self).DEFAULT_ENGINE_PARAMETERS)

        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self.client_id)

        # Disable the GUI (e.g., synthetic camera views and parameters).
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, self.enable_debug_gui, physicsClientId=self.client_id)
        if self.enable_debug_gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1, physicsClientId=self.client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1, physicsClientId=self.client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1, physicsClientId=self.client_id)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.client_id)

        if self.assets_root is not None:
            file_io = p.loadPlugin('fileIOPlugin', physicsClientId=self.client_id)
            if file_io >= 0:
                p.executePluginCommand(file_io, textArgument=self.assets_root, intArgs=[p.AddFileIOAction], physicsClientId=self.client_id)
            else:
                raise RuntimeError('pybullet: cannot load FileIO!')
            p.setAdditionalSearchPath(self.assets_root, physicsClientId=self.client_id)

        # NB(Jiayuan Mao @ 10/04): also add the temp dir to the asset path so that we can load JIT URDF files.
        p.setAdditionalSearchPath(tempfile.gettempdir(), physicsClientId=self.client_id)

        p.setGravity(*self.gravity, physicsClientId=self.client_id)
        p.setTimeStep(1.0 / self.fps, physicsClientId=self.client_id)
        self.w.set_client_id(self.client_id)
        self.p.set_client_id(self.client_id)

        # Set the title of the window.
        if self.additional_title is not None:
            p.addUserDebugText(self.additional_title, [0, 0, 1], [0, 0, 0], parentObjectUniqueId=0, physicsClientId=self.client_id)

    def is_connected(self):
        return p.isConnected(physicsClientId=self.client_id)

    def has_gui(self):
        return p.getConnectionInfo(physicsClientId=self.client_id)['connectionMethod'] == p.GUI

    def disconnect(self):
        p.disconnect(physicsClientId=self.client_id)

    def reset_world(self):
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(*self.gravity, physicsClientId=self.client_id)
        p.setTimeStep(1.0 / self.fps, physicsClientId=self.client_id)

        # Should also remember to reset the world record.
        self.w = BulletWorld()
        self.w.set_client_id(self.client_id)

    def set_rendering(self, enable: bool):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, int(enable), physicsClientId=self.client_id)

    @contextlib.contextmanager
    def disable_rendering(self, disable_rendering: bool = True, reset: bool = False, suppress_stdout: bool = False):
        if reset:
            self.reset_world()

        with jacinle.cond_with(
            jacinle.suppress_stdout(),
            suppress_stdout
        ):
            if disable_rendering:
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.client_id)
            yield
            if disable_rendering and self.is_connected():
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.client_id)

    @contextlib.contextmanager
    def disable_stdout(self, activate: bool = True):
        with jacinle.cond_with(
            jacinle.suppress_stdout(),
            activate
        ):
            yield

    @contextlib.contextmanager
    def disable_world_update(self):
        """Temporarily disable the world update. Specifically, when loading a new model, the world object `self.w` will not be updated.
        This function also disables rendering of the pybullet debug renderer. Thus, this functionality is useful when loading a large number of models."""
        warnings.warn('`disable_world_update` is deprecated. Use `disable_rendering` instead.', DeprecationWarning)
        with self.disable_rendering(suppress_stdout=True):
            yield

    def step(self, steps=1, realtime_rendering: Optional[bool] = None):
        clock = None
        actual_reatime_rendering = realtime_rendering if realtime_rendering is not None else self.enable_realtime_rendering
        if actual_reatime_rendering:
            if self.render_fps > 0:
                clock = jacinle.Clock(1 / self.render_fps)
        for i in range(steps):
            p.stepSimulation(physicsClientId=self.client_id)
            self._nonphysics_step()
            if clock is not None:
                clock.tick()

    def step_until_stable(self, max_steps: int = int(1e6), velocity_threshold: float = 1e-3, angular_velocity_threshold: float = 1e-3, joint_velocity_threshold: float = 1e-3):
        for _ in range(max_steps):
            p.stepSimulation(physicsClientId=self.client_id)
            self._nonphysics_step()
            if self.is_stable(velocity_threshold=velocity_threshold, angular_velocity_threshold=angular_velocity_threshold, joint_velocity_threshold=joint_velocity_threshold):
                break

    def is_stable(self, velocity_threshold: float = 1e-3, angular_velocity_threshold: float = 1e-3, joint_velocity_threshold: float = 1e-3):
        for body_id in self.world.body_names.int_to_string:
            vel, ang_vel = p.getBaseVelocity(body_id, physicsClientId=self.client_id)
            if any(abs(v) > velocity_threshold for v in vel):
                return False
            if any(abs(v) > angular_velocity_threshold for v in ang_vel):
                return False

        for body_id, joint_id in self.world.joint_names.int_to_string:
            vel = p.getJointState(body_id, joint_id, physicsClientId=self.client_id)[1]
            if isinstance(vel, (list, tuple)):
                if any(abs(v) > joint_velocity_threshold for v in vel):
                    return False
            else:
                if abs(vel) > joint_velocity_threshold:
                    return False
        return True

    def _nonphysics_step(self):
        for cb in self._nonphysics_step_callbacks:
            cb()

    def add_nonphysics_step_callback(self, cb):
        self._nonphysics_step_callbacks.append(cb)

    def remove_nonphysics_step_callback(self, cb):
        self._nonphysics_step_callbacks.remove(cb)

    def load_urdf(self, xml_path, pos=(0, 0, 0), quat=(0, 0, 0, 1), body_name: Optional[str] = None, group: Optional[str] = '__UNSET__', static=False, scale: float = 1.0, rgba=None, notify_world_update=True) -> int:
        xml_path = self.canonicalize_asset_path(xml_path)
        pos, quat = canonicalize_default_pos_and_quat(pos, quat)
        try:
            ret = p.loadURDF(xml_path, pos, quat, useFixedBase=static, globalScaling=scale, physicsClientId=self.client_id, flags=p.URDF_USE_SELF_COLLISION)
        except p.error as e:
            raise RuntimeError('pybullet: cannot load URDF file: {}'.format(xml_path)) from e
        if notify_world_update:
            if group == '__UNSET__':
                group = 'fixed' if static else 'rigid'
            self.w.notify_update(ret, body_name=body_name, group=group)
        if rgba is not None:
            self.w.change_visual_color(ret, rgba=rgba)
        return ret

    def load_urdf_template(self, xml_path: str, replaces: Dict[str, Any], pos=None, quat=None, **kwargs) -> int:
        xml_path = self.canonicalize_asset_path(xml_path)
        with open(xml_path) as f:
            xml_content = f.read()
        for k, v in sorted(replaces.items(), key=lambda x: len(x[0]), reverse=True):
            if isinstance(v, (tuple, list)):
                for i in range(len(v)):
                    xml_content = xml_content.replace(k + str(i), str(v[i]))
            else:
                xml_content = xml_content.replace(k, str(v))

        with io.tempfile('w', '.xml') as f:
            f.write(xml_content)
            f.flush()
            return self.load_urdf(f.name, pos=pos, quat=quat, **kwargs)

    def load_urdf_string(self, string: str, pos=None, quat=None, **kwargs) -> int:
        with io.tempfile('w', '.urdf') as f:
            f.write(string)
            f.flush()
            return self.load_urdf(f.name, pos=pos, quat=quat, **kwargs)

    def load_sdf(self, xml_path, scale=1.0, notify_world_update=True) -> int:
        xml_path = self.canonicalize_asset_path(xml_path)
        ret = p.loadSDF(xml_path, globalScaling=scale, physicsClientId=self.client_id)
        if notify_world_update:
            self.w.notify_update(ret)
        return ret

    def load_mjcf(self, xml_path, pos=(0, 0, 0), quat=(0, 0, 0, 1), body_name=None, group='__UNSET__', static=False, notify_world_update=True) -> int:
        xml_path = self.canonicalize_asset_path(xml_path)
        pos, quat = canonicalize_default_pos_and_quat(pos, quat)
        ret = p.loadMJCF(xml_path, pos, quat, useFixedBase=static, physicsClientId=self.client_id, flags=p.MJCF_COLORS_FROM_FILE)
        if notify_world_update:
            if group == '__UNSET__':
                group = 'fixed' if static else 'rigid'
            self.w.notify_update(ret, body_name=body_name, group=group)
        return ret

    def loads_mjcf(self, xml_content, pos=None, quat=None, save_to=None, **kwargs) -> int:
        if not isinstance(xml_content, str):
            xml_content = io.dumps_xml(xml_content)

        if save_to is not None:
            with open(save_to, 'w') as f:
                f.write(xml_content)

        with io.tempfile('w', '.xml') as f:
            f.write(xml_content)
            f.flush()
            return self.load_mjcf(f.name, pos=pos, quat=quat, **kwargs)

    def remove_body(self, body_id):
        return p.removeBody(body_id, physicsClientId=self.client_id)

    def canonicalize_asset_path(self, path):
        return path.replace('assets://', self.assets_root + '/')

    def perform_collision_detection(self):
        warnings.warn('`perform_collision_detection` is deprecated. Use `world.perform_collision_detection` instead.', DeprecationWarning)
        p.performCollisionDetection(physicsClientId=self.client_id)

    def get_mouse_events(self) -> List[MouseEvent]:
        return list(MouseEvent(*event) for event in self.p.getMouseEvents())

    def update_viewer(self):
        self.p.getMouseEvents()

    def update_viewer_twice(self):
        self.update_viewer()
        time.sleep(0.1)
        self.update_viewer()

    def wait_for_duration(self, duration):
        t0 = time.time()
        while time.time() - t0 <= duration:
            self.update_viewer()

    def wait_forever(self):
        print(jacinle.colored('Entering the infinite loop. Press Ctrl+C to exit.', 'yellow'))
        try:
            while True:
                self.update_viewer()
        except KeyboardInterrupt:
            print(jacinle.colored('Ctrl+C detected. Exiting...', 'yellow'))
            pass

    def wait_for_user(self, message='Press enter to continue...'):
        import platform

        try:
            message = jacinle.colored('Entering the infinite loop. Enter a command to continue. Enter ipdb to enter the debugger and exit to force quit. User message:', 'yellow') + '\n' + message
            if self.has_gui() and platform.system() == 'Darwin':
                # OS X doesn't multi-thread the OpenGL visualizer
                rv = self._threaded_input(message)
            else:
                rv = input(message)
            if rv.strip().lower() == 'ipdb':
                import ipdb
                ipdb.set_trace()
            elif rv.strip().lower() == 'exit':
                import sys
                sys.exit(0)
            return rv
        except KeyboardInterrupt:
            return None

    def timeout(self, duration: float):
        for _ in range(int(duration * self.fps)):
            yield

    def absolute_timeout(self, duration: float):
        return jacinle.timeout(duration, fps=self.fps)

    def _threaded_input(self, *args, **kwargs):
        # OS X doesn't multi-thread the OpenGL visualizer
        data = []
        thread = threading.Thread(target=lambda: data.append(input(*args, **kwargs)), args=[])
        thread.start()
        try:
            while thread.is_alive():
                self.update_viewer()
        finally:
            thread.join()
        return data[-1]

    def add_debug_line(self, start_pos, end_pos, color, name=None, life_time=0) -> int:
        rv = p.addUserDebugLine(start_pos, end_pos, color, life_time, physicsClientId=self.client_id)
        return self.register_debug_item(name, rv)

    def add_debug_text(self, text, pos, color, name=None, life_time=0) -> int:
        rv = p.addUserDebugText(text, pos, color, life_time, physicsClientId=self.client_id)
        return self.register_debug_item(name, rv)

    def add_debug_ray(self, start_pos, delta, color, length: float = 1.0,name=None, life_time=0) -> int:
        rv = list()
        rv.append(p.addUserDebugLine(start_pos, start_pos + np.asarray(delta) * length, color, life_time, physicsClientId=self.client_id))
        rv.extend(self.add_debug_cube(start_pos, (0.05, 0.05, 0.05), color, life_time=life_time))
        return self.register_debug_item(name, tuple(rv))

    def add_debug_cube(self, center, extent, color, name=None, life_time=0) -> Tuple[int, ...]:
        rv = list()
        center = np.asarray(center)
        extent = np.asarray(extent)

        min_point = center - extent / 2
        max_point = center + extent / 2

        edges = [
            (min_point, min_point + np.array([extent[0], 0, 0])),
            (min_point, min_point + np.array([0, extent[1], 0])),
            (min_point, min_point + np.array([0, 0, extent[2]])),
            (min_point + np.array([0, extent[1], 0]), min_point + np.array([extent[0], extent[1], 0])),
            (min_point + np.array([0, extent[1], 0]), min_point + np.array([0, extent[1], extent[2]])),
            (min_point + np.array([extent[0], 0, 0]), min_point + np.array([extent[0], extent[1], 0])),
            (min_point + np.array([extent[0], 0, 0]), min_point + np.array([extent[0], 0, extent[2]])),
            (min_point + np.array([0, 0, extent[2]]), min_point + np.array([0, extent[1], extent[2]])),
            (min_point + np.array([0, 0, extent[2]]), min_point + np.array([extent[0], 0, extent[2]])),
            (min_point + np.array([extent[0], extent[1], 0]), max_point),
            (min_point + np.array([0, extent[1], extent[2]]), max_point),
            (min_point + np.array([extent[0], 0, extent[2]]), max_point),
        ]

        for edge in edges:
            rv.append(self.add_debug_line(edge[0], edge[1], color, life_time=life_time))

        return self.register_debug_item(name, tuple(rv))

    def register_debug_item(self, name: Optional[str], item_id: Union[int, Tuple[int, ...]]) -> Union[int, Tuple[int, ...]]:
        if name is not None:
            if name in self.debug_items:
                self.remove_debug_item(name)
            self.debug_items[name] = item_id
        return item_id

    def remove_debug_item(self, item_id: Union[str, int, Tuple[int, ...]]):
        if isinstance(item_id, str):
            item_id = self.debug_items[item_id]

        if isinstance(item_id, (tuple, list)):
            for i in item_id:
                p.removeUserDebugItem(i, physicsClientId=self.client_id)
        else:
            return p.removeUserDebugItem(item_id, physicsClientId=self.client_id)

    def remove_all_debug_items(self):
        p.removeAllUserDebugItems(physicsClientId=self.client_id)

    def remove_all_named_debug_items(self):
        for item_id in self.debug_items.values():
            self.remove_debug_item(item_id)
        self.debug_items.clear()

    def add_debug_coordinate_system(self, pos, principle_axes, size: float = 0.1, name=None, life_time=0) -> Tuple[int, ...]:
        items = list()
        for i, axis in enumerate(principle_axes):
            items.append(self.add_debug_line(pos, pos + size * axis, color=(i == 0, i == 1, i == 2), life_time=life_time))
        rv = tuple(items)
        return self.register_debug_item(name, rv)

    def add_debug_coordinate_system_quat(self, pos, quat, size: float = 0.1, name=None, life_time=0) -> Tuple[int, ...]:
        mat = quat2mat(quat)
        return self.add_debug_coordinate_system(pos, mat.T, size=size, name=name, life_time=life_time)


def canonicalize_gravity(gravity):
    if isinstance(gravity, (int, float)):
        return (0, 0, gravity)
    else:
        gravity = tuple(gravity)
        assert len(gravity) == 3
        return gravity


def canonicalize_default_pos_and_quat(pos: Optional[Vec3f], quat: Optional[Vec4f]):
    if pos is None:
        pos = (0, 0, 0)
    if quat is None:
        quat = (0, 0, 0, 1)
    return pos, quat

