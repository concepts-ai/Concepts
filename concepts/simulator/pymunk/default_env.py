#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : default_env.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/26/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import random
import math
import time
import pygame
import numpy as np
import pymunk
import pymunk.pygame_util as pygame_util

from PIL import Image
from typing import Any, Tuple, List, Optional, Callable

import concepts.simulator.pymunk.body_utils as body_utils

from concepts.simulator.pymunk.world import PymunkWorld
from concepts.simulator.pymunk.constants import color_consts


class PymunkDefaultEnv(object):
    """A basic 2D physics environment based on pymunk.

    The main function to be implemented by subclasses is `reset_scene`.
    """

    TITLE = 'Pymunk Default'
    SCREENSHOT_TITLE = 'screenshot'

    def __init__(
        self,
        damping: float = 0.0,
        gravity: Tuple[float, float] = (0., 0.),
        world_width: int = 800, world_height: int = 800, fps: int = 60,
        action_velocity: float = 100, velocity_jitter: float = 0.0, direction_jitter: float = 0.0,
        display: bool = True, render_fps: Optional[int] = None
    ):
        self.gravity = gravity
        self.damping = damping
        self.world_width = world_width
        self.world_height = world_height
        self.fps = fps
        self.action_velocity = action_velocity
        self.velocity_jitter = velocity_jitter
        self.direction_jitter = direction_jitter
        self.additional_drawing_regions = list()

        self.display = display
        self.render_fps = render_fps or fps

        self.world: PymunkWorld = None
        self.reset_world()
        self.current_selection = None
        self.recorded_frames = None

        self.additional_step_callable = list()

        if display:
            pygame.init()
            pygame.display.set_caption(type(self).TITLE)
            self.screen = pygame.display.set_mode((self.world.screen_width, self.world.screen_height))
            self.draw_options = pygame_util.DrawOptions(self.screen)
        else:
            self.screen = None
            self.draw_options = None

        self.clock = pygame.time.Clock()

    def set_additional_step_callable(self, fn: Callable[['PymunkDefaultEnv'], None]):
        self.additional_step_callable.append(fn)

    def set_current_selection(self, body: pymunk.Body):
        self.current_selection = body

    def add_additional_drawing_region_rect(self, top_left: Tuple[int, int], bottom_right: Tuple[int, int], color: Tuple[int, int, int]):
        self.additional_drawing_regions.append({'type': 'rect', 'top_left': top_left, 'bottom_right': bottom_right, 'color': color})

    def add_additional_drawing_region_circle(self, center: Tuple[int, int], radius: int, color: Tuple[int, int, int]):
        self.additional_drawing_regions.append({'type': 'circle', 'center': center, 'radius': radius, 'color': color})

    def reset_world(self) -> PymunkWorld:
        world = PymunkWorld(screen_width=self.world_width, screen_height=self.world_height)
        world.gravity = self.gravity
        world.damping = self.damping
        self.world = world
        return world

    def reset(self, **kwargs):
        self.reset_world()
        self.current_selection = None
        self.additional_drawing_regions = list()

        # Main reset functionality.
        self.reset_scene(**kwargs)

        # NB(Jiayuan Mao @ 06/29): for selectable objects, save their original color.
        for b in self.world.selectable_bodies:
            for s in b.shapes:
                s.original_color = s.color

    def reset_scene(self, **kwargs):
        raise NotImplementedError()

    def get_observation(self):
        """Get an observation dict of the current state of the environment."""
        objects = dict()
        for body, selectable, label in self.world.bodies_extra:
            objects[label] = body
        return objects

    def humanplay_mainloop(self):
        """Run a mainloop so that the user can control the objects in the environment."""
        if not self.display:
            raise ValueError('env.display is set to false. Exiting the mainloop.')

        self.step(1, render_mode='human')
        while True:
            running = True
            for event in pygame.event.get():
                running &= self._handle_humanplay_event(event)
            if not running:
                break

            self.step(1, render_mode='human')

    RENDER_MODES = ['image', 'human', 'pose', 'state']

    def non_physical_execute_trajectory(self, body_name: str, trajectory: List[Tuple[float, float]]) -> None:
        """Execute a trajectory without physical simulation. This is done by manually setting the positions of the bodies.

        Args:
            body_name: the name of the body to be moved.
            trajectory: a list of positions to be set to the body.
        """
        body = self.world.get_body_by_label(body_name)
        for p in trajectory:
            body.position = p
            self.step(1, render_mode='human')

    def step(self, steps: int, render_mode: Optional[str] = 'image', callback: Optional[Callable[[], bool]] = None) -> List[Any]:
        """Step the simulation for a number of steps.

        Args:
            steps: the number of steps to be executed.
            render_mode: the mode of rendering. Can be one of ['image', 'human', 'pose', 'state']. Set to None to disable any rendering.
            callback: a callback function that returns False to stop the simulation.
        """
        trajectory = list()

        for i in range(steps):
            self._step_with_render(render_mode, trajectory)
            if callback is not None and not callback():
                break

        return trajectory

    def _step_with_render(self, render_mode: Optional[str] = 'image', trajectory: Optional[List[Any]] = None) -> None:
        """Step the simulation and render the environment.

        Args:
            render_mode: the mode of rendering. Can be one of ['image', 'human', 'pose', 'state'].
            trajectory: the trajectory object (by reference) to store the rendered frames or poses.
        """
        self.world.step(1 / self.fps)
        for fn in self.additional_step_callable:
            fn(self)
        self.render_and_display(render_mode, trajectory)

    def render(self) -> None:
        """Render the environment. If recording is enabled, the rendered frames will be stored in the recorded_frames list.
        Use :meth:`start_recording` and :meth:`stop_recording` to control the recording process.

        The rendering is done by calling the `debug_draw` method of the PymunkWorld object.
        """
        assert self.display, 'Cannot render when display is disabled.'

        self.screen.fill((255, 255, 255))
        # screen_width, screen_height = self.screen.get_width(), self.screen.get_height()
        # pygame.draw.rect(self.screen, (245, 245, 245), (0, screen_height // 2, screen_width, screen_height // 2))

        for region in self.additional_drawing_regions:
            if region['type'] == 'rect':
                pygame.draw.rect(self.screen, region['color'], region['top_left'] + region['bottom_right'])
            elif region['type'] == 'circle':
                pygame.draw.circle(self.screen, region['color'], region['center'], region['radius'])
            else:
                raise ValueError('Unknown drawing region type: {}.'.format(region['type']))

        for b in self.world.selectable_bodies:
            for s in b.shapes:
                if hasattr(s, 'original_color'):
                    s.color = s.original_color

        # draw the selection.
        if self.current_selection is not None:
            for s in self.current_selection.shapes:
                s.color = color_consts.RED

        self.world.debug_draw(self.draw_options)
        pygame.display.flip()

        if self.recorded_frames is not None:
            data = pygame.image.tostring(self.screen, 'RGBA')
            img = Image.frombytes('RGBA', (self.world.screen_width, self.world.screen_height), data)
            self.recorded_frames.append(img)

    def render_and_display(self, render_mode: Optional[str] = 'image', trajectory: Optional[List[Any]] = None) -> None:
        if render_mode is None:
            pass
        elif render_mode == 'image':
            self.render()
            data = pygame.image.tostring(self.screen, 'RGBA')
            img = Image.frombytes('RGBA', (self.world.screen_width, self.world.screen_height), data)
            assert trajectory is not None, 'trajectory must be provided when render_mode is image.'
            trajectory.append(img)
        elif render_mode == 'human':
            self.render()
            self.clock.tick(self.render_fps)
        elif render_mode == 'pose':
            assert trajectory is not None, 'trajectory must be provided when render_mode is pose.'
            trajectory.append(self.world.get_body_poses())
        elif render_mode == 'state':
            assert trajectory is not None, 'trajectory must be provided when render_mode is pose.'
            trajectory.append(self.world.get_body_states())
        else:
            raise ValueError('Unknown render mode: {}.'.format(render_mode))

    def start_recording(self):
        self.recorded_frames = list()

    def stop_recording(self):
        try:
            return self.recorded_frames
        finally:
            self.recorded_frames = None

    def _handle_humanplay_event(self, event: pygame.event.Event) -> bool:
        """Handle the humanplay event."""
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and (
            event.key in [pygame.K_ESCAPE, pygame.K_q]
        ):
            return False

        if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
            fname = type(self).SCREENSHOT_TITLE + '-{}.png'.format(time.strftime('%Y%m%d-%H%M%S'))
            pygame.image.save(self.screen, fname)
            print('Saved screenshot to "{}".'.format(fname))
            return True

        if event.type == pygame.MOUSEBUTTONDOWN:
            p = pygame_util.from_pygame(event.pos, self.screen)
            self.current_selection = body_utils.select_body(self.world, p, self.world.selectable_bodies)
            if self.current_selection is not None:
                print('Object selected: {}'.format(self.current_selection))

            return True

        if self.current_selection is not None and event.type in (pygame.KEYDOWN, pygame.KEYUP):
            player_body = self.current_selection
            if self.can_control(player_body):
                if event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                    player_body.velocity = sample_velocity(self.action_velocity, self.velocity_jitter, 180, self.direction_jitter)
                elif event.type == pygame.KEYUP and event.key == pygame.K_LEFT:
                    player_body.velocity = 0, 0
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                    player_body.velocity = sample_velocity(self.action_velocity, self.velocity_jitter, 0, self.direction_jitter)
                elif event.type == pygame.KEYUP and event.key == pygame.K_RIGHT:
                    player_body.velocity = 0, 0
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                    player_body.velocity = sample_velocity(self.action_velocity, self.velocity_jitter, -90, self.direction_jitter)
                elif event.type == pygame.KEYUP and event.key == pygame.K_UP:
                    player_body.velocity = 0, 0
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                    player_body.velocity = sample_velocity(self.action_velocity, self.velocity_jitter, 90, self.direction_jitter)
                elif event.type == pygame.KEYUP and event.key == pygame.K_DOWN:
                    player_body.velocity = 0, 0
            else:
                player_body.velocity = 0, 0
                print('Invalid selection of object.')

            self._handle_keyboard_event(event)

        return True

    def _handle_keyboard_event(self, event: pygame.event.Event) -> None:
        pass

    def can_control(self, body) -> bool:
        """Check if the body can be controlled by the user. This function can be overridden by subclasses."""
        return True

    def get_jittered_velocity(self, velocity: Tuple[float, float]) -> Tuple[float, float]:
        velocity_scale = np.linalg.norm(velocity)
        velocity_direction = np.degrees(np.arctan2(velocity[1], velocity[0]))
        return sample_velocity(velocity_scale, self.velocity_jitter, velocity_direction, self.direction_jitter)


def sample_velocity(velocity: float, velocity_jitter: float, direction_deg: float, direction_jitter: float) -> Tuple[float, float]:
    velocity = velocity + random.uniform(-velocity_jitter, velocity_jitter)
    direction_deg = direction_deg + random.uniform(-direction_jitter, direction_jitter)
    direction = math.radians(direction_deg)
    return velocity * math.cos(direction), velocity * math.sin(direction)
