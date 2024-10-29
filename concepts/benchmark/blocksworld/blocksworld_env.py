#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : blocksworld_env.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/03/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import numpy as np
from typing import Optional, Tuple

from concepts.benchmark.common.random_env import RandomizedEnv
from concepts.benchmark.blocksworld.blocksworld import BlockWorld, random_generate_blocks_world


class BlockWorldEnvBase(RandomizedEnv):
    def __init__(self, nr_blocks: int, random_order: bool = False, prob_unchanged: float = 0.0, prob_fall: float = 0.0, np_random: Optional[np.random.RandomState] = None, seed: Optional[int] = None):
        """Initialize the blocksworld environment.

        Args:
            nr_blocks: number of blocks.
            random_order: randomly permute the indexes of the blocks. This option prevents the models from memorizing the configurations.
            prob_unchanged: the probability of not changing the state.
            prob_fall: the probability of falling to the ground.
        """
        super().__init__(np_random=np_random, seed=seed)
        self.nr_blocks = nr_blocks
        self.random_order = random_order
        self.prob_unchanged = prob_unchanged
        self.prob_fall = prob_fall

        self.world = None
        self.is_over = False
        self.cached_result = None

    world: Optional[BlockWorld]
    """The current blocksworld."""

    is_over: bool
    """Whether the current episode is over."""

    cached_result: Optional[Tuple[float, bool]]
    """The result of the current episode. It is a tuple of (reward, is_over)."""

    @property
    def nr_objects(self):
        """Get the number of objects in the environment."""
        return self.nr_blocks + 1

    def reset_nr_blocks(self, nr_blocks: int):
        """Reset the number of blocks."""
        self.nr_blocks = nr_blocks

    def reset(self, **kwargs):
        """Reset the environment. This function first generates a random blocksworld, and then returns the current state."""
        self.world = random_generate_blocks_world(self.nr_blocks, random_order=self.random_order, np_random=self.np_random)
        self.is_over = False
        self.cached_result = self._get_result()

        return self._get_decorated_states()

    def render(self, mode: str = 'human'):
        print(self.world.get_world_string())

    def step(self, action):
        raise NotImplementedError()

    def get_current_state(self):
        return self._get_decorated_states()

    def _get_decorated_states(self, decorate: bool = False, world_id: int = 0):
        state = self.world.get_coordinates()
        if decorate:
            state = _decorate(state, self.nr_objects, world_id)
        return state

    def _get_result(self):
        raise NotImplementedError()


class SimpleMoveBlockWorldEnvBase(BlockWorldEnvBase):
    world: BlockWorld
    is_over: bool
    cached_result: Optional[Tuple[float, bool]]

    def step(self, action):
        assert self.world is not None, 'You need to call restart() first.'
        if self.is_over:
            return self.get_current_state(), 0, True
        r, is_over = self.cached_result
        if is_over:
            self.is_over = True
            return self.get_current_state(), r, is_over

        x, y = action
        assert 0 <= x <= self.nr_blocks and 0 <= y <= self.nr_blocks

        p = self.np_random.rand()
        if p >= self.prob_unchanged:
            if p < self.prob_unchanged + self.prob_fall:
                y = self.world.blocks.inv_index(0) # fall to the ground
            self.world.move(x, y)

        r, is_over = self._get_result()
        if is_over:
            self.is_over = True
        return self.get_current_state(), r, is_over

    def _get_heights(self):
        """Get the list of heights of the block towers. This function will return a sortes list of heights."""
        coor = self.world.get_coordinates()
        height = {}
        for i in coor:
            x, y = i
            if not x in height:
                height[x] = y
            else:
                height[x] = max(height[x], y)
        heights = []
        for i in height.keys():
            heights.append(height[i])
        heights.sort()
        return heights

    def _get_result(self):
        raise NotImplementedError()


class SingleClearBlockWorldEnv(SimpleMoveBlockWorldEnvBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clear_idx = 0

    world: BlockWorld
    is_over: bool
    cached_result: Optional[Tuple[float, bool]]

    def reset(self):
        self.clear_idx = 0
        while True:
            super().reset()

            blocks = [self.world.blocks[i] for i in range(self.nr_blocks)]
            blocks = [b for b in blocks if not b.is_ground]
            non_clear_blocks = [b for b in blocks if len(b.children) > 0]
            if len(non_clear_blocks) == 0:
                continue

            idx = non_clear_blocks[self.np_random.randint(len(non_clear_blocks))].index
            self.clear_idx = idx
            self.is_over = False
            self.cached_result = r, is_over = self._get_result()
        return self.get_current_state()

    def get_current_state(self):
        on = self.world.get_on_relation()
        ground = self.world.get_is_ground()
        clear = 1 - on.max(0)
        clear_goal = np.zeros_like(ground)
        clear_goal[self.clear_idx] = 1

        return np.stack([
            on,
            np.broadcast_to(clear_goal[:, None], on.shape),
            np.broadcast_to(clear[:, None], on.shape),
            np.broadcast_to(ground[:, None], on.shape)
        ], axis=-1)

    def _get_result(self):
        block = self.world.blocks[self.world.blocks.inv_index(self.clear_idx)]
        if len(block.children) > 0:
            return 0, False
        else:
            return 1, True

    def get_groundtruth_steps(self):
        block = self.world.blocks[self.world.blocks.inv_index(self.clear_idx)]
        count = 0

        def dfs(b):
            nonlocal count
            if len(b.children) == 0:
                return
            for child in b.children:
                count += 1
                dfs(child)

        dfs(block)
        return count


class ToGroundBlockWorldEnv(SimpleMoveBlockWorldEnvBase):
    world: BlockWorld
    is_over: bool
    cached_result: Optional[Tuple[float, bool]]

    def _get_result(self):
        ground = self.world.blocks.raw[0]
        assert ground.is_ground
        if len(ground.children) == self.nr_blocks:
            return 1, True
        else:
            return 0, False


class ToGroundBind2ndBlockWorldEnv(ToGroundBlockWorldEnv):
    world: BlockWorld
    is_over: bool
    cached_result: Optional[Tuple[float, bool]]

    def step(self, action):
        assert 0 <= action <= self.nr_blocks
        return super().step((action, self.world.blocks.inv_index(0)))


class StackBlockWorldEnv(SimpleMoveBlockWorldEnvBase):
    world: BlockWorld
    is_over: bool
    cached_result: Optional[Tuple[float, bool]]

    def _get_result(self):
        ground = self.world.blocks.raw[0]
        assert ground.is_ground
        if len(ground.children) == 1:
            return 1, True
        else:
            return 0, False


class DenseStackBlockWorldEnv(StackBlockWorldEnv):
    world: BlockWorld
    is_over: bool
    cached_result: Optional[Tuple[float, bool]]

    highest: int
    """The height of the highest block towel."""

    def reset(self):
        super().reset()
        heights = self._get_heights()
        self.highest = heights[0]
        return self.get_current_state()

    def _get_result(self):
        r, is_over = super()._get_result()
        if is_over:
            return r, is_over
        if not hasattr(self, 'highest'):
            return 0, False
        heights = self._get_heights()
        if r == 0 and heights[0] > self.highest:
            r = 0.1
        self.highest = heights[0]
        return r, is_over


class TwinTowerBlockWorldEnv(SimpleMoveBlockWorldEnvBase):
    world: BlockWorld
    is_over: bool
    cached_result: Optional[Tuple[float, bool]]

    def reset(self):
        super().reset()
        self._customize_reset_worlds()
        return self.get_current_state()

    def _get_result(self):
        heights = self._get_heights()
        if len(heights) == 2 and heights[-1] - heights[-2] <= 1:
            return 1, True
        else:
            return 0, False

    def _customize_reset_worlds(self):
        pass


class DenseTwinTowerBlockWorldEnv(TwinTowerBlockWorldEnv):
    world: BlockWorld
    is_over: bool
    cached_result: Optional[Tuple[float, bool]]

    high2nd: int
    """The height of the second highest block towel."""

    def reset(self):
        super().reset()
        heights = self._get_heights()
        heights.append(0)
        self.high2nd = heights[1]
        return self._get_decorated_states(), 0, False

    def _get_result(self):
        r, is_over = super()._get_result()
        heights = self._get_heights()
        heights.append(0)
        if r == 0 and heights[1] > self.high2nd:
            r = 0.1
        self.high2nd = heights[1]
        return r, is_over


class FromGroundTwinTowerBlockWorldEnv(TwinTowerBlockWorldEnv):
    world: BlockWorld
    is_over: bool
    cached_result: Optional[Tuple[float, bool]]

    def _customize_reset_worlds(self):
        # TODO:: Accelerate this.
        for i in range(self.nr_objects):
            for j in range(self.nr_objects):
                self.world.move(j, self.world.blocks.inv_index(0))


class FinalBlockWorldEnv(BlockWorldEnvBase):
    def __init__(self, nr_blocks, random_order=False, shape_only=False, fix_ground=False, lstack=False, rstack=False, prob_unchanged=0.0, prob_fall=0.0, np_random=None, seed=None):
        super().__init__(nr_blocks, random_order, prob_unchanged, prob_fall,np_random=np_random, seed=seed)

        self.shape_only = shape_only
        self.fix_ground = fix_ground
        self.lstack = lstack
        self.rstack = rstack

        self.start_world = None
        self.final_world = None
        self.start_state = None
        self.final_state = None

    world: BlockWorld
    is_over: bool
    cached_result: Optional[Tuple[float, bool]]

    start_world: Optional[BlockWorld]
    """The initial blocksworld."""

    final_world: Optional[BlockWorld]
    """The target blocksworld that the agent needs to reach."""

    start_state: Optional[np.ndarray]
    """The initial state of the blocksworld."""

    final_state: Optional[np.ndarray]
    """The target state of the blocksworld."""

    def reset(self):
        self.start_world = random_generate_blocks_world(self.nr_blocks, random_order=False, one_stack=self.lstack)
        self.final_world = random_generate_blocks_world(self.nr_blocks, random_order=False, one_stack=self.rstack)
        self.world = self.start_world
        if self.random_order:
            n = self.world.size
            ground_ind = 0 if self.fix_ground else self.np_random.randint(n)

            def get_order():
                raw_order = self.np_random.permutation(n - 1)
                order = []
                for i in range(n - 1):
                    if i == ground_ind:
                        order.append(0)
                    order.append(raw_order[i] + 1)
                if ground_ind == n - 1:
                    order.append(0)
                return order

            self.start_world.blocks.set_random_order(get_order())
            self.final_world.blocks.set_random_order(get_order())

        self._customize_reset_worlds()
        self.start_state = _decorate(self._get_coordinates(self.start_world), self.nr_objects, 0)
        self.final_state = _decorate(self._get_coordinates(self.final_world), self.nr_objects, 1)

        self.is_over = False
        self.cached_result = self._get_result()
        return self.get_current_state()

    def _customize_reset_worlds(self):
        pass

    def step(self, action):
        assert self.start_world is not None, 'you need to call restart() first'

        if self.is_over:
            return 0, True
        r, is_over = self.cached_result
        if is_over:
            self.is_over = True
            return r, is_over

        x, y = action
        assert 0 <= x <= self.nr_blocks and 0 <= y <= self.nr_blocks

        p = self.np_random.rand()
        if p >= self.prob_unchanged:
            if p < self.prob_unchanged + self.prob_fall:
                y = self.start_world.blocks.inv_index(0) # fall to ground
            self.start_world.move(x, y)
            self.start_state = _decorate(self._get_coordinates(self.start_world), self.nr_objects, 0)
        r, is_over = self._get_result()
        if is_over:
            self.is_over = True
        return r, is_over

    def get_current_state(self):
        assert self.start_world is not None, 'you need to call restart() first'
        return np.vstack([self.start_state, self.final_state])

    def _get_result(self):
        sorted_start_state = self._get_coordinates(self.start_world, sort=True)
        sorted_final_state = self._get_coordinates(self.final_world, sort=True)
        if (sorted_start_state == sorted_final_state).all():
            return 1, True
        else:
            return 0, False

    def _get_coordinates(self, world, sort=False):
        coordinates = world.get_coordinates(absolute=not self.shape_only)
        if sort:
            if not self.shape_only:
                coordinates = _decorate(coordinates, self.nr_objects, 0)
            coordinates = np.array(sorted(list(map(tuple, coordinates))))
        return coordinates


class FromGroundFinalBlockWorldEnv(FinalBlockWorldEnv):
    world: BlockWorld
    is_over: bool
    cached_result: Optional[Tuple[float, bool]]

    start_world: Optional[BlockWorld]
    final_world: Optional[BlockWorld]
    start_state: Optional[np.ndarray]
    final_state: Optional[np.ndarray]

    def _customize_reset_worlds(self):
        # TODO:: Accelerate this.
        for i in range(self.nr_objects):
            for j in range(self.nr_objects):
                self.start_world.move(j, self.start_world.blocks.inv_index(0))


class DenseRewardFinalBlockWorldEnv(FinalBlockWorldEnv):
    world: BlockWorld
    is_over: bool
    cached_result: Optional[Tuple[float, bool]]

    start_world: Optional[BlockWorld]
    final_world: Optional[BlockWorld]
    start_state: Optional[np.ndarray]
    final_state: Optional[np.ndarray]

    dense_reward_potential: int

    def reset(self):
        super().reset()
        self.dense_reward_potential = self._get_potential()
        return self.get_current_state()

    def _get_result(self):
        r, is_over = super()._get_result()
        potential = self._get_potential()
        if not hasattr(self, '_potential'):
            return 0, False
        if r == 0 and potential > self.dense_reward_potential:
            r = 0.2
            self.dense_reward_potential = potential
        return r, is_over

    def _get_sorted_coordinates(self, world):
        coordinates = self.world.get_coordinates(absolute=not self.shape_only)
        coordinates = _decorate(coordinates, self.nr_objects, 0)
        def trans(x):
            x = tuple(x)
            return x[0], x[2], x[3], x[1]
        coordinates = np.array(sorted(list(map(trans, coordinates))))
        return coordinates

    def _get_potential(self):
        a = self._get_sorted_coordinates(self.start_world)
        b = self._get_sorted_coordinates(self.final_world)
        n, i, j = self.nr_objects, 0, 0
        flag, cnt = False, 0
        while i < n and j < n:
            x, y = tuple(a[i]), tuple(b[j])
            if x == y:
                if x[2] == 1 or flag:
                    flag = True
                    cnt += 1
                i, j = i + 1, j + 1
            else:
                flag = False
                if x < y:
                    i += 1
                else:
                    j += 1
        return cnt


class SubgoalRewardFinalBlockWorldEnv(FinalBlockWorldEnv):
    world: BlockWorld
    is_over: bool
    cached_result: Optional[Tuple[float, bool]]

    start_world: Optional[BlockWorld]
    final_world: Optional[BlockWorld]
    start_state: Optional[np.ndarray]
    final_state: Optional[np.ndarray]

    subgoal_achieved: bool

    def reset(self):
        self.subgoal_achieved = False
        super().reset()

    def _get_result(self):
        r, is_over = super()._get_result()
        if not self.subgoal_achieved:
            sorted_start_state = self._get_coordinates(self.start_world, sort=True)
            sorted_final_state = self._get_coordinates(self.final_world, sort=True)
            assert not self.shape_only, "not support yet"
            subgoal = True
            for i in range(len(sorted_start_state)):
                if (sorted_start_state[i] != sorted_final_state[i]).any() and sorted_start_state[i][3] != 1:
                    subgoal = False
            if subgoal:
                # print(sorted_start_state)
                # print(sorted_final_state)
                self.subgoal_achieved = True
                r += 0.5
        return r, is_over


def _decorate(state, nr_objects, world_id=None):
    info = []
    if world_id is not None:
        info.append(np.ones((nr_objects, 1)) * world_id)
    info.extend([np.array(range(nr_objects))[:, np.newaxis], state])
    return np.hstack(info)