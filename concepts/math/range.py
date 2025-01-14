#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : range.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/23/2021
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Below are classes for defining the ranges of joint configurations. These classes are generic, they don't depend on the particular robot."""

import random
from typing import Optional

__all__ = ['Range']


class Range(object):
    """A range of values, handles wraparound."""

    def __init__(self, low: float, high: float, wrap_around: bool = False):
        """Initialize the range.

        Args:
            low: The lower bound of the range.
            high: The upper bound of the range.
            wrap_around: Whether the range wraps around, i.e. whether the upper bound is adjacent to the lower bound.
        """

        # TODO(Jiayuan Mao @ 2024/12/20): this is a hack for "continuous" joints in pybullet.
        # When a joint is continuous, they will actually give lower=0 and upper=-1...
        # This is a hack to fix this issue. We should fix this in the future by considering those wheels as "wrap-around" joints.
        if low > high:
            low, high = high, low
        self.low = low
        self.high = high
        self.wrap_around = wrap_around

    low: float
    """The lower bound of the range."""

    high: float
    """The upper bound of the range."""

    wrap_around: bool
    """Whether the range wraps around, i.e. whether the upper bound is adjacent to the lower bound."""

    def __str__(self):
        return f'Range({self.low}, {self.high}, wrap_around={self.wrap_around})'

    def __repr__(self):
        return str(self)

    def difference(self, config1: float, config2: float) -> float:
        """Return the difference between two configurations.

        Args:
            config1: The first configuration.
            config2: The second configuration.

        Returns:
            The difference between the two configurations.
        """
        if self.wrap_around:
            if config1 < config2:
                if abs(config2 - config1) < abs((self.low - config1) + (config2 - self.high)):
                    return config2 - config1
                else:
                    return (self.low - config1) + (config2 - self.high)
            else:
                if abs(config2 - config1) < abs((self.high - config1) + (config2 - self.low)):
                    return config2 - config1
                else:
                    return (self.high - config1) + (config2 - self.low)
        else:
            return config2 - config1

    def make_in_range(self, value: float) -> Optional[float]:
        """Return the value if it is in range, otherwise return None. When wrap_around is True, the value is wrapped around.

        Args:
            value: the value to be checked.

        Returns:
            the value if it is in range, otherwise return None. When wrap_around is True, a value will always be returned.
        """
        if self.wrap_around:
            altered = value
            while not (self.low <= altered <= self.high):
                if altered > self.high:
                    altered -= self.high - self.low
                else:
                    altered += self.high - self.low
            return altered
        else:
            if self.contains(value):
                return value
            else:
                return self.low if abs(value - self.low) < abs(value - self.high) else self.high

    def sample(self) -> float:
        """Sample a value from the range."""
        return (self.high - self.low) * random.random() + self.low

    def contains(self, x: float) -> bool:
        """Return whether the value is in the range."""
        return self.wrap_around or (self.low <= x <= self.high)
