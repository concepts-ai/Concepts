#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : solution_score_tracker.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 11/20/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Utilities for PDSketch planners."""

from typing import Any

__all__ = ['MostPromisingTrajectoryTracker']


class MostPromisingTrajectoryTracker(object):
    """This is a tracker for tracking the most promising next action, used in joint learning settings
    where we don't have knowledge about the actual transition function or the goal."""

    def __init__(self, threshold: float):
        """Initialize the tracker.

        Args:
            threshold: the threshold for the score.
        """

        self.threshold = threshold
        self.best_score = float('-inf')
        self.solution = None

    threshold: float
    """A score threshold."""

    best_score: float
    """The best score achived so far."""

    solution: Any
    """The solution associated with the `best_score`."""

    def check(self, new_score: float) -> bool:
        """If the new score is better than the current best score, return True.

        Args:
            new_score: the new score.

        Returns:
            True if the new score is better than the current best score.
        """
        return new_score > self.best_score

    def update(self, new_score: float, solution: Any):
        """Update the best score and the solution.

        Args:
            new_score: the new score.
            solution: the new solution.
        """
        assert new_score > self.best_score
        self.best_score = new_score
        self.solution = solution

