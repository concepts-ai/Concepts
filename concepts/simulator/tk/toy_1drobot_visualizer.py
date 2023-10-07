#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : toy_1drobot_visualizer.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/29/2021
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Optional
from collections import namedtuple
from concepts.simulator.tk.drawing_window import DrawingWindow


class Toy1DRobotVState(namedtuple('_Toy1DRobotVState', ['robot', 'object', 'goal'])):
    pass


class Toy1DRobotVisualizer(object):
    def __init__(self, name='Toy1DRobot'):
        self.window = DrawingWindow(512 * 2, 116 * 2, -11, 11, 3, 8, title=name)

    def draw(
        self,
        current: Toy1DRobotVState,
        predicted: Optional[Toy1DRobotVState] = None,
        text: Optional[str] = None,
        wait_key=True
    ):
        current = Toy1DRobotVState(*[float(x) if x is not None else None for x in current])
        if predicted is not None:
            predicted = Toy1DRobotVState(*[float(x) if x is not None else None for x in predicted])

        self.window.clear()

        self.window.drawText(0, 7.5, label="Blue: Robot; Red: Object; Green: Goal")
        if text is None:
            self.window.drawText(0, 7, label=text)

        self.window.drawText(0, 6, label="Observation")
        x = current.robot
        self.window.drawRect(x - 0.3, 5, x + 0.3, 5 + 0.3 * 2, color='blue')
        x = current.object
        self.window.drawRect(x - 0.3, 5, x + 0.3, 5 + 0.3 * 2, color='red')
        x = current.goal
        self.window.drawRect(x - 1, 5 - 0.2, x + 1, 5, color='green')

        if predicted is not None:
            self.window.drawText(0, 4.5, label="Prediction")
            x = predicted.object
            self.window.drawRect(x - 0.3, 5 - 1.5, x + 0.3, 5 - 1.5 + 0.3 * 2, color='#FF8888')
            x = predicted.goal
            self.window.drawRect(x - 1, 5 - 1.5 - 0.2, x + 1, 5 - 1.5, color='#88FF88')

        if wait_key:
            char = input('Press <Enter> to continue...')
            if char == 'q':
                raise KeyboardInterrupt
