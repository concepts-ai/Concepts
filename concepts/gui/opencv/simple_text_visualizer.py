#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : simple_text_visualizer.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/21/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Optional

import cv2
import numpy as np

___all__ = ['CV2SimpleTextVisualizer', 'cv2_simple_text_visualizer', 'cv2_simple_pixel_value_visualizer']


def _draw_text(
    img, text,
    font=cv2.FONT_HERSHEY_PLAIN,
    pos=(0, 0),
    font_scale=1,
    font_thickness=2,
    text_color=(0, 0, 0),
    text_color_bg=(200, 200, 200)
):
    """Adapted from: https://stackoverflow.com/questions/60674501/how-to-make-black-background-in-cv2-puttext-with-python-opencv"""
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, (x - 5, y - 5), (x + text_w + 5, y + text_h + 5), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)


class CV2SimpleTextVisualizer(object):
    """A simple visualizer for text. The class should be initialized with a callback function, which takes the image and
    the mouse position as input, and returns the text to be displayed. The visualizer will display the image and the text.

    Example:

        .. code-block:: python

                def callback(image, x, y):
                    return f'Pixel value: {image[y, x]} (x={x}, y={y})'

                visualizer = SimpleTextVisualizer(callback)
                visualizer.run(image)
    """
    def __init__(self, callback, title: Optional[str] = None, font: int = cv2.FONT_HERSHEY_SIMPLEX, font_scale: int = 1, font_thickness: int = 2):
        """Initialize the visualizer.

        Args:
            callback: the callback function.
            title: the title of the window.
            font: the font of the text.
            font_scale: the scale of the font.
            font_thickness: the thickness of the font.
        """
        self.callback = callback
        self.font = font
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.title = title

        if title is None:
            self.title = str(self)

        self.image = None

    title: str
    """The title of the window."""

    image: Optional[np.ndarray]
    """The image currently being displayed."""

    def on_mouse_event(self, event, x, y, flags, param):
        # mouse move
        if event == cv2.EVENT_MOUSEMOVE:
            text = self.callback(self.image, x, y)
            image = self.image.copy()
            _draw_text(image, text, pos=(x, y), font=self.font, font_scale=self.font_scale, font_thickness=self.font_thickness)
            cv2.imshow(self.title, image)

    def run(self, image: np.ndarray):
        self.image = image
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.title, self.on_mouse_event)
        cv2.imshow(self.title, image)

        while True:
            key = cv2.waitKey(0)
            if key == 27 or key == ord('q'):
                break


def cv2_simple_text_visualizer(callback, image: np.ndarray, title: Optional[str] = None):
    """A simple wrapper for the SimpleTextVisualizer. See the documentation of :class:`SimpleTextVisualizer` for more details.

    Args:
        callback: the callback function.
        image: the image to be displayed.
        title: the title of the window.
    """
    if title is None:
        title = 'Simple Text Visualizer'
    visualizer = CV2SimpleTextVisualizer(callback, title=title)
    visualizer.run(image)


def cv2_simple_pixel_value_visualizer(image: np.ndarray, title: Optional[str] = None):
    """A simple visualizer for pixel values. The visualizer will display the image and the pixel value at the mouse position.

    Args:
        image: the image to be displayed.
        title: the title of the window.
    """
    def callback(f, x, y):
        return f'Pixel value: {f[y, x]} (x={x}, y={y})'
    if title is None:
        title = 'Simple Pixel Value Visualizer'
    return cv2_simple_text_visualizer(callback, image, title)
