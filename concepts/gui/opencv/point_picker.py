#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : point_picker.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/21/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Optional, Tuple, List

import cv2
import numpy as np
import jacinle.io as io

__all__ = ['CV2PointPicker', 'cv2_point_picker']


class CV2PointPicker(object):
    """A simple point picker using OpenCV.

    Example:

        .. code-block:: python

            import cv2
            import numpy as np
            from concepts.gui.opencv.point_picker import CV2PointPicker

            color_image = np.zeros((480, 640, 3), dtype=np.uint8)
            depth_image = np.zeros((480, 640), dtype=np.uint16)

            picker = CV2PointPicker()
            points = picker.run(color_image, depth_image)

    The order of the points corresponds to the order in which they were clicked.
    To remove a point, click on it again. The current checking radius is 10 pixels.
    """

    def __init__(self, registered_points: Optional[List[Tuple[int, int]]] = None):
        """Initialize the point picker.

        Args:
            registered_points: list of points to be registered. Defaults to None.
        """
        if registered_points is None:
            self.registered_points = []
        else:
            self.registered_points = list(registered_points)

    def register_point(self, event, x, y, flag, param):
        """Register a point based on the click by the user. See OpenCV documentation for :meth:`cv2.setMouseCallback` for more details.

        Args:
            event: the event type.
            x: the x-coordinate of the click.
            y: the y-coordinate of the click.
            flag: the flag.
            param: the parameter.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            found = None
            for i, (xx, yy) in enumerate(self.registered_points):
                if np.linalg.norm(np.array([xx, yy], dtype='float32') - np.array([x, y], dtype='float32')) < 10:
                    found = i
                    break

            if found is not None:
                self.registered_points = self.registered_points[:found] + self.registered_points[found + 1:]
            else:
                self.registered_points.append((x, y))

    def imshow(self, color_image: np.ndarray, depth_image: Optional[np.ndarray], window_name: Optional[str] = None) -> None:
        """Helper function to visualize the registered points on the color and depth images.

        Args:
            color_image: the color image.
            depth_image: the depth image.
            window_name: the name of the window. Defaults to None, in which case the name of the class is used.
        """

        color_image_new = color_image.copy()
        for x, y in self.registered_points:
            cv2.circle(color_image_new, (x, y), 5, (255, 0, 0), -1)

        if depth_image is None:
            # Form heatmap for depth and stack with color image
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )
            concat_images = np.hstack((color_image_new, depth_colormap))
        else:
            concat_images = color_image_new

        if window_name is None:
            window_name = str(self)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, concat_images)
        cv2.setMouseCallback(window_name, self.register_point)
        cv2.waitKey(1)

    def run(
        self, color_image: np.ndarray, depth_image: Optional[np.ndarray] = None,
        file_dirname: str = 'visualization', file_prefix: str = '', save: bool = False,
        window_name: Optional[str] = None
    ) -> List[Tuple[int, int]]:
        """Run the point picker.

        - To pick a point, click on the image.
        - To remove a point, click on it again.
        - Press 'c' to print the registered points, and save it to a file if `save` is True.
        - Press 'esc' or 'q' to exit. The function will return the registered points.

        If `save` is True, the visualization and the points will be saved to the specified directory:
        `{file_dirname}/{file_prefix}_visualization.png` and `{file_dirname}/{file_prefix}_points.pkl`.

        Args:
            color_image: the color image.
            depth_image: the depth image. Defaults to None. If specified, it will be shown side by side with the color image.
            file_dirname: the directory to save the visualization and the points. Defaults to 'visualization'.
            file_prefix: the prefix for the file names. Defaults to ''.
            save: whether to save the visualization and the points. Defaults to False.
            window_name: the name of the window. Defaults to None, in which case the name of the class is used.

        Returns:
            list of registered points.
        """
        depth_colormap = None
        if depth_image is not None:
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )
        if window_name is None:
            window_name = str(self)

        while True:
            color_image_new = color_image.copy()
            for x, y in self.registered_points:
                cv2.circle(color_image_new, (x, y), 5, (255, 0, 0), -1)

            # Form heatmap for depth and stack with color image
            if depth_image is not None:
                concat_images = np.hstack((color_image_new, depth_colormap))
            else:
                concat_images = color_image_new

            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, concat_images)
            cv2.setMouseCallback(window_name, self.register_point)
            key = cv2.waitKey(1)

            # Exit on 'esc' or 'q'
            if key & 0xFF == ord("q") or key == 27:
                cv2.destroyWindow(window_name)
                break

            if key == ord('c'):
                print('Registered points:')
                print('-' * 80)
                print(self.registered_points)

                if save:
                    cv2.imwrite(f'{file_dirname}/{file_prefix}visualization.png', concat_images)
                    io.dump(f'{file_dirname}/{file_prefix}points.pkl', self.registered_points)

        if save:
            cv2.imwrite(f'{file_dirname}/{file_prefix}visualization.png', concat_images)
            io.dump(f'{file_dirname}/{file_prefix}points.pkl', self.registered_points)

        cv2.destroyAllWindows()
        return self.registered_points

    def print_help(self):
        print()
        print('-' * 80)
        print('Click on the image to register a point.')
        print('Click on a already-registered point to remove it.')
        print('Press "c" to print the registered points. If save is True, the points will be saved to a file.')
        print('Press "esc" or "q" to exit.')
        print('-' * 80)
        print()


def cv2_point_picker(color_image: np.ndarray, depth_image: Optional[np.ndarray] = None, **kwargs) -> List[Tuple[int, int]]:
    """A simple point picker using OpenCV. See :class:`CV2PointPicker` for more details.

    Args:
        color_image: the color image.
        depth_image: the depth image. Defaults to None. If specified, it will be shown side by side with the color image.
        kwargs: additional keyword arguments.

    Returns:
        list of registered points.
    """
    picker = CV2PointPicker()
    return picker.run(color_image, depth_image, **kwargs)
