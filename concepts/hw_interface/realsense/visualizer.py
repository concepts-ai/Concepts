#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : visualizer.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/12/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os
import numpy as np
import cv2
import jacinle.io as io

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Optional, Tuple, List

if TYPE_CHECKING:
    from concepts.hw_interface.realsense.device import RealSenseDevice


class _WindowEvent(Enum):
    EXIT = "exit"
    SAVE = "save"
    NONE = "none"


class CV2Visualizer(object):
    def __init__(self, registered_points: Optional[List[Tuple[int, int]]] = None):
        if registered_points is None:
            self.registered_points = []
        else:
            self.registered_points = list(registered_points)

    def register_point(self, event, x, y, flag, param):
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

    def imshow(self, color_image: np.ndarray, depth_image: np.ndarray) -> None:
        color_image_new = color_image.copy()
        for x, y in self.registered_points:
            cv2.circle(color_image_new, (x, y), 5, (255, 0, 0), -1)

        # Form heatmap for depth and stack with color image
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )
        images = np.hstack((color_image_new, depth_colormap))

        window_name = str(self)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, images)
        cv2.setMouseCallback(window_name, self.register_point)
        cv2.waitKey(1)

    def run(self, color_image: np.ndarray, depth_image: Optional[np.ndarray] = None, file_dirname: str = 'visualization', file_suffix: str = '', save: bool = True) -> None:
        while True:
            color_image_new = color_image.copy()
            for x, y in self.registered_points:
                cv2.circle(color_image_new, (x, y), 5, (255, 0, 0), -1)

            # Form heatmap for depth and stack with color image
            if depth_image is not None:
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
                )
                images = np.hstack((color_image_new, depth_colormap))
            else:
                images = color_image_new

            window_name = str(self)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, images)
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
                    cv2.imwrite(f'{file_dirname}/calibration{file_suffix}.png', images)
                    io.dump(f'{file_dirname}/calibration_image_points{file_suffix}.pkl', self.registered_points)

        if save:
            cv2.imwrite(f'{file_dirname}/calibration{file_suffix}.png', images)
            io.dump(f'{file_dirname}/calibration_image_points{file_suffix}.pkl', self.registered_points)

        cv2.destroyAllWindows()
        return self.registered_points


class RealSenseVisualizer(object):
    def __init__(self, device):
        self.device: 'RealSenseDevice' = device

    def visualize(self, save_dir: str = "", save_image: bool = False) -> _WindowEvent:
        """
        Visualize color and depth images in a cv2 window.
        Terminates when 'esc' or 'q' key is pressed.
        Saves an image when the 's' key is pressed or if the 'save_image'
        flag is specified.
        Images are saved to the specified save_dir, which we
        assume to already exist.
        Returns a _WindowEvent indicating status of cv2.
        """
        color_image, depth_image = self.device.capture_images()

        # Form heatmap for depth and stack with color image
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )
        images = np.hstack((color_image, depth_colormap))

        window_name = str(self)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, images)
        cv2.setMouseCallback(window_name, self.register_point)
        key = cv2.waitKey(1)

        # Exit on 'esc' or 'q'
        if key & 0xFF == ord("q") or key == 27:
            cv2.destroyWindow(window_name)
            return _WindowEvent.EXIT

        # Save images if 's' key is pressed
        if key == 115 or save_image:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            depth_fname = os.path.join(
                save_dir, f"{timestamp}-{self.device.serial_number}-depth.png"
            )
            color_fname = os.path.join(
                save_dir, f"{timestamp}-{self.device.serial_number}-color.png"
            )
            cv2.imwrite(depth_fname, depth_image)
            cv2.imwrite(color_fname, color_image)
            print(f"Saved depth image for {self} to {depth_fname}")
            print(f"Saved color image to {self} to {color_fname}")
            return _WindowEvent.SAVE

        if key == ord('c'):
            print('CALIBRATION registered points:')
            print('-' * 80)
            print(self.registered_points)

        return _WindowEvent.NONE


def visualize_devices(devices: List['RealSenseDevice'], save_dir: str = "") -> None:
    """
    Visualizes all the devices in a cv2 window. Press 'q' or 'esc' on any of
    the windows to exit the infinite loop.
    Press the 's' key in a specific window to save the color and depth image
    to disk.
    You can use a similar loop interface in other places where you need
    a live camera feed (e.g. collecting demonstrations).
    """
    while True:
        should_exit = False
        save = False

        for device in devices:
            window_event = device.visualize(save_dir, save_image=save)
            if window_event == _WindowEvent.SAVE:
                # We use this to propagate a save command across all windows
                save = True

            # Exit all windows
            if window_event == _WindowEvent.EXIT:
                should_exit = True
                break

        if should_exit:
            print("Exit key pressed.")
            cv2.destroyAllWindows()
            break


def run_4corner_calibration(devices: List['RealSenseDevice'], save_dir: str = "") -> None:
    while True:
        should_exit = False

        for device in devices:
            window_event = device.visualize(save_dir, save_image=False)
            # Exit all windows
            if window_event == _WindowEvent.EXIT:
                should_exit = True
                break

        if should_exit:
            print("Exit key pressed.")
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    import jacinle
    from concepts.hw_interface.realsense.device import start_pipelines, stop_pipelines

    parser = jacinle.JacArgumentParser()
    parser.add_argument('--device', choices=['l515', 'd435'], default='d435')
    args = parser.parse_args()

    # Detect devices, start pipelines, visualize, and stop pipelines
    devices = RealSenseDevice.find_devices(args.device)
    start_pipelines(devices)

    try:
        visualize_devices(devices)
    finally:
        stop_pipelines(devices)

