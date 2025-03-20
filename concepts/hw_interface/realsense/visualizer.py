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
import time

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, List
from concepts.hw_interface.realsense.device import RealSenseInterface, get_concat_rgbd_visualization

__all__ = ['WindowEvent', 'RealSenseVisualizer', 'visualize_devices', 'run_4corner_calibration']


class WindowEvent(Enum):
    EXIT = "exit"
    SAVE = "save"
    NONE = "none"


class RealSenseVisualizer(object):
    def __init__(self, device):
        self.device: RealSenseInterface = device

    def run(self, save_dir: str = "", save_image: bool = False) -> WindowEvent:
        """Visualize color and depth images in a cv2 window.

        - Terminates when 'esc' or 'q' key is pressed.
        - Saves an image when the 's' key is pressed or if the 'save_image' flag is specified.
        - Images are saved to the specified save_dir, which we assume to already exist.

        Returns:
            WindowEvent indicating status of cv2.
        """
        color_image, depth_image = self.device.get_rgbd_image(format='bgr')

        # Form heatmap for depth and stack with color image
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )
        images = np.hstack((color_image, depth_colormap))

        window_name = str(self)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, images)
        key = cv2.waitKey(1)

        # Exit on 'esc' or 'q'
        if key & 0xFF == ord("q") or key == 27:
            cv2.destroyWindow(window_name)
            return WindowEvent.EXIT

        # Save images if 's' key is pressed
        if key == 115 or save_image:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            depth_fname = os.path.join(
                save_dir, f"{timestamp}-{self.device.get_serial_number()}-depth.png"
            )
            color_fname = os.path.join(
                save_dir, f"{timestamp}-{self.device.get_serial_number()}-color.png"
            )
            cv2.imwrite(depth_fname, depth_image)
            cv2.imwrite(color_fname, color_image)
            print(f"Saved depth image for {self} to {depth_fname}")
            print(f"Saved color image to {self} to {color_fname}")
            return WindowEvent.SAVE

        return WindowEvent.NONE


class RealSenseStreamVisualizer(object):
    def __init__(self, capture: RealSenseInterface, fps: int = 6):
        self.window_name = 'RealSense Capture'
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        self.capture = capture
        self.fps = fps
        self.last_frame_time = 0
        self.recording = False
        self.recorded = list()

    def show(self, img):
        img = img.copy()

        if self.recording:
            # Draw a red rectangle around the image
            cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, 0, 255), 2)

        cv2.imshow(self.window_name, img)
        if self.last_frame_time != 0:
            time_diff = cv2.getTickCount() - self.last_frame_time
            delay = max(1, (1000 // self.fps) - int(time_diff / cv2.getTickFrequency() * 1000))
        else:
            delay = 1000 // self.fps
        self.last_frame_time = cv2.getTickCount()
        key = cv2.waitKey(delay)
        return key

    def close(self):
        cv2.destroyWindow(self.window_name)

    def print_help(self):
        print()
        print('-' * 80)
        print('RealSense Capture Visualizer')
        print('Press ESC or q to exit')
        print('Press r to toggle recording')
        print('-' * 80)
        print()

    def run(self):
        self.print_help()

        frame_count = 0
        while True:
            frame_count += 1
            rgb, depth = self.capture.get_rgbd_image(format='rgb')

            if self.recording:
                self.recorded.append((rgb, depth))

            if self.recording:
                current_time = time.strftime('[REC] Frame {:04d} %H:%M:%S'.format(frame_count), time.localtime())
            else:
                current_time = time.strftime('Frame {:04d} %H:%M:%S'.format(frame_count), time.localtime())

            frame = get_concat_rgbd_visualization(rgb[..., ::-1], depth)  # To BGR
            cv2.putText(frame, current_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            frame = cv2.resize(frame, (960, int(960 * frame.shape[0] / frame.shape[1])))

            key = self.show(frame)
            if key == 27 or key == ord('q'):
                break
            if key == ord('r'):
                self.recording = not self.recording
        self.close()


def visualize_devices(devices: List[RealSenseInterface], save_dir: str = "") -> None:
    """Visualizes all the devices in a cv2 window. Press 'q' or 'esc' on any of the windows to exit the infinite loop.
    Press the 's' key in a specific window to save the color and depth image to disk.
    You can use a similar loop interface in other places where you need a live camera feed (e.g. collecting demonstrations).
    """
    while True:
        should_exit = False
        save = False

        for device in devices:
            window_event = RealSenseVisualizer(device).run(save_dir, save_image=save)
            if window_event == WindowEvent.SAVE:
                # We use this to propagate a save command across all windows
                save = True

            # Exit all windows
            if window_event == WindowEvent.EXIT:
                should_exit = True
                break

        if should_exit:
            print("Exit key pressed.")
            cv2.destroyAllWindows()
            break


def run_4corner_calibration(devices: List[RealSenseInterface], save_dir: str = "") -> None:
    while True:
        should_exit = False

        for device in devices:
            window_event = RealSenseVisualizer(device).run(save_dir, save_image=False)
            # Exit all windows
            if window_event == WindowEvent.EXIT:
                should_exit = True
                break

        if should_exit:
            print("Exit key pressed.")
            cv2.destroyAllWindows()
            break


def _main():
    import jacinle
    from concepts.hw_interface.realsense.device import start_pipelines, stop_pipelines

    parser = jacinle.JacArgumentParser()
    parser.add_argument('--device', choices=['l515', 'd435'], default='d435')
    args = parser.parse_args()

    # Detect devices, start pipelines, visualize, and stop pipelines
    from concepts.hw_interface.realsense.device import RealSenseDevice
    devices = RealSenseDevice.find_devices(args.device)
    start_pipelines(devices)

    try:
        visualize_devices(devices)
    finally:
        stop_pipelines(devices)


if __name__ == "__main__":
    _main()
