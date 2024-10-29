#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : point_picker.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/29/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Tuple
from PIL import Image, ImageTk

import numpy as np
import tkinter as tk


def get_click_coordinates_from_image(image: Image.Image, min_dimension: int = 0) -> Tuple[int, int]:
    """Opens an image in a UI window and waits for the user to click on the image.
    Returns the coordinates of the click and closes the window.

    Args:
        image: a PIL Image object.
        min_dimension: minimum dimension of the image window. If the image is smaller than this, it will be
            resized to fit this dimension while maintaining the aspect ratio.

    Returns:
        a tuple containing the u and v coordinates of the click (axis 1 and axis 0).
    """

    # Define a simple class to hold the application's state.
    class ImageClickApp:
        def __init__(self, master, img):
            self.master = master
            self.clicked_coords = None

            # Load and display the image.
            img_tk = ImageTk.PhotoImage(img)
            self.lbl = tk.Label(master, image=img_tk)
            self.lbl.pack()

            # Bind the click event.
            self.lbl.bind("<Button-1>", self.on_click)

            # Keep a reference to prevent garbage-collection.
            self.lbl.img_tk = img_tk

        def on_click(self, event):
            # Store the coordinates and close the window.
            self.clicked_coords = (event.x, event.y)
            self.master.destroy()

    # Resize the image if it is smaller than the minimum dimension.
    scaling_factor = (1, 1)

    if min(image.size) < min_dimension:
        old_size = image.size
        aspect_ratio = image.size[0] / image.size[1]
        new_size = (min_dimension, int(min_dimension / aspect_ratio))
        image = image.resize(new_size)
        scaling_factor = new_size[0] / old_size[0], new_size[1] / old_size[1]

    # Create the Tkinter window.
    root = tk.Tk()
    app = ImageClickApp(root, image)

    # Run the event loop and wait for it to finish.
    root.mainloop()

    # Return the coordinates after the window has been closed.
    coords = app.clicked_coords
    if coords is None:
        raise ValueError("No coordinates were clicked.")
    return int(round(coords[0] / scaling_factor[0])), int(round(coords[1] / scaling_factor[1]))


def get_click_coordinates_from_image_path(image_path: str, min_dimension: int = 0) -> Tuple[int, int]:
    """Opens an image in a UI window and waits for the user to click on the image.
    Returns the coordinates of the click and closes the window.

    Args:
        image_path: path to the image file.
        min_dimension: minimum dimension of the image window. If the image is smaller than this, it will be
            resized to fit this dimension while maintaining the aspect ratio.

    Returns:
        a tuple containing the u and v coordinates of the click (axis 1 and axis 0).
    """

    image = Image.open(image_path)
    return get_click_coordinates_from_image(image, min_dimension)


def get_click_coordinates_from_array(image_array: np.ndarray, min_dimension: int = 0) -> Tuple[int, int]:
    """Opens an image (from a numpy array) in a UI window and waits for the user to click on the image.
    Returns the coordinates of the click and closes the window.

    Args:
        image_array: a numpy array representing the image.
        min_dimension: minimum dimension of the image window. If the image is smaller than this, it will be
            resized to fit this dimension while maintaining the aspect ratio.

    Returns:
        a tuple containing the u and v coordinates of the click (axis 1 and axis 0).
    """

    image = Image.fromarray(image_array)
    return get_click_coordinates_from_image(image, min_dimension)


# Example usage:
if __name__ == "__main__":
    image_path = "image_scene/hook_with_ball.png"  # Change this to the path of your image.
    coordinates = get_click_coordinates_from_image_path(image_path)
    print(f"Clicked coordinates: {coordinates}")

