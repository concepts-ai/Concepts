#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : namo_polygon_map_creator.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/26/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import json
import numpy as np
import cv2
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import triangulate, unary_union

from typing import Optional, List


class NamoMapCreator(object):
    COLOR_FREE = (255, 255, 255)
    COLOR_WALL = (0, 0, 0)
    COLOR_OBSTACLE = (255, 128, 128)  # Blue
    COLOR_REMOVING = (128, 128, 255)  # Yellow

    AVAILABLE_MODES = ('poly', 'rect')

    KEY_BINDINGS_DESCRIPTION = r"""
Key bindings:

- 'q': quit the current drawing process and enter the idle status.
- 'a': start drawing an obstacle.
- 'w': start drawing a wall.
- 'd': start removing.
- 'z': set the start position.
- 'x': set the goal position.
- 'f': finish the current drawing.
- ESC: quit the program.
"""

    def __init__(self, mode='poly', edit=None, width=800, height=800):
        self.mode = mode
        self.width = width
        self.height = height
        self.margin = 50
        self.border = 2
        self.actual_border_width = 10
        self.total_margin = self.margin + self.border

        self._status = 'idle'
        self._mouse_pos_sequence = list()
        self._current_mouse_pos = None
        self._start_pos = None
        self._goal_pos = None
        self._obstacles = list()
        self._walls = list()

        if edit is not None:
            self._load_map(edit)
        else:
            self._add_initial_walls()

    def _load_map(self, filename: str):
        with open(filename, 'r') as f:
            data = json.load(f)

        print('Loaded map from:', filename)

        self.width = data['width']
        self.height = data['height']
        self._obstacles = [np.array(polygon) for polygon in data['obstacles']]
        self._walls = [np.array(polygon) for polygon in data['raw_walls']]
        self._start_pos = data['start']
        self._goal_pos = data['goal']

        print('Map size:', self.width, 'x', self.height)
        print('Number of obstacles:', len(self._obstacles))
        print('Number of walls:', len(self._walls))

    def _add_initial_walls(self):
        self._walls.append(self._gen_poly_from_mouse_sequence([(0, 0), (self.width, self.actual_border_width)]))
        self._walls.append(self._gen_poly_from_mouse_sequence([(0, self.height - self.actual_border_width), (self.width, self.height)]))
        self._walls.append(self._gen_poly_from_mouse_sequence([(0, 0), (self.actual_border_width, self.height)]))
        self._walls.append(self._gen_poly_from_mouse_sequence([(self.width - self.actual_border_width, 0), (self.width, self.height)]))

    def _make_image(self):
        image = np.zeros((self.height + self.margin * 2 + 4, self.width + self.margin * 2 + 4, 3), dtype=np.uint8)
        image[:, :] = self.COLOR_FREE

        # Draw the border
        image[self.total_margin - self.border:self.total_margin, :] = self.COLOR_WALL
        image[-self.total_margin:-self.total_margin + self.border, :] = self.COLOR_WALL
        image[:, self.total_margin - self.border:self.total_margin] = self.COLOR_WALL
        image[:, -self.total_margin:-self.total_margin + self.border] = self.COLOR_WALL

        for polygon in self._obstacles:
            self._make_polygon(image, polygon, type='obstacle')
        for polygon in self._walls:
            self._make_polygon(image, polygon, type='wall')

        if self._start_pos is not None:
            cv2.circle(image, (self._start_pos[0] + self.total_margin, self._start_pos[1] + self.total_margin), 15, (255, 0, 0), -1)
        if self._goal_pos is not None:
            cv2.circle(image, (self._goal_pos[0] + self.total_margin, self._goal_pos[1] + self.total_margin), 15, (0, 255, 0), -1)

        # Draw the current drawing
        if self._status == 'drawing_obstacle':
            self._make_polygon(image, np.array(self._mouse_pos_sequence), type='obstacle', temporary=True)
        elif self._status == 'drawing_wall':
            self._make_polygon(image, np.array(self._mouse_pos_sequence), type='wall', temporary=True)
        elif self._status == 'removing':
            self._make_polygon(image, np.array(self._mouse_pos_sequence), type='removing', temporary=True)

        return image

    def _gen_poly_from_mouse_sequence(self, mouse_sequence):
        if self.mode == 'poly':
            return np.array(mouse_sequence)
        elif self.mode == 'rect':
            return np.array([mouse_sequence[0], (mouse_sequence[-1][0], mouse_sequence[0][1]), mouse_sequence[-1], (mouse_sequence[0][0], mouse_sequence[-1][1])])
        else:
            raise ValueError('Invalid mode: {}'.format(self.mode))

    def _make_polygon(self, image, polygon, type='obstacle', temporary: bool = False):
        polygon = polygon + self.total_margin

        if type == 'obstacle':
            color = self.COLOR_OBSTACLE
        elif type == 'wall':
            color = self.COLOR_WALL
        elif type == 'removing':
            color = self.COLOR_REMOVING

        if temporary:
            if len(polygon) > 0:
                polygon = np.concatenate([polygon, [(self._current_mouse_pos[0] + self.total_margin, self._current_mouse_pos[1] + self.total_margin)]], axis=0)
                polygon = self._gen_poly_from_mouse_sequence(polygon)
                cv2.polylines(image, np.int32([polygon]), isClosed=True, color=color, thickness=2)
        else:
            cv2.fillPoly(image, np.int32([polygon]), color)

    _status: str
    """The status of the creator. Can be 'idle', 'drawing_obstacle', 'drawing_wall', or 'removing'."""

    def mainloop(self):
        print(self.KEY_BINDINGS_DESCRIPTION)
        cv2.namedWindow('map', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('map', self._mouse_callback)
        while True:
            cv2.imshow('map', self._make_image())
            key = cv2.waitKey(5)
            q = self._key_callback(key)
            if q:
                break
        return self.encode_map()

    def encode_map(self):
        """Encode the map as a dictionary."""
        return {
            'width': self.width,
            'height': self.height,
            'obstacles': _to_list(self._obstacles),
            'raw_walls': _to_list(self._walls),
            'walls': _to_list(polygon_union(self._walls)),
            'start': self._start_pos,
            'goal': self._goal_pos
        }

    def encode_map_image(self) -> np.ndarray:
        """Encode the map as an image."""
        image = self._make_image()
        # Exclude the added border.
        return image[self.total_margin:self.total_margin + self.height, self.total_margin:self.total_margin + self.width]

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self._status in ('drawing_obstacle', 'drawing_wall', 'removing'):
                self._mouse_pos_sequence.append((x - self.total_margin, y - self.total_margin))
                if len(self._mouse_pos_sequence) > 1 and self.mode == 'rect':
                    self._finish_drawing()
            elif self._status == 'setting_start':
                self._start_pos = (x - self.total_margin, y - self.total_margin)
                self._status = 'idle'
            elif self._status == 'setting_goal':
                self._goal_pos = (x - self.total_margin, y - self.total_margin)
                self._status = 'idle'
            else:
                print('Please start drawing first.')
        elif event == cv2.EVENT_MOUSEMOVE:
            self._current_mouse_pos = (x - self.total_margin, y - self.total_margin)

    def _key_callback(self, key) -> bool:
        """Handle the key events."""

        if key == ord('q'):
            self._status = 'idle'
            self._mouse_pos_sequence.clear()
        elif key == ord('z'):
            self._status = 'setting_start'
        elif key == ord('x'):
            self._status = 'setting_goal'
        elif key == ord('a'):
            if len(self._mouse_pos_sequence) == 0:
                self._status = 'drawing_obstacle'
            else:
                print('Quit or finish the current drawing first.')
        elif key == ord('w'):
            if len(self._mouse_pos_sequence) == 0:
                self._status = 'drawing_wall'
            else:
                print('Quit or finish the current drawing first.')
        elif key == ord('d'):
            if len(self._mouse_pos_sequence) == 0:
                self._status = 'removing'
            else:
                print('Quit or finish the current drawing first.')
        elif key == ord('f'):
            self._finish_drawing()
        elif key == 27:
            return True

        return False

    def _finish_drawing(self):
        new_polygon = self._gen_poly_from_mouse_sequence(self._mouse_pos_sequence)
        if self._status == 'drawing_obstacle':
            poly = intersect_polygon(new_polygon, np.array([[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]]))
            if poly is not None:
                self._obstacles.append(poly)
        elif self._status == 'drawing_wall':
            poly = intersect_polygon(new_polygon, np.array([[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]]))
            if poly is not None:
                self._walls.append(poly)
        elif self._status == 'removing':
            new_obstacles = list()
            for polygon in self._obstacles:
                new_obstacles.extend(subtract_polygon(polygon, new_polygon))
            self._obstacles = new_obstacles
            new_walls = list()
            new_walls.extend(self._walls[:4]) # Skip the first four walls (corresponding to the outer boundary).
            for polygon in self._walls[4:]:
                new_walls.extend(subtract_polygon(polygon, new_polygon))
            self._walls = new_walls
        self._mouse_pos_sequence.clear()


def intersect_polygon(polygon1: np.ndarray, polygon2: np.ndarray) -> Optional[np.ndarray]:
    """Return a new polygon that is the intersection of two polygons."""
    """
    Return a new polygon that is the intersection of two polygons.

    Args:
        polygon1 (numpy.ndarray): Nx2 array representing the first polygon.
        polygon2 (numpy.ndarray): Nx2 array representing the second polygon.

    Returns:
        Mx2 array representing the intersection polygon.
    """
    # Convert the numpy arrays to Shapely polygons
    poly1 = Polygon(polygon1)
    poly2 = Polygon(polygon2)

    # Compute the intersection
    intersection = poly1.intersection(poly2)

    # If the intersection is empty, return an empty array
    if intersection.is_empty:
        return None

    # Convert the intersection polygon to a numpy array
    intersection_coords = np.array(intersection.exterior.coords)

    return intersection_coords.astype(np.int32)


def subtract_polygon(polygon1: np.ndarray, polygon2: np.ndarray) -> List[np.ndarray]:
    """Return a new polygon that is the difference of two polygons."""
    """
    Return a new polygon that is the difference of two polygons.

    Args:
        polygon1 (numpy.ndarray): Nx2 array representing the first polygon.
        polygon2 (numpy.ndarray): Nx2 array representing the second polygon.

    Returns:
        a list Mx2 array representing the difference polygon, possibly empty.
    """
    # Convert the numpy arrays to Shapely polygons
    poly1 = Polygon(polygon1)
    poly2 = Polygon(polygon2)

    # Compute the difference
    difference = poly1.difference(poly2)

    # If the difference is empty, return an empty array
    if difference.is_empty:
        return list()

    all_new_polygons = list()
    if isinstance(difference, MultiPolygon):
        for poly in difference.geoms:
            all_new_polygons.append(np.array(poly.exterior.coords, dtype=np.int32))
    else:
        all_new_polygons.append(np.array(difference.exterior.coords, dtype=np.int32))

    return all_new_polygons


def polygon_remove_intersections(polygons: List[np.ndarray]) -> List[np.ndarray]:
    """For each polygon in the list of polygons, remove any intersections with other polygons."""

    # Convert the list of numpy arrays to a list of Shapely polygons
    shapely_polygons = [Polygon(polygon) for polygon in polygons]

    # Remove any intersections between the polygons
    for i, polygon in enumerate(shapely_polygons):
        for other_polygon in shapely_polygons[:i]:
            shapely_polygons[i] = shapely_polygons[i].difference(other_polygon)

    # Convert the convex decomposition to a list of numpy arrays
    convex_polygons = [np.array(polygon.exterior.coords) for polygon in shapely_polygons]
    return convex_polygons


def polygon_union(polygons: List[np.ndarray]) -> List[np.ndarray]:
    """Take the union of a list of polygons and return a list of convex polygons."""
    if len(polygons) == 0:
        return []

    # Convert the list of numpy arrays to a list of Shapely polygons
    shapely_polygons = [Polygon(polygon) for polygon in polygons]

    union = unary_union(shapely_polygons)
    if isinstance(union, MultiPolygon):
        union_pieces = union.geoms
    elif isinstance(union, Polygon):
        union_pieces = [union]
    else:
        raise ValueError('Invalid output type: {}'.format(type(union)))

    convex_triangles = list()
    for piece in union_pieces:
        convex_triangles.extend(polygon_convex_decomposition(piece))

    return convex_triangles


def polygon_convex_decomposition(polygon: Polygon) -> List[np.ndarray]:
    """Decompose a polygon into a list of convex polygons."""
    """
    Decompose a polygon into a list of convex polygons.

    Args:
        polygon: a Shapely polygon.

    Returns:
        A list of numpy arrays, each representing a convex polygon.
    """

    triangles = [triangle for triangle in triangulate(polygon) if triangle.within(polygon)]
    convex_triangles = [np.array(triangle.exterior.coords) for triangle in triangles]
    return convex_triangles


def _to_list(polygons: List[np.ndarray]) -> List[List[List[int]]]:
    """Convert a list of polygons to a list of lists of lists."""
    return [[[int(x), int(y)] for x, y in polygon] for polygon in polygons]

