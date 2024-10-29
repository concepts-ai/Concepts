#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : drawing_window.py
# Author : Zhezheng Luo
# Email  : ezzluo@mit.edu
# Date   : 04/23/2021
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

# DrawingWindow.py - Michael Haimes, Leslie Kaelbling                      #
# Hacked by LPK to be independent of SoaR                                  #
############################################################################
#    MIT SoaR 2.0 - A Python abstraction layer for MobileRobots            #
#    Pioneer3DX robots, and a simulator to simulate their operation        #
#    in a python environment (for testing)                                 #
#                                                                          #
#    Copyright (C) 2006-2007 Michael Haimes <mhaimes@mit.edu>              #
#                                                                          #
#   This program is free software; you can redistribute it and/or modify   #
#   it under the terms of the GNU General Public License as published by   #
#   the Free Software Foundation; either version 2 of the License, or      #
#   (at your option) any later version.                                    #
#                                                                          #
#   This program is distributed in the hope that it will be useful,        #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of         #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          #
#   GNU General Public License for more details.                           #
#                                                                          #
#   You should have received a copy of the GNU General Public License along#
#   with this program; if not, write to the Free Software Foundation, Inc.,#
#   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.            #
############################################################################

__all__ = ['DrawingWindow']


from tkinter import *
import math


class Thing:
    pass


class DrawingWindow:
    def __init__(self, windowWidth, windowHeight, xMin, xMax, yMin, yMax, title, parent=None):
        self.title = title
        if parent:
            self.parent = parent
            self.top = parent.getWindow(title)
        else:
            self.tk = Tk()
            self.tk.withdraw()
            self.top = Toplevel(self.tk)
            self.top.wm_title(title)
            self.top.protocol('WM_DELETE_WINDOW', self.top.destroy)

        self.windowWidth = windowWidth
        self.windowHeight = windowHeight
        self.canvas = Canvas(self.top, width=self.windowWidth, height=self.windowHeight, background="white")
        self.canvas.pack()

        # multiply an input value by this to get pixels
        self.xScale = windowWidth / float(xMax - xMin)
        self.yScale = windowHeight / float(yMax - yMin)

        self.drawingframe = Thing()
        self.drawingframe.width_px = windowWidth

        self.xMin = xMin
        self.yMin = yMin
        self.xMax = xMax
        self.yMax = yMax

    def scaleX(self, x):
        return self.xScale * (x - self.xMin)

    def scaleY(self, y):
        return self.windowHeight - self.yScale * (y - self.yMin)

    def drawPoint(self, x, y, color="blue", radius=1):
        windowX = self.scaleX(x)
        windowY = self.scaleY(y)
        return self.canvas.create_rectangle(
            windowX - radius, windowY - radius, windowX + radius, windowY + radius, fill=color, outline=color
        )

    def drawRobotWithNose(self, x, y, theta, color="blue", size=6):
        rawx = math.cos(theta)
        rawy = math.sin(theta)
        hx, hy = 0.15, 0.0
        noseX = x + rawx * hx - rawy * hy
        noseY = y + rawy * hx + rawx * hy
        return self.drawRobot(x, y, noseX, noseY, color=color, size=size)

    def drawRobot(self, x, y, noseX, noseY, color="blue", size=8):
        windowX = self.scaleX(x)
        windowY = self.scaleY(y)
        hsize = int(size) / 2  # For once, we want the int division!
        return (
            self.canvas.create_rectangle(
                windowX - hsize, windowY - hsize, windowX + hsize, windowY + hsize, fill=color, outline=color
            ),
            self.canvas.create_line(
                windowX, windowY, self.scaleX(noseX), self.scaleY(noseY), fill=color, width=2, arrow="last"
            ),
        )

    def drawText(self, x, y, label):
        windowX = self.scaleX(x)
        windowY = self.scaleY(y)
        # font="Arial 20",fill="#ff0000"
        return self.canvas.create_text(windowX, windowY, text=label)

    def drawPolygon(self, verts, color="black", outline="black"):
        return self.canvas.create_polygon(
            [(self.scaleX(point.x), self.scaleY(point.y)) for point in verts], fill=color, outline=outline
        )

    def drawRect(self, x1, y1, x2, y2, color="black"):
        return self.canvas.create_rectangle(
            self.scaleX(x1), self.scaleY(y1), self.scaleX(x2), self.scaleY(y2), fill=color
        )

    def drawOval(self, x1, y1, x2, y2, color="black"):
        return self.canvas.create_oval(self.scaleX(x1), self.scaleY(y1), self.scaleX(x2), self.scaleY(y2), fill=color)

    def drawLineSeg(self, x1, y1, x2, y2, color="black", width=2):
        return self.canvas.create_line(
            self.scaleX(x1), self.scaleY(y1), self.scaleX(x2), self.scaleY(y2), fill=color, width=width
        )

    def drawUnscaledLineSeg(self, x1, y1, xproj, yproj, color="black", width=1):
        return self.canvas.create_line(
            self.scaleX(x1), self.scaleY(y1), self.scaleX(x1) + xproj, self.scaleY(y1) - yproj, fill=color, width=width
        )

    def drawUnscaledRect(self, x1, y1, xproj, yproj, color="black"):
        return self.canvas.create_rectangle(
            self.scaleX(x1) - xproj,
            self.scaleY(y1) + yproj,
            self.scaleX(x1) + xproj,
            self.scaleY(y1) - yproj,
            fill=color,
        )

    def drawLine(self, a, b, c, color="black"):
        if abs(b) < 0.001:
            startX = self.scaleX(-c / a)
            endX = self.scaleX(-c / a)
            startY = self.scaleY(self.yMin)
            endY = self.scaleY(self.yMax)
        else:
            startX = self.scaleX(self.xMin)
            startY = self.scaleY(-(a * self.xMin + c) / b)
            endX = self.scaleX(self.xMax)
            endY = self.scaleY(-(a * self.xMax + c) / b)
        return self.canvas.create_line(startX, startY, endX, endY, fill=color)

    def delete(self, thing):
        self.canvas.delete(thing)

    def clear(self):
        self.canvas.delete("all")

    def update(self):
        self.canvas.update()
