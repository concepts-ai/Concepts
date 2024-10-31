#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test-deoxys-client.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/11/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from concepts.hw_interface.franka.deoxys_server import DeoxysClient


def main():
    client = DeoxysClient('localhost')

    client.single_move_qpos(1, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])


if __name__ == '__main__':
    main()

