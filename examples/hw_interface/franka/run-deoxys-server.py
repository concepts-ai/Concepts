#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : run-deoxys-server.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/11/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from concepts.hw_interface.franka.deoxys_server import DeoxysService


def main():
    server = DeoxysService({}, mock=True)
    server.serve_socket()


if __name__ == '__main__':
    main()

