#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : setup.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/09/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.
# Available at setup time due to pyproject.toml

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "0.5.1"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

ext_modules = [
    Pybind11Extension(
        "concepts._C",
        ["concepts/cc/main.cc"],
        define_macros=[('VERSION_INFO', __version__)],
    ),
]

setup(
    name="concepts",
    version=__version__,
    author="Jiayuan Mao",
    author_email="maojiayuan@gmail.com",
    url="https://concepts.jiayuanm.com",
    description="Concepts",
    long_description="",
    ext_modules=ext_modules,
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.10",
)
