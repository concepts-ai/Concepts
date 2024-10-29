#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/24/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""This module contains CSP solvers for PDSketch.
These solvers will be primarily used in search-and-optimization problems in the task and motion planning problem,
although some strategies and solvers can be used in general domains as well.

Current solvers include:

- Brute-force sampling-based CSP solver. See :mod:`concepts.dm.pdsketch.csp_solvers.brute_force_sampling`.
- DPLL-sampling-based CSP solver (a generalization of the DPLL algorithm to mixed discrete-continuous CSPs). See :mod:`concepts.dm.pdsketch.csp_solvers.dpll_sampling`.
"""

from .dpll_sampling import csp_dpll_sampling_solve, csp_dpll_simplify

