#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : execution_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/22/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Utility functions for executing expressions for PDSketch."""

from concepts.dsl.dsl_types import QINDEX
from concepts.dm.pdsketch.executor import PDSketchExecutor
from concepts.dm.pdsketch.predicate import Predicate
from concepts.dm.pdsketch.domain import State


def recompute_state_variable_predicates_(executor: PDSketchExecutor, state: State):
    """Recompute the state variable predicates for a given state.

    Args:
        executor: the executor.
        state: the state.
    """

    for predicate in executor.domain.functions.values():
        predicate: Predicate
        if predicate.is_state_variable and not predicate.is_observation_variable:
            assert predicate.is_derived
            bounded_variables = {v: QINDEX for v in predicate.arguments}
            state.features[predicate.name] = executor.execute(predicate.derived_expression, state=state, bounded_variables=bounded_variables)


def recompute_all_cacheable_predicates_(executor: PDSketchExecutor, state: State):
    """Recompute all cacheable predicates for a given state.

    Args:
        executor: the executor.
        state: the state.
    """

    for predicate in executor.domain.functions.values():
        predicate: Predicate
        if predicate.is_cacheable and predicate.is_derived and not predicate.is_state_variable:
            bounded_variables = {v: QINDEX for v in predicate.arguments}
            state.features[predicate.name] = executor.execute(predicate.derived_expression, state=state, bounded_variables=bounded_variables)

