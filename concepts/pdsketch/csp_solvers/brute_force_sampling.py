#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : brute_force_sampling.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 11/09/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""
This file corresponds to a brute-force sampling-based strategy for CSP solving.

Given a set of generators, and a CSP problem, we randomly sample a lot of values from the generators,
and then assign variables in the CSP problem to the sampled values (enumeratively). If the CSP problem
is satisfied, we return the assignment. Otherwise, we continue to sample and assign.

In practice, we do not directly implement this algorithm. When we are solving a task and motion planning problem
with continuous-parameterized operators, we first sample a large collection of possible continuous parameters,
and then reduce the problem into a basic discrete search problem.

The main function of this file is `generate_continuous_values`, which implements the sampling for continuous parameters
given the initial state and a collection of generators.
"""

import itertools
from typing import Tuple, List, Dict

import torch

from concepts.dsl.dsl_types import NamedTensorValueType
from concepts.dsl.tensor_value import TensorValue
from concepts.dsl.tensor_state import TensorState
from concepts.pdsketch.generator import Generator
from concepts.pdsketch.executor import PDSketchExecutor

__all__ = ['extract_generator_data', 'ContinuousValueDict', 'generate_continuous_values', 'expand_continuous_values_']


def extract_generator_data(executor: PDSketchExecutor, state: TensorState, generator: Generator) -> Tuple[List[TensorValue], List[TensorValue]]:
    """Extract the positive data from the state for a given generator. Specifically, it will test the `certifies` condition
    of the generator on the input state. If the condition is satisfied, it will extract the inputs (context) and outputs (generates)
    of the generator.

    Args:
        executor: a pdsketch expression executor.
        state: the input state.
        generator: the generator.

    Returns:
        a tuple of (inputs, outputs) for the generator.
    """
    result = executor.execute(generator.certifies, state, generator.arguments)
    result.tensor = torch.ge(result.tensor, 0.5)
    if result.tensor_mask is not None:
        result.tensor = torch.logical_and(result.tensor, torch.ge(result.tensor_mask, 0.5))

    def _index(value, mask):
        value = value.expand_as(mask)
        return value.tensor[mask.tensor]

    contexts = [_index(executor.execute(c, state, generator.arguments), result) for c in generator.context]
    generates = [_index(executor.execute(c, state, generator.arguments), result) for c in generator.generates]
    return contexts, generates


ContinuousValueDict = Dict[str, List[TensorValue]]


def generate_continuous_values(executor: PDSketchExecutor, state: TensorState, nr_iterations: int = 1, nr_samples: int = 5) -> ContinuousValueDict:
    """The function generate_continuous_values and expand_continuous_values jointly implements the incremental search
    algorithm for Task and Motion Planning.

    Basically, the algorithm starts from generating a large collection of possible continuous parameters by "expanding"
    from the continuous parameters in the input state. Next, it reduces the TAMP problem into a basic discrete search
    problem. The downside of this approach is that it requires grounding a large collection of possible values,
    but it is in theory probabilistically complete.

    Args:
        executor: a pdsketch expression executor.
        state: the input state.
        nr_iterations: the number of iterations.
        nr_samples: the number of samples for each generator.

    Returns:
        a dictionary of continuous values.
    """

    domain = executor.domain
    continuous_values = dict()
    for dtype in domain.types.values():
        if isinstance(dtype, NamedTensorValueType):
            continuous_values[dtype.typename] = list()

    for key, value in domain.functions.items():
        if key in state.features.all_feature_names and isinstance(value.return_type, NamedTensorValueType):
            dtype = value.return_type
            assert isinstance(dtype, NamedTensorValueType)
            feat = state.features[key].tensor
            feat = feat.reshape((-1, ) + feat.shape[-dtype.ndim():])
            continuous_values[dtype.typename].extend([TensorValue.from_tensor(x, dtype) for x in feat])

    for i in range(nr_iterations):
        expand_continuous_values_(executor, continuous_values, nr_samples=nr_samples)
    return continuous_values


def expand_continuous_values_(executor: PDSketchExecutor, current: ContinuousValueDict, nr_samples: int = 5):
    """Internal function used by :func:`generate_continuous_values`. Given the current set of continuous values,
    it tries to apply all generators using current values as their input. It inplace updates the `current` dictionary.
    with the new values.

    Args:
        executor: a pdsketch expression executor.
        current: the current continuous values.
        nr_samples: the number of samples for each generator.
    """

    domain = executor.domain
    for gen_name, gen_def in domain.generators.items():
        arguments = list()
        for arg in gen_def.context:
            assert isinstance(arg.return_type, NamedTensorValueType)
            arguments.append(current[arg.return_type.typename])
        for comb in itertools.product(*arguments):
            generator = executor.get_function_implementation(f'generator::{gen_name}').iter_from(*comb)
            for outputs in itertools.islice(generator, nr_samples):
                for output, output_def in zip(outputs, gen_def.generates):
                    assert isinstance(output_def.return_type, NamedTensorValueType)
                    current[output_def.return_type.typename].append(output)

