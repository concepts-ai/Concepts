#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : crow_generator.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/16/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""A generator is the "inverse" function of a black-box function. It is used in solving constraint satisfaction problems.
In particular, a generator matches a set of constraints, and will generate a set of values that satisfy the constraints.

There are two types of generators:

- directed generators: the function implementation takes a particular set of values, and generates the rest of the values.
- undirected generators: the function implementation takes an arbitrary subset of values, and generates a set of values that satisfy the constraints.
"""

from typing import Optional, Union, Sequence, Tuple

from jacinle.utils.meta import repr_from_str
from concepts.dsl.dsl_types import ObjectType, ValueType, Variable, TupleType
from concepts.dsl.dsl_functions import FunctionType
from concepts.dsl.expression import ValueOutputExpression, ObjectOrValueOutputExpression
from concepts.dm.crow.crow_function import CrowFunctionEvaluationMode

__all__ = ['CrowGeneratorBase', 'CrowDirectedGenerator', 'CrowUndirectedGenerator', 'CrowGeneratorApplicationExpression']


class CrowGeneratorBase(object):
    def __init__(
        self, name: str,
        all_arguments: Sequence[Variable],
        certifies: Sequence[ValueOutputExpression],
        priority: int = 0,
        simulation: bool = False, execution: bool = False,
    ):
        self.name = name
        self.all_arguments = tuple(all_arguments)
        self.certifies = tuple(certifies)
        self.priority = priority
        self.evaluation_mode = CrowFunctionEvaluationMode.from_bools(simulation, execution)

    name: str
    """The name of the generator."""

    all_arguments: Sequence[Variable]
    """The complete list of arguments that the generator takes."""

    certifies: Sequence[ValueOutputExpression]
    """The list of expressions that the generator certifies."""

    priority: int
    """The priority of the generator. Generators with higher priority will be executed first."""

    evaluation_mode: CrowFunctionEvaluationMode
    """The evaluation mode of the generator."""

    @property
    def argument_names(self) -> Tuple[str, ...]:
        return tuple(arg.name for arg in self.all_arguments)

    @property
    def argument_types(self) -> Tuple[Union[ObjectType, ValueType], ...]:
        return tuple(arg.dtype for arg in self.all_arguments)

    def __str__(self):
        flag = self.evaluation_mode.get_prefix()
        return f'{self.name}{flag}({", ".join(map(str, self.all_arguments))}) -> {", ".join(map(str, self.certifies))}'

    __repr__ = repr_from_str


class CrowDirectedGenerator(CrowGeneratorBase):
    def __init__(
        self, name: str, all_arguments: Sequence[Variable], certifies: Sequence[ValueOutputExpression],
        inputs: Sequence[Variable], outputs: Sequence[Variable],
        priority: int = 0, simulation: bool = False, execution: bool = False,
    ):
        super().__init__(name, all_arguments, certifies, priority=priority, simulation=simulation, execution=execution)
        self.inputs = tuple(inputs)
        self.outputs = tuple(outputs)

        if len(self.outputs) == 1:
            self.ftype = FunctionType([arg.dtype for arg in self.inputs], self.outputs[0].dtype, argument_names=[arg.name for arg in self.inputs], return_name=self.outputs[0].name)
        else:
            self.ftype = FunctionType([arg.dtype for arg in self.inputs], TupleType([output.dtype for output in self.outputs]), argument_names=[arg.name for arg in self.inputs], return_name=[output.name for output in self.outputs])

    name: str
    all_arguments: Sequence[Variable]
    certifies: Sequence[ValueOutputExpression]
    priority: int
    evaluation_mode: CrowFunctionEvaluationMode

    inputs: Sequence[Variable]
    """The list of input variables that the generator takes."""

    outputs: Sequence[Variable]
    """The list of output variables that the generator generates."""

    ftype: FunctionType
    """The function type of the generator."""


class CrowUndirectedGenerator(CrowGeneratorBase):
    name: str
    all_arguments: Sequence[Variable]
    certifies: Sequence[ValueOutputExpression]
    priority: int
    evaluation_mode: CrowFunctionEvaluationMode


class CrowGeneratorApplicationExpression(object):
    def __init__(self, generator: CrowGeneratorBase, arguments: Sequence[ObjectOrValueOutputExpression], outputs: Optional[Sequence[Variable]] = None):
        self.generator = generator
        self.arguments = tuple(arguments)
        self.outputs = tuple(outputs) if outputs is not None else None

        self._validate_arguments()

    def _validate_arguments(self):
        if isinstance(self.generator, CrowDirectedGenerator):
            assert len(self.outputs) == len(self.generator.outputs), f'Expected {len(self.generator.outputs)} outputs, got {len(self.outputs)}.'
            for i, (output, expected) in enumerate(zip(self.outputs, self.generator.outputs)):
                assert output.dtype == expected.dtype, f'Output {i} has type {output.dtype}, expected {expected.dtype}.'

    generator: CrowGeneratorBase
    """The generator to be applied."""

    arguments: Tuple[ObjectOrValueOutputExpression, ...]
    """The arguments to be passed to the generator."""

    outputs: Optional[Tuple[Variable, ...]]
    """The output variables of the generator application."""

    def __str__(self):
        return f'{self.generator.name}({", ".join(map(str, self.arguments))}) -> {", ".join(map(str, self.outputs))}'

    def __repr__(self):
        return str(self)

