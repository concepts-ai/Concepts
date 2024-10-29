#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : generator.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/04/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Optional, Union, Sequence, Tuple

from jacinle.utils.meta import repr_from_str

from concepts.dsl.dsl_types import ObjectType, ValueType, Variable, TensorValueTypeBase, PyObjValueType
from concepts.dsl.dsl_functions import Function
from concepts.dsl.expression import VariableExpression, ValueOutputExpression
from concepts.dm.pdsketch.operator import Implementation

__all__ = ['Generator', 'FancyGenerator', 'GeneratorApplicationExpression']


class Generator(object):
    """A generator is function that generates a set of values from a set of given values. Semantically, it certifies that
    the generated values satisfy a given condition."""

    def __init__(
        self,
        name: str,
        arguments: Sequence[Variable],
        certifies: ValueOutputExpression,
        context: Sequence[Union[VariableExpression, ValueOutputExpression]],
        generates: Sequence[Union[VariableExpression, ValueOutputExpression]],
        function: Function,
        output_vars: Sequence[Variable],
        flatten_certifies: ValueOutputExpression,
        implementation: Optional[Implementation] = None,
        priority: int = 0,
        unsolvable: bool = False
    ):
        self.name = name
        self.arguments = tuple(arguments)
        self.certifies = certifies
        self.context = tuple(context)
        self.generates = tuple(generates)
        self.function = function
        self.output_vars = tuple(output_vars)
        self.output_types = tuple(v.dtype for v in output_vars)
        self.flatten_certifies = flatten_certifies
        self.implementation = implementation
        self.priority = priority
        self.unsolvable = unsolvable

    name: str
    """The name of the generator."""

    arguments: Tuple[Variable, ...]
    """The arguments of the generator."""

    certifies: ValueOutputExpression
    """The condition that the generated values should satisfy."""

    context: Tuple[Union[VariableExpression, ValueOutputExpression], ...]
    """The context values that the generator depends on."""

    generates: Tuple[Union[VariableExpression, ValueOutputExpression], ...]
    """The values that the generator generates."""

    function: Function
    """The declaration of the underlying function that generates the values."""

    output_vars: Tuple[Variable, ...]
    """The output variables of the function."""

    output_types: Tuple[Union[TensorValueTypeBase, PyObjValueType], ...]
    """The output type of the function."""

    flatten_certifies: ValueOutputExpression
    """The condition that the generated values should satisfy, flattened."""

    implementation: Optional[Implementation]
    """The implementation of the generator."""

    priority: int
    """The priority of the generator."""

    unsolvable: bool
    """Whether the generator is unsolvable."""

    @property
    def input_vars(self) -> Tuple[Variable, ...]:
        """The input variables of the function."""
        return self.function.arguments

    def short_str(self):
        return f'{self.name}({", ".join([str(c) for c in self.context])}) -> {", ".join([str(c) for c in self.generates])}'

    @property
    def argument_names(self) -> Tuple[str, ...]:
        """The names of the arguments of the operator."""
        return tuple(arg.name for arg in self.arguments)

    @property
    def argument_types(self) -> Tuple[Union[ObjectType, ValueType], ...]:
        """The types of the arguments of the operator."""
        return tuple(arg.dtype for arg in self.arguments)

    def __str__(self):
        arg_string = ', '.join([str(c) for c in self.context])
        gen_string = ', '.join([str(c) for c in self.generates])
        return (
            f'{self.name}({arg_string}) -> {gen_string}' + ' {\n'
            '  function:   ' + str(self.function) + '\n'
            '  parameters: ' + str(self.arguments) + '\n'
            '  certifies:  ' + str(self.flatten_certifies) + '\n'
            '}'
        )

    __repr__ = repr_from_str


class Generator3(object):
    """A generator is function that generates a set of values from a set of given values. Semantically, it certifies that
    the generated values satisfy a given condition."""

    def __init__(
        self,
        name: str,
        arguments: Sequence[Variable],
        certifies: ValueOutputExpression,
        context: Sequence[Union[VariableExpression, ValueOutputExpression]],
        generates: Sequence[Union[VariableExpression, ValueOutputExpression]],
        function: Function,
        output_vars: Sequence[Variable],
        flatten_certifies: ValueOutputExpression,
        implementation: Optional[Implementation] = None,
        priority: int = 0,
        unsolvable: bool = False
    ):
        self.name = name
        self.arguments = tuple(arguments)
        self.certifies = certifies
        self.context = tuple(context)
        self.generates = tuple(generates)
        self.function = function
        self.output_vars = tuple(output_vars)
        self.output_types = tuple(v.dtype for v in output_vars)
        self.flatten_certifies = flatten_certifies
        self.implementation = implementation
        self.priority = priority
        self.unsolvable = unsolvable

    name: str
    """The name of the generator."""

    arguments: Tuple[Variable, ...]
    """The arguments of the generator."""

    certifies: ValueOutputExpression
    """The condition that the generated values should satisfy."""

    context: Tuple[Union[VariableExpression, ValueOutputExpression], ...]
    """The context values that the generator depends on."""

    generates: Tuple[Union[VariableExpression, ValueOutputExpression], ...]
    """The values that the generator generates."""

    function: Function
    """The declaration of the underlying function that generates the values."""

    output_vars: Tuple[Variable, ...]
    """The output variables of the function."""

    output_types: Tuple[Union[TensorValueTypeBase, PyObjValueType], ...]
    """The output type of the function."""

    flatten_certifies: ValueOutputExpression
    """The condition that the generated values should satisfy, flattened."""

    implementation: Optional[Implementation]
    """The implementation of the generator."""

    priority: int
    """The priority of the generator."""

    unsolvable: bool
    """Whether the generator is unsolvable."""

    @property
    def input_vars(self) -> Tuple[Variable, ...]:
        """The input variables of the function."""
        return self.function.arguments

    def short_str(self):
        return f'{self.name}({", ".join([str(c) for c in self.context])}) -> {", ".join([str(c) for c in self.generates])}'

    @property
    def argument_names(self) -> Tuple[str, ...]:
        """The names of the arguments of the operator."""
        return tuple(arg.name for arg in self.arguments)

    @property
    def argument_types(self) -> Tuple[Union[ObjectType, ValueType], ...]:
        """The types of the arguments of the operator."""
        return tuple(arg.dtype for arg in self.arguments)

    def __str__(self):
        arg_string = ', '.join([str(c) for c in self.context])
        gen_string = ', '.join([str(c) for c in self.generates])
        return (
            f'{self.name}({arg_string}) -> {gen_string}' + ' {\n'
            '  function:   ' + str(self.function) + '\n'
            '  parameters: ' + str(self.arguments) + '\n'
            '  certifies:  ' + str(self.flatten_certifies) + '\n'
            '}'
        )

    __repr__ = repr_from_str


class FancyGenerator(object):
    def __init__(
        self,
        name: str,
        certifies: ValueOutputExpression,
        function: Function,
        flatten_certifies: ValueOutputExpression,
        implementation: Optional[Implementation] = None,
        priority: int = 0,
        unsolvable: bool = False
    ):
        self.name = name
        self.certifies = certifies
        self.function = function
        self.flatten_certifies = flatten_certifies
        self.implementation = implementation
        self.priority = priority
        self.unsolvable = unsolvable

    name: str
    """The name of the generator."""

    certifies: ValueOutputExpression
    """The condition that the generated values should satisfy."""

    function: Function
    """The declaration of the underlying function that generates the values."""

    flatten_certifies: ValueOutputExpression
    """The condition that the generated values should satisfy, flattened."""

    implementation: Optional[Implementation]
    """The implementation of the generator."""

    priority: int
    """The priority of the generator."""

    unsolvable: bool
    """Whether the generator is unsolvable."""

    def short_str(self):
        return f'{self.name}() -> {str(self.flatten_certifies)}'

    def __str__(self):
        return (
            f'{self.name}() -> ' + ' {\n'
            '  ' + str(self.function) + '\n'
            '  certifies: ' + str(self.flatten_certifies) + '\n'
            '}'
        )

    __repr__ = repr_from_str


class GeneratorApplicationExpression(object):
    """An abstract operator grounding. For example :code:`(move ?x ?y)` where :code:`?x` and :code:`?y` are variables in the context."""

    def __init__(self, generator: Generator, arguments: Sequence[Union[VariableExpression, ValueOutputExpression]]):
        self.generator = generator
        self.arguments = tuple(arguments)

    generator: Generator
    """The operator that is applied."""

    arguments: Tuple[Union[VariableExpression, ValueOutputExpression], ...]
    """The arguments of the operator."""

    @property
    def name(self) -> str:
        """The name of the operator."""
        return self.generator.name

    def __str__(self) -> str:
        def_name = 'generator'
        arg_string = ', '.join([
            arg_def.name + '=' + str(arg)
            for arg_def, arg in zip(self.generator.arguments, self.arguments)
        ])
        return f'{def_name}::{self.generator.name}({arg_string})'

    __repr__ = repr_from_str

    def pddl_str(self) -> str:
        arg_str = ' '.join([str(arg) for arg in self.arguments])
        return f'({self.generator.name} {arg_str})'
