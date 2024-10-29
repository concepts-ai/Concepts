#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : controller.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/15/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Optional, Union, Sequence, Tuple, TYPE_CHECKING

from jacinle.utils.meta import repr_from_str
from jacinle.utils.printing import indent_text
from concepts.dsl.dsl_types import ObjectType, ValueType, Variable
from concepts.dsl.constraint import OptimisticValue
from concepts.dsl.tensor_value import TensorValue
from concepts.dsl.tensor_state import StateObjectReference
from concepts.dsl.expression import ObjectOrValueOutputExpression

if TYPE_CHECKING:
    from concepts.dm.crow.behavior import CrowBehaviorOrderingSuite

__all__ = ['CrowController', 'CrowControllerApplier', 'CrowControllerApplicationExpression']


class CrowController(object):
    """A controller is a class that defines a primitive action in the environment."""

    def __init__(self, name: str, arguments: Sequence[Variable], effect_body: Optional['CrowBehaviorOrderingSuite'] = None):
        self.name = name
        self.arguments = tuple(arguments)
        self.effect_body = effect_body

    name: str
    """The name of the controller."""

    arguments: Tuple[Variable, ...]
    """The arguments of the controller."""

    effect_body: Optional['CrowBehaviorOrderingSuite']
    """The effect body of the controller."""

    @property
    def argument_names(self) -> Tuple[str, ...]:
        return tuple(arg.name for arg in self.arguments)

    @property
    def argument_types(self) -> Tuple[Union[ObjectType, ValueType], ...]:
        return tuple(arg.dtype for arg in self.arguments)

    def short_str(self):
        return f'{self.name}({", ".join(str(arg) for arg in self.arguments)})'

    def long_str(self):
        fmt = f'controller {self.name}({", ".join(str(arg) for arg in self.arguments)})'
        if self.effect_body is not None:
            effect_string = '\n'.join(map(str, self.effect_body.statements))
            fmt += ':\n'
            fmt += f'  effects:\n{indent_text(effect_string, 2)}'
        return fmt

    def __str__(self):
        return self.short_str()

    __repr__ = repr_from_str


class CrowControllerApplier(object):
    def __init__(self, controller: CrowController, arguments: Sequence[Union[str, StateObjectReference, TensorValue, OptimisticValue]]):
        self.controller = controller
        self.arguments = tuple(arguments)

    controller: CrowController
    """The controller to be applied."""

    arguments: Tuple[Union[str, TensorValue], ...]
    """The arguments of the controller application."""

    @property
    def name(self) -> str:
        return self.controller.name

    def __str__(self):
        return f'{self.controller.name}({", ".join(_argument_string(arg) for arg in self.arguments)})'

    __repr__ = repr_from_str


class CrowControllerApplicationExpression(object):
    def __init__(self, controller: CrowController, arguments: Sequence[ObjectOrValueOutputExpression]):
        self.controller = controller
        self.arguments = tuple(arguments)

    controller: CrowController
    """The controller to be applied."""

    arguments: Tuple[ObjectOrValueOutputExpression, ...]
    """The arguments of the controller application."""

    def __str__(self):
        return f'{self.controller.name}({", ".join(str(arg) for arg in self.arguments)})'

    __repr__ = repr_from_str


def _argument_string(arg: Union[str, StateObjectReference, TensorValue, OptimisticValue]) -> str:
    if isinstance(arg, bool):
        return str(arg)
    if isinstance(arg, str):
        return arg
    if isinstance(arg, StateObjectReference):
        return arg.name
    if isinstance(arg, TensorValue):
        if arg.is_single_elem:
            return str(arg.single_elem())
        return str(arg)
    if isinstance(arg, OptimisticValue):
        return str(arg)
    raise TypeError(f'Unsupported argument type: {type(arg)}')
