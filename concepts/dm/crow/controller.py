#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : controller.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/15/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Optional, Union, Sequence, Tuple, Dict, TYPE_CHECKING

from jacinle.utils.meta import repr_from_str
from jacinle.utils.printing import indent_text
from concepts.dsl.dsl_types import ObjectType, ValueType, Variable
from concepts.dsl.constraint import OptimisticValue
from concepts.dsl.tensor_value import TensorValue
from concepts.dsl.tensor_state import StateObjectReference, StateObjectList
from concepts.dsl.expression import ObjectOrValueOutputExpression, ValueOutputExpression

if TYPE_CHECKING:
    from concepts.dm.crow.behavior import CrowBehaviorOrderingSuite

__all__ = ['CrowController', 'CrowControllerApplier', 'CrowControllerApplicationExpression']


class CrowController(object):
    """A controller is a class that defines a primitive action in the environment."""

    def __init__(self, name: str, arguments: Sequence[Variable], effect_body: Optional['CrowBehaviorOrderingSuite'] = None, python_effect: bool = False):
        self.name = name
        self.arguments = tuple(arguments)
        self.effect_body = effect_body
        self.python_effect = python_effect

    name: str
    """The name of the controller."""

    arguments: Tuple[Variable, ...]
    """The arguments of the controller."""

    effect_body: Optional['CrowBehaviorOrderingSuite']
    """The effect body of the controller."""

    python_effect: bool
    """Whether the effect body is implemented in Python."""

    @property
    def argument_names(self) -> Tuple[str, ...]:
        return tuple(arg.name for arg in self.arguments)

    @property
    def argument_types(self) -> Tuple[Union[ObjectType, ValueType], ...]:
        return tuple(arg.dtype for arg in self.arguments)

    def short_str(self):
        return f'{self.name}({", ".join(str(arg) for arg in self.arguments)})'

    def long_str(self):
        flag_string = ''
        if self.python_effect:
            flag_string = '[[python_effect]]'
        fmt = f'controller {flag_string}{self.name}({", ".join(str(arg) for arg in self.arguments)})'
        if self.effect_body is not None:
            effect_string = '\n'.join(map(str, self.effect_body.statements))
            fmt += ':\n'
            fmt += f'  effects:\n{indent_text(effect_string, 2)}'
        return fmt

    def __str__(self):
        return self.short_str()

    __repr__ = repr_from_str


class CrowControllerApplier(object):
    def __init__(
        self, controller: CrowController, arguments: Sequence[Union[str, StateObjectReference, StateObjectList, TensorValue, OptimisticValue]],
        global_constraints: Optional[Dict[int, Sequence[ValueOutputExpression]]] = None, local_constraints: Optional[Sequence[ValueOutputExpression]] = None
    ):
        self.controller = controller
        self.arguments = tuple(arguments)
        self.global_constraints = global_constraints
        self.local_constraints = local_constraints

    controller: CrowController
    """The controller to be applied."""

    arguments: Tuple[Union[str, TensorValue], ...]
    """The arguments of the controller application."""

    global_constraints: Optional[Dict[int, Tuple[Tuple[ValueOutputExpression, ...], dict]]]
    """The global constraints of the controller application."""

    local_constraints: Optional[Tuple[Tuple[ValueOutputExpression, ...], dict]]
    """The local constraints of the controller application."""

    def set_constraints(self, global_constraints: Dict[int, Sequence[ValueOutputExpression]], global_scopes: Dict[int, dict], scope_id: int):
        # TODO(Jiayuan Mao @ 2025/01/14): set the scopes associated with these constraints...
        self.global_constraints = {k: (tuple(v), global_scopes[k]) for k, v in global_constraints.items()}
        self.local_constraints = self.global_constraints.get(scope_id, None)
        return self

    @property
    def name(self) -> str:
        return self.controller.name

    def long_str(self) -> str:
        from concepts.dm.crow.behavior_utils import format_behavior_statement
        fmt = f'{self.controller.name}({", ".join(_argument_string(arg) for arg in self.arguments)})'
        if self.global_constraints is not None:
            global_constraints_str = indent_text('\n'.join(f'{k}: {{{", ".join(str(format_behavior_statement(c, scope=scope)) for c in constraints)}}}' for k, (constraints, scope) in self.global_constraints.items()), 2)
            fmt += f'\n  with global constraints:\n{global_constraints_str}'
        if self.local_constraints is not None:
            local_constraints_str = '{' + ', '.join(str(format_behavior_statement(c, scope=self.local_constraints[1])) for c in self.local_constraints[0]) + '}'
            fmt += f'\n  with local constraints: {local_constraints_str}'
        return fmt

    def short_str(self) -> str:
        return f'{self.controller.name}({", ".join(_argument_string(arg) for arg in self.arguments)})'

    def __str__(self):
        return self.short_str()

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


def _argument_string(arg: Union[str, StateObjectReference, StateObjectList, TensorValue, OptimisticValue]) -> str:
    if isinstance(arg, bool):
        return str(arg)
    if isinstance(arg, str):
        return arg
    if isinstance(arg, StateObjectReference):
        return arg.name
    if isinstance(arg, StateObjectList):
        return f'[{", ".join([x.name for x in arg.values])}]'
    if isinstance(arg, TensorValue):
        if arg.is_single_elem:
            return str(arg.single_elem())
        return str(arg)
    if isinstance(arg, OptimisticValue):
        return str(arg)
    raise TypeError(f'Unsupported argument type: {type(arg)}')
