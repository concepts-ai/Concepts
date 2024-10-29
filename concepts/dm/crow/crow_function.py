#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : crow_function.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/16/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Optional, Any, Sequence, Tuple, Dict

from jacinle.utils.enum import JacEnum
from jacinle.utils.meta import repr_from_str
from jacinle.utils.printing import indent_text

from concepts.dsl.dsl_types import ObjectType
from concepts.dsl.dsl_functions import FunctionType, Function
from concepts.dsl.expression import ValueOutputExpression

__all__ = ['CrowFunctionEvaluationMode', 'CrowFunctionBase', 'CrowFeature', 'CrowFunction']


class CrowFunctionEvaluationMode(JacEnum):
    """The evaluation mode of a function. This enum has three values:

    - ``FUNCTIONAL``: the function is a pure function.
    - ``SIMULATION``: the function is a simulation-dependent function, i.e., it is a function that can only be evaluated given the current state in simulation.
    - ``EXECUTION``: the function is an execution-dependent function, i.e., it is a function that can only be evaluated given the current state in execution.
    """

    FUNCTIONAL = 'functional'
    SIMULATION = 'simulation'
    EXECUTION = 'execution'

    @classmethod
    def from_bools(cls, simulation: bool, execution: bool):
        if simulation:
            assert not execution, 'Cannot set both simulation and execution mode.'
            return cls.SIMULATION
        elif execution:
            return cls.EXECUTION
        else:
            return cls.FUNCTIONAL

    def get_prefix(self) -> str:
        if self == self.FUNCTIONAL:
            return ''
        return f'[[{self.get_short_name()}]]'

    def get_short_name(self) -> str:
        return self.name[:3]


class CrowFunctionBase(Function):
    """The base class for Crow functions. This is the base class for both :class:`CrowFeature` and :class:`CrowFunction`."""
    def __init__(
        self, name: str, ftype: FunctionType,
        derived_expression: Optional[ValueOutputExpression] = None,
    ):
        """Initialize the Crow function.

        Args:
            name: the name of the function.
            ftype: the type of the function.
            derived_expression: an optional derived expression of the function.
        """
        super().__init__(name, ftype, derived_expression)

        self.is_static = False
        self.is_cacheable = self._guess_is_cacheable()

    is_static: bool
    """Whether the function is static (i.e., its grounded value will never change)."""

    is_cacheable: bool
    """Whether the function can be cached. Specifically, if it contains only "ObjectTypes" as arguments, it can be statically evaluated."""

    def _guess_is_cacheable(self) -> bool:
        """Return whether the function can be cached. Specifically, if it contains only "ObjectTypes" as arguments, it can be statically evaluated."""
        for arg_def in self.arguments:
            if not isinstance(arg_def.dtype, ObjectType):
                return False
        return True

    def mark_static(self, flag: bool = True):
        """Mark a predicate as static (i.e., its grounded value will never change).

        Args:
            flag: Whether to mark the predicate as static.
        """
        self.is_static = flag

    @property
    def is_feature(self) -> bool:
        """Whether the object is defined as a state feature."""
        return False

    @property
    def is_function(self) -> bool:
        """Whether the object is defined as a function."""
        return False

    def flags(self, short: bool = False) -> Dict[str, bool]:
        """Return the flags of the function."""
        if short:
            return {
                'static': self.is_static,
                'cacheable': self.is_cacheable,
            }
        return {
            'is_derived': self.is_derived,
            'is_static': self.is_static,
            'is_cacheable': self.is_cacheable,
        }

    def __str__(self) -> str:
        flags = self.flags(short=True)
        flags = ', '.join([f for f, v in flags.items() if v])
        if len(flags) > 0:
            flags = f'[{flags}]'
        fmt = f'{self.name}{flags}({", ".join([str(arg) for arg in self.arguments])})'
        if self.is_derived:
            fmt += ':\n' + indent_text('return ' + str(self.derived_expression))
        return fmt

    __repr__ = repr_from_str



class CrowFeature(CrowFunctionBase):
    def __init__(
        self, name: str, ftype: FunctionType,
        derived_expression: Optional[ValueOutputExpression] = None,
        observation: Optional[bool] = None,
        state: Optional[bool] = None,
        default: Optional[Any] = None,
    ):
        super().__init__(name, ftype, derived_expression)

        self.is_observation_variable = observation if observation is not None else self._guess_is_observation()
        self.is_state_variable = state if state is not None else self._guess_is_state()
        self.default = default
        self._check_argument_types()

    is_static: bool
    is_cacheable: bool

    is_observation_variable: bool
    """Whether the feature is an observation variable."""

    is_state_variable: bool
    """Whether the feature is a state variable."""

    def _check_argument_types(self):
        for arg_def in self.arguments:
            assert isinstance(arg_def.dtype, ObjectType), f'Invalid argument type {arg_def.dtype} for feature {self.name}.'

    def _guess_is_observation(self) -> bool:
        """Guess whether the feature is an observation variable."""
        return not self.is_derived

    def _guess_is_state(self) -> bool:
        """Guess whether the feature is a state variable."""
        return True

    @property
    def is_feature(self) -> bool:
        return True

    def flags(self, short: bool = False) -> Dict[str, bool]:
        flags = super().flags(short)
        if short:
            del flags['cacheable']
            flags.update({
                'observation': self.is_observation_variable,
                'state': self.is_state_variable,
            })
        else:
            del flags['is_cacheable']
            flags.update({
                'is_observation_variable': self.is_observation_variable,
                'is_state_variable': self.is_state_variable,
            })
        return flags


class CrowFunction(CrowFunctionBase):
    def __init__(
        self, name: str, ftype: FunctionType,
        derived_expression: Optional[ValueOutputExpression] = None,
        generator_placeholder: bool = False,
        inplace_generators: Optional[Sequence[str]] = None,
        sgc: bool = False,
        simulation: bool = False, execution: bool = False,
    ):
        super().__init__(name, ftype, derived_expression)

        self.is_generator_placeholder = generator_placeholder
        self.inplace_generators = tuple(inplace_generators) if inplace_generators is not None else tuple()
        self.is_sgc_function = sgc
        self.evaluation_mode = CrowFunctionEvaluationMode.from_bools(simulation, execution)

    is_static: bool
    is_cacheable: bool

    evaluation_mode: CrowFunctionEvaluationMode
    """The evaluation mode of the function. This enum has three values:"""

    is_generator_placeholder: bool
    """Whether the function is a generator placeholder."""

    inplace_generators: Tuple[str, ...]
    """The list of inplace generators. This is usually used together with generator-placeholder functions."""

    is_sgc_function: bool
    """Whether the function is a SGC (state-goal-constraint) function."""

    @property
    def is_function(self) -> bool:
        return True

    @property
    def is_simulation_dependent(self) -> bool:
        """Whether the function is simulation-dependent."""
        return self.evaluation_mode == CrowFunctionEvaluationMode.SIMULATION

    @property
    def is_execution_dependent(self) -> bool:
        """Whether the function is execution-dependent."""
        return self.evaluation_mode == CrowFunctionEvaluationMode.EXECUTION

    def flags(self, short: bool = False) -> Dict[str, bool]:
        flags = super().flags(short)
        if short:
            flags.update({
                'gen': self.is_generator_placeholder,
                'sgc': self.is_sgc_function,
                'sim': self.is_simulation_dependent,
                'exe': self.is_execution_dependent,
            })
        else:
            flags.update({
                'is_generator_placeholder': self.is_generator_placeholder,
                'is_sgc_function': self.is_sgc_function,
                'is_simulation_dependent': self.is_simulation_dependent,
                'is_execution_dependent': self.is_execution_dependent,
            })
        return flags

