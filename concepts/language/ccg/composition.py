#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : composition.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/05/2020
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Composition rules for CCG.

The two main classes are :class:`CCGComposable` and :class:`CCGCompositionSystem`.
"""

import contextlib
from typing import TYPE_CHECKING, Optional, Union, List, Dict, Callable
from dataclasses import dataclass
from jacinle.utils.cache import cached_property
from jacinle.utils.defaults import option_context
from jacinle.utils.enum import JacEnum
from jacinle.utils.printing import indent_text

if TYPE_CHECKING:
    from concepts.language.ccg.grammar import CCGNode

__all__ = [
    'CCGCompositionDirection', 'CCGCompositionType', 'CCGCompositionError',
    'CCGCompositionContext', 'get_ccg_composition_context',
    'CCGCompositionResult', 'CCGCoordinationImmNode', 'CCGComposable',
    'CCGCompositionSystem'
]


class CCGCompositionDirection(JacEnum):
    """Composition directions (left or right)."""
    LEFT = 'left'
    RIGHT = 'right'


class CCGCompositionType(JacEnum):
    """Composition types (e.g., application and coordination)."""

    LEXICON = 'lexicon'
    FORWARD_APPLICATION = 'forward_application'
    BACKWARD_APPLICATION = 'backward_application'
    COORDINATION = 'coordination'
    NONE = 'none'


class CCGCompositionError(Exception):
    """The error raised when composition fails."""
    pass


class CCGCompositionContext(option_context(
    '_CCGCompositionContext',
    syntax=True,
    semantics=True,
    semantics_lazy_composition=False,
    exc_verbose=True
)):
    """An option context for CCG composition."""

    syntax: bool
    """Whether to perform syntax composition."""

    semantics: bool
    """Whether to perform semantics composition."""

    semantics_lazy_composition: bool
    """Whether to perform lazy semantics composition."""

    exc_verbose: bool
    """Whether to raise verbose exceptions."""

    @contextlib.contextmanager
    def exc(self, exc_type: Optional[type] = None, from_: Optional[Exception] = None):
        """Context manager for handling composition errors. If `exc_verbose` is True, the error will be printed out.

        Example:
            >>> with get_ccg_composition_context().exc():
            >>>     raise CCGCompositionError('some error')

        Args:
            exc_type: the exception type to raise. If None, the original exception will be raised.
            from_: the original exception to raise.
        """
        if self.exc_verbose:
            yield
        else:
            if exc_type is None:
                exc_type = CCGCompositionError
            if from_ is not None:
                raise exc_type() from from_
            raise exc_type()


get_ccg_composition_context: Callable[[], CCGCompositionContext] = CCGCompositionContext.get_default


@dataclass
class CCGCoordinationImmNode(object):
    """An intermediate node for coordination."""

    conj: 'CCGComposable'
    """The conjunction node."""

    rhs: 'CCGComposable'
    """The right-hand side node."""

    # Adding a few properties to make it compatible with CCGComposable.

    @property
    def is_none(self) -> bool:
        """Whether this node is a None node. This property is always False."""
        return False

    @property
    def is_conj(self) -> bool:
        """Whether this node is a conjunction node. This property is always False."""
        return False

    @property
    def is_value(self) -> bool:
        """Whether this node is a value node. This property is always False."""
        return False

    @property
    def is_function(self) -> bool:
        """Whether this node is a function node. This property is always False."""
        return False


class CCGComposable(object):
    """The basic class for composable elements (including syntax and semantics) for CCG."""

    @property
    def is_none(self) -> bool:
        """Whether this element is None."""
        return False

    @property
    def is_conj(self) -> bool:
        """Whether this element is a conjunction."""
        return False

    def compose(self, rhs: Union['CCGComposable', CCGCoordinationImmNode], composition_type: CCGCompositionType) -> Union['CCGComposable', CCGCoordinationImmNode]:
        """Compose this element with another element. This function will call the corresponding composition function according to the composition type.
        Note that since the coordination composition has three arguments, this function will return a :class:`CCGCoordinationImmNode` for the first two arguments
        in coordination composition.

        Args:
            rhs: the right-hand side element.
            composition_type: the composition type.

        Returns:
            The composed element.
        """

        if isinstance(rhs, CCGCoordinationImmNode) and composition_type is not CCGCompositionType.COORDINATION:
            raise CCGCompositionError('Can not make non-coordination composition for CCGCoordinationImmNode.')
        if (self.is_none or (not isinstance(rhs, CCGCoordinationImmNode) and rhs.is_none)) and composition_type is not CCGCompositionType.NONE:
            raise CCGCompositionError('Can not make non-None composition with none elements.')

        if composition_type is CCGCompositionType.LEXICON:
            raise CCGCompositionError('Lexicon composition type is only used for leaf level nodes.')
        elif composition_type is CCGCompositionType.FORWARD_APPLICATION:
            return self.fapp(rhs)
        elif composition_type is CCGCompositionType.BACKWARD_APPLICATION:
            return rhs.bapp(self)
        elif composition_type is CCGCompositionType.COORDINATION:
            return self.coord(rhs)
        elif composition_type is CCGCompositionType.NONE:
            return self.none(rhs)

    def fapp(self, rhs: 'CCGComposable') -> 'CCGComposable':
        """Forward application composition."""
        assert not self.is_none and not rhs.is_none
        return self._fapp(rhs)

    def bapp(self, lhs: 'CCGComposable') -> 'CCGComposable':
        """Backward application composition."""
        assert not self.is_none and not lhs.is_none
        return self._bapp(lhs)

    def none(self, rhs: 'CCGComposable') -> 'CCGComposable':
        """None composition (composition with a None element)."""
        if rhs.is_none:
            return self
        elif self.is_none:
            return rhs
        with get_ccg_composition_context().exc(CCGCompositionError):
            raise CCGCompositionError(f'Invalid None composition: lhs={self} and rhs={rhs}.')

    def coord(self, other: Union['CCGComposable', CCGCoordinationImmNode]) -> Union['CCGComposable', CCGCoordinationImmNode]:
        """Coordination composition."""
        if isinstance(other, CCGCoordinationImmNode):
            return other.conj.coord3(self, other.rhs)
        elif self.is_conj:
            return CCGCoordinationImmNode(self, other)
        with get_ccg_composition_context().exc(CCGCompositionError):
            raise CCGCompositionError(f'Invalid coordination composition: lhs={self}, rhs={other}.')

    def coord3(self, lhs: 'CCGComposable', rhs: 'CCGComposable') -> 'CCGComposable':
        """Coordination composition with three elements."""
        assert not self.is_none and not lhs.is_none and not rhs.is_none
        return self._coord3(lhs, rhs)

    def _fapp(self, rhs: 'CCGComposable') -> 'CCGComposable':
        raise NotImplementedError()

    def _bapp(self, lhs: 'CCGComposable') -> 'CCGComposable':
        raise NotImplementedError()

    def _coord3(self, lhs: 'CCGComposable', rhs: 'CCGComposable') -> 'CCGComposable':
        raise NotImplementedError()


@dataclass
class CCGCompositionResult(object):
    """The result of a CCG composition."""

    composition_type: CCGCompositionType
    """The composition type applied at the current node."""

    result: Union[CCGComposable, CCGCoordinationImmNode, 'CCGNode']
    """The result of the composition."""


class CCGCompositionSystem(object):
    """The CCG composition system. It keeps track of the rules that can be used for composition."""

    def __init__(self, name: str, weights: Dict[CCGCompositionType, float]):
        """Initialize the CCG composition system.

        Args:
            name: the name of the composition system.
            weights: the weights of the composition types, which should be a dictionary mapping from :class:`CCGCompositionType` to float.
        """
        self.name = name
        self.weights = weights

    @cached_property
    def allowed_composition_types(self) -> List[CCGCompositionType]:
        """Get the list of allowed composition types.

        Returns:
            the list of allowed composition types.
        """
        return [c for c in CCGCompositionType.choice_objs() if c in self.weights and c is not CCGCompositionType.LEXICON]

    def try_compose(self, lhs: CCGComposable, rhs: CCGComposable) -> CCGCompositionResult:
        """Try to compose two elements.
        This function will try to compose the two elements with all allowed composition types, and return the result that works.

        Args:
            lhs: the left-hand side element.
            rhs: the right-hand side element.

        Returns:
            The composition result.
        """

        results = list()
        exceptions = list()
        for composition_type in self.allowed_composition_types:
            try:
                ret = lhs.compose(rhs, composition_type)
                results.append((composition_type, ret))
            except CCGCompositionError as e:
                exceptions.append(e)

        if len(results) == 1:
            return CCGCompositionResult(*results[0])
        elif len(results) == 0:
            with get_ccg_composition_context().exc():
                fmt = f'Failed to compose CCGNodes {lhs} and {rhs}.\n'
                fmt += 'Detailed messages are:\n'
                for t, e in zip(self.allowed_composition_types, exceptions):
                    fmt += indent_text('Trying CCGCompositionType.{}:\n{}'.format(t.name, str(e))) + '\n'
                raise CCGCompositionError(fmt.rstrip())
        else:
            with get_ccg_composition_context().exc():
                fmt = f'Got ambiguous composition for CCGNodes {lhs} and {rhs}.\n'
                fmt += 'Candidates are:\n'
                for r in results:
                    fmt += indent_text('CCGCompositionType.' + str(r[0].name)) + '\n'
                raise CCGCompositionError(fmt.rstrip())

    def __str__(self) -> str:
        return f'CCGCompositionSystem({self.name})'

    __repr__ = __str__

    def format_summary(self) -> str:
        """Format the summary of the composition system."""
        fmt = 'Allowed composition types:\n'
        for type, weight in self.weights.items():
            fmt += '  CCGCompositionType.' + type.name + ': ' + str(weight) + '\n'
        fmt = 'CCGCompositionSystem: {}\n'.format(self.name) + indent_text(fmt.rstrip())
        return fmt

    def print_summary(self):
        """Print the summary of the composition system."""
        print(self.format_summary())

    @classmethod
    def make_default(cls) -> 'CCGCompositionSystem':
        """Make the default CCG composition system."""
        return cls.make_function_application()

    @classmethod
    def make_function_application(cls) -> 'CCGCompositionSystem':
        """Make the CCG composition system that only allows function application."""
        return cls('function_application', {
            CCGCompositionType.LEXICON: 0,
            CCGCompositionType.FORWARD_APPLICATION: 0,
            CCGCompositionType.BACKWARD_APPLICATION: 0,
            CCGCompositionType.NONE: 0
        })

    @classmethod
    def make_categorial_grammar(cls) -> 'CCGCompositionSystem':
        """Make the CCG composition system that allows function application and coordination (i.e., categorial grammar)."""
        return cls('categorial_grammar', {
            CCGCompositionType.LEXICON: 0,
            CCGCompositionType.FORWARD_APPLICATION: 0,
            CCGCompositionType.BACKWARD_APPLICATION: 0,
            CCGCompositionType.COORDINATION: 0,
            CCGCompositionType.NONE: 0,
        })

