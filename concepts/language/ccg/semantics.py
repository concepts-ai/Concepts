#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : semantics.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/04/2020
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Data structures for semantic forms in a linguistic CCG.

The basic class is :class:`CCGSemantics`, which is a wrapper of a semantic form (a functor or a value).
There is a
"""

from typing import Optional, Union, Tuple, Dict, Callable
from dataclasses import dataclass
from jacinle.utils.cache import cached_property
from jacinle.utils.printing import indent_text

from concepts.dsl.dsl_functions import Function, FunctionArgumentResolutionContext, FunctionArgumentResolutionError
from concepts.dsl.function_domain import FunctionDomain
from concepts.dsl.value import Value
from concepts.dsl.expression import ConstantExpression, FunctionApplicationExpression
from concepts.language.ccg.composition import CCGCompositionType, CCGComposable, CCGCompositionError, get_ccg_composition_context

__all__ = [
    'CCGSemanticsCompositionError',
    'CCGSemanticsConjFunction', 'CCGSemanticsSimpleConjFunction',
    'CCGSemanticsLazyValue', 'CCGSemantics',
    'CCGSemanticsSugar',
]


class CCGSemanticsCompositionError(CCGCompositionError):
    """Raised when the semantics composition fails."""

    def __init__(self, message=None, lhs=None, rhs=None, error=None, conj=None):
        """Initialize the error.

        Args:
            message: the error message.
            lhs: the lhs semantics.
            rhs: the rhs semantics.
            error: the error raised by the left/right semantics.
            conj: the conjunction.
        """

        if message is None:
            super().__init__(None)
        else:
            message += '\nLeft:' + indent_text(str(lhs)) + '\nRight:' + indent_text(str(rhs))
            if conj is not None:
                message += '\nConj: ' + indent_text(str(conj))
            message += '\nOriginal error message: ' + indent_text(str(error)).lstrip()
            super().__init__('(Semantics) ' + message)


@dataclass
class CCGSemanticsConjFunction(object):
    """CCGSemanticsConjFunction is a wrapper that represents the semantics form of a conjunction term (e.g., AND)."""

    impl: Callable[[Union[Value, Function], Union[Value, Function]], Union[Value, Function]]
    """The underlying implementation of the conjunction."""

    def __call__(self, lhs: Union[Value, Function], rhs: Union[Value, Function]) -> Union[Value, Function]:
        """Perform the conjunction.

        Args:
            lhs: the left-hand side semantics.
            rhs: the right-hand side semantics.

        Returns:
            the conjunction result.
        """
        return self.impl(lhs, rhs)


class CCGSemanticsSimpleConjFunction(CCGSemanticsConjFunction):
    """A simple implementation for :class:`CCGSemanticsConjValue`.

    This function takes a function that works for :class:`~concepts.dsl.value.Value` and automatically
    converts it to a function that works for :class:`~concepts.dsl.function.Function`.
    """
    impl: Callable[[Union[Value, Function], Union[Value, Function]], Union[Value, Function]]

    def __call__(self, lhs: Union[Value, Function], rhs: Union[Value, Function]) -> Union[Value, Function]:
        """Perform the conjunction.

        Args:
            lhs: the left-hand side semantics.
            rhs: the right-hand side semantics.

        Returns:
            the conjunction result.
        """
        if isinstance(lhs, Function) and isinstance(rhs, Function):
            def body(*args, **kwargs):
                return self.impl(lhs(*args, **kwargs), rhs(*args, **kwargs))
            return Function('__conj__', lhs.ftype, overridden_call=body)

        if isinstance(lhs, Value) and isinstance(rhs, Value):
            return self.impl(lhs, rhs)

        raise CCGSemanticsCompositionError(f'Cannot perform conjunction between {lhs} and {rhs}.')


@dataclass
class CCGSemanticsLazyValue(object):
    """A wrapper that represents the semantic form of a node that is not yet evaluated.
    Specifically, this function stores the composition type and the individual components."""

    composition_type: Optional[CCGCompositionType] = None
    """The composition type."""

    lhs: Optional[Union['CCGSemantics', 'CCGSemanticsLazyValue']] = None
    """The left-hand side semantics."""

    rhs: Optional[Union['CCGSemantics', 'CCGSemanticsLazyValue']] = None
    """The right-hand side semantics."""

    conj: Optional[Callable] = None
    """The conjunction function (optional)."""

    def execute(self) -> Union[Value, Function]:
        """Execute the lazy value recursively and return the semantic form."""
        lhs, rhs = self.lhs, self.rhs
        if isinstance(lhs, CCGSemanticsLazyValue):
            lhs = lhs.execute()
        if isinstance(rhs, CCGSemanticsLazyValue):
            rhs = rhs.execute()

        if self.composition_type in (CCGCompositionType.FORWARD_APPLICATION, CCGCompositionType.BACKWARD_APPLICATION):
            if self.composition_type is CCGCompositionType.FORWARD_APPLICATION:
                return _forward_application(lhs, rhs)
            else:
                return _backward_application(lhs, rhs)
        elif self.composition_type is CCGCompositionType.COORDINATION:
            return _coordination(lhs, self.conj, rhs)
        else:
            raise NotImplementedError(f'Unknown composition type: {self.composition_type}.')


class CCGSemantics(CCGComposable):
    """CCGSemantics is a wrapper of a semantic form (a functor or a value)."""

    def __init__(self, value: Union[None, Callable, Function, ConstantExpression, FunctionApplicationExpression, CCGSemanticsLazyValue], *, is_conj: bool = False):
        self._value = value
        self._is_conj = is_conj

        if isinstance(self._value, Function) and self._value.nr_arguments == 0:
            self._value = self._value()

    @property
    def value(self) -> Union[Callable, Function, ConstantExpression, FunctionApplicationExpression, CCGSemanticsLazyValue]:
        """The semantic form."""
        return self._value

    @property
    def is_none(self):
        """Whether the semantics is None."""
        return self.value is None

    @property
    def is_conj(self):
        """Whether the semantics is a conjunction."""
        return self._is_conj

    @property
    def is_py_function(self):
        """Whether the semantics is a Python function."""
        return callable(self.value) and not self.is_function

    @property
    def is_lazy(self):
        """Whether the semantics is a lazy value."""
        return isinstance(self.value, CCGSemanticsLazyValue)

    @property
    def is_function(self):
        """Whether the semantics is a function."""
        if self.is_lazy:
            raise ValueError('Cannot check is_function for CCGSemanticsLazyValue')
        return isinstance(self.value, Function)

    @property
    def is_constant(self):
        """Whether the semantics is a constant."""
        if self.is_lazy:
            raise ValueError('Cannot check is_value for CCGSemanticsLazyValue')
        return isinstance(self.value, ConstantExpression)

    @property
    def is_function_application(self):
        """Whether the semantics is a function application expression."""
        if self.is_lazy:
            raise ValueError('Cannot check is_function_application for CCGSemanticsLazyValue')
        return isinstance(self.value, FunctionApplicationExpression)

    @property
    def is_value(self):
        """Whether the semantics is a value (either a constant or a function application expression)."""
        return self.is_constant or self.is_function_application

    @property
    def flags(self) -> Dict[str, bool]:
        """A set of flags that indicates the type of the semantics."""
        return {
            'is_none': self.is_none,
            'is_conj': self.is_conj,
            'is_py_function': self.is_py_function,
            'is_lazy': self.is_lazy,
            'is_function': self.is_function if not self.is_lazy else None,
            'is_constant': self.is_constant if not self.is_lazy else None,
            'is_function_application': self.is_function_application if not self.is_lazy else None,
            'is_value': self.is_value if not self.is_lazy else None,
        }

    @property
    def return_type(self):
        """The return type of the semantics. If the semantics is a function or a value, the return type is the type of the
        function or the value. Otherwise, this function will raise an error."""
        if self.is_value:
            return self.value.return_type
        elif self.is_function:
            return self.value.ftype.return_type
        else:
            raise AttributeError('Cannot get the return type of None, PyFunction, or Lazy semantics.')

    @cached_property
    def arity(self):
        """The arity of the semantics. If the semantics is a value, this function will return 0. If the semantics
        is a function, this function will return the arity of the function. Otherwise, this function will raise an error."""
        if self.is_value:
            return 0
        elif self.is_function:
            return self.value.ftype.nr_variable_arguments
        else:
            raise AttributeError('Cannot get the arity of None, PyFunction, or Lazy semantics.')

    def __str__(self) -> str:
        if self.value is None:
            return type(self).__name__ + '[None]'
        if self.is_conj:
            return type(self).__name__ + '[' + str(self.value) + ', CONJ]'
        return type(self).__name__ + '[' + str(self.value) + ']'

    def __repr__(self) -> str:
        return str(self)

    @cached_property
    def hash(self):  # for set/dict indexing.
        return hash(str(self))

    def _fapp(self, rhs: 'CCGSemantics') -> 'CCGSemantics':
        if get_ccg_composition_context().semantics_lazy_composition:
            return type(self)(CCGSemanticsLazyValue(CCGCompositionType.FORWARD_APPLICATION, self.value, rhs.value))
        else:
            return type(self)(_forward_application(self.value, rhs.value))

    def _bapp(self, lhs: 'CCGSemantics') -> 'CCGSemantics':
        if get_ccg_composition_context().semantics_lazy_composition:
            return type(self)(CCGSemanticsLazyValue(CCGCompositionType.BACKWARD_APPLICATION, lhs.value, self.value))
        else:
            return type(self)(_backward_application(lhs.value, self.value))

    def _coord3(self, lhs: 'CCGSemantics', rhs: 'CCGSemantics') -> 'CCGSemantics':
        if get_ccg_composition_context().semantics_lazy_composition:
            return type(self)(CCGSemanticsLazyValue(CCGCompositionType.COORDINATION, lhs.value, rhs.value, conj=self.value))
        else:
            return type(lhs)(_coordination(lhs.value, self.value, rhs.value))


def _forward_application(lhs, rhs):
    ctx = get_ccg_composition_context()

    if isinstance(lhs, Function):
        try:
            with FunctionArgumentResolutionContext(exc_verbose=ctx.exc_verbose).as_default():
                return lhs.partial(rhs, execute_fully_bound_functions=True)
        except FunctionArgumentResolutionError as e:
            with ctx.exc(CCGSemanticsCompositionError, e):
                raise CCGSemanticsCompositionError('Cannot make forward application.', lhs, rhs, e) from e
    with ctx.exc(CCGSemanticsCompositionError):
        raise CCGSemanticsCompositionError('Cannot make forward application.', lhs, rhs, 'Functor/Value types do not match.')


def _backward_application(lhs, rhs):
    ctx = get_ccg_composition_context()

    if isinstance(rhs, Function):
        try:
            with FunctionArgumentResolutionContext(exc_verbose=ctx.exc_verbose).as_default():
                return rhs.partial(lhs, execute_fully_bound_functions=True)
        except FunctionArgumentResolutionError as e:
            with ctx.exc(CCGSemanticsCompositionError, e):
                raise CCGSemanticsCompositionError('Cannot make backward application.', lhs, rhs, e) from e
    with ctx.exc(CCGSemanticsCompositionError):
        raise CCGSemanticsCompositionError('Cannot make backward application.', lhs, rhs, 'Functor/Value types do not match.')


def _coordination(lhs, conj, rhs):
    ctx = get_ccg_composition_context()

    if isinstance(lhs, Function) and isinstance(rhs, Function) and callable(conj):
        if lhs.ftype.nr_variable_arguments == rhs.ftype.nr_variable_arguments:
            return conj(lhs, rhs)
    if isinstance(lhs, Value) and isinstance(rhs, Value) and callable(conj):
        return conj(lhs, rhs)

    with ctx.exc(CCGSemanticsCompositionError):
        raise CCGSemanticsCompositionError(
            'Cannot make coordination.',
            lhs, rhs, conj=conj,
            error='Functor arity does not match.'
        )


class CCGSemanticsSugar(object):
    """A syntax sugar that allows users to write CCG semantics in a more natural way.
    This class is initialized in the :class:`~concepts.language.ccg.grammar.CCG` class.
    """

    domain: FunctionDomain

    def __init__(self, domain: FunctionDomain):
        """Initialize the syntax sugar.

        Args:
            domain: the function domain.
        """
        self.domain = domain

    domain: FunctionDomain
    """The underlying function domain."""

    def __getitem__(self, item: Optional[Union[Value, Function, ConstantExpression, FunctionApplicationExpression, Callable, Tuple[Callable, Dict]]]) -> CCGSemantics:
        """Create a :class:`CCGSemantics` instance from a function or a value."""
        if item is None:
            return CCGSemantics(None)
        if isinstance(item, CCGSemantics):
            return item
        if isinstance(item, Value):
            return CCGSemantics(ConstantExpression(item))
        if isinstance(item, (Function, ConstantExpression, FunctionApplicationExpression)):
            return CCGSemantics(item)
        if isinstance(item, tuple):
            assert len(item) == 2
            return CCGSemantics(self.domain.lam(item[0], typing_cues=item[1]))
        assert callable(item)
        return CCGSemantics(self.domain.lam(item))

