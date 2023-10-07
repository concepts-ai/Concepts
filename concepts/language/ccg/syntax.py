#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : syntax.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/05/2020
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Data structures for syntax types in a linguistic CCG."""

from typing import Optional, Union, Tuple, List
from jacinle.utils.cache import cached_property
from jacinle.utils.meta import repr_from_str
from jacinle.utils.printing import indent_text

from concepts.language.ccg.composition import CCGCompositionDirection, CCGCompositionError, get_ccg_composition_context, CCGComposable

__all__ = [
    'CCGSyntaxCompositionError', 'CCGSyntaxTypeParsingError',
    'CCGSyntaxType', 'CCGPrimitiveSyntaxType', 'CCGConjSyntaxType', 'CCGComposedSyntaxType',
    'CCGSyntaxSystem', 'parse_syntax_type'
]


class CCGSyntaxCompositionError(CCGCompositionError):
    """Raised when the composition of two syntax types is not allowed."""

    def __init__(self, message: Optional[str] = None):
        if message is None:
            super().__init__(None)
        else:
            super().__init__('(Syntax) ' + message)


class CCGSyntaxTypeParsingError(Exception):
    """Raised when the parsing of a syntax type string fails."""


class CCGSyntaxType(CCGComposable):
    """Syntax types for CCG.

    There are three types of syntax types:

        - Primitive syntax types: `N`, `S`, `NP`, `VP`, etc.
        - Composed syntax types: `S/NP`, `S\\NP`, etc.
        - Conjunction syntax types: `CONJ`, etc.
    """

    def __init__(self, typename: Optional[str] = None):
        self.typename = typename

    # This property is inherited from CCGComposable.
    @property
    def is_none(self) -> bool:
        return self.typename is None

    # This property is inherited from CCGComposable.
    @property
    def is_conj(self) -> bool:
        return False

    @property
    def arity(self) -> int:
        """The arity of the syntax type. That is, the number of arguments it needs to combine before it becomes a primitive syntax type."""
        return 0

    @property
    def is_function(self) -> bool:
        """Whether the syntax type is a function type. That is, whether it can do function application with another syntax type."""
        return False

    @property
    def is_value(self) -> bool:
        """Whether the syntax type is a value type. That is, whether it is a primitive syntax type."""
        return False

    @property
    def parenthesis_typename(self) -> str:
        """Return the typename with parenthesis."""
        return self.typename

    def _fapp(self, right: 'CCGSyntaxType') -> 'CCGSyntaxType':
        return _forward_application(self, right)

    def _bapp(self, lhs: 'CCGSyntaxType') -> 'CCGSyntaxType':
        return _backward_application(lhs, self)

    def _coord3(self, lhs: 'CCGSyntaxType', rhs: 'CCGSyntaxType') -> 'CCGSyntaxType':
        return _coordination(lhs, self, rhs)

    def __str__(self) -> str:
        return str(self.typename)

    __repr__ = repr_from_str

    def __truediv__(self, other: 'CCGSyntaxType') -> 'CCGSyntaxType':
        """Construct a `A/B` syntax type."""
        return CCGComposedSyntaxType(self, other, direction=CCGCompositionDirection.RIGHT)

    def __floordiv__(self, other: 'CCGSyntaxType') -> 'CCGSyntaxType':
        """Construct a `A\\B` syntax type."""
        return CCGComposedSyntaxType(self, other, direction=CCGCompositionDirection.LEFT)

    def __eq__(self, other: 'CCGSyntaxType') -> bool:
        """Return whether two syntax types are equal."""
        return self.typename == other.typename

    def __ne__(self, other: 'CCGSyntaxType') -> bool:
        return self.typename != other.typename

    def __hash__(self):
        return str(self)

    def __lt__(self, other: 'CCGSyntaxType') -> bool:
        """Customized comparison function for sorting a list of syntax types."""
        a, b = str(self), str(other)
        return (a.count('/') + a.count('\\'), a) < (b.count('/') + b.count('\\'), b)

    def flatten(self) -> List[Union['CCGSyntaxType', Tuple['CCGSyntaxType', CCGCompositionDirection]]]:
        """Flatten the recursive definition of a syntax type into a list of lower-level syntax types. For example,
        the syntax type ``S/NP`` will be flattened into ``[S, NP, (S/NP, RIGHT)]``.

        Returns:
            the list of flattened lower-level syntax types.
        """
        raise NotImplementedError()


class CCGPrimitiveSyntaxType(CCGSyntaxType):
    """The primitive syntax types (e.g., NP)."""

    @property
    def is_value(self) -> bool:
        return True

    def flatten(self) -> List[CCGSyntaxType]:
        return [self]


class CCGConjSyntaxType(CCGSyntaxType):
    """A conjunction syntax type."""

    @property
    def is_conj(self):
        return True

    def __call__(self, lhs: CCGSyntaxType, rhs: CCGSyntaxType) -> CCGSyntaxType:
        """Construct the resulting syntax type for `A CONJ B` given A and B.

        Args:
            lhs: The left syntax type (A).
            rhs: The right syntax type (B).

        Returns:
            CCGSyntaxType: The resulting syntax type.
        """
        return lhs

    def flatten(self) -> List[CCGSyntaxType]:
        return [self]


class CCGComposedSyntaxType(CCGSyntaxType):
    """A composed syntax type (e.g., S/NP)."""

    def __init__(self, main: CCGSyntaxType, sub: CCGSyntaxType, direction: CCGCompositionDirection):
        """Initialize the composed syntax type.

        Args:
            main: the main syntax type (e.g., S).
            sub: the sub syntax type (e.g., NP).
            direction: the composition direction (e.g., RIGHT).
        """

        self.main = main
        self.sub = sub
        self.direction = CCGCompositionDirection.from_string(direction)

        if self.direction is CCGCompositionDirection.RIGHT:
            typename = self.main.typename + '/' + self.sub.parenthesis_typename
        else:
            typename = self.main.typename + '\\' + self.sub.parenthesis_typename
        super().__init__(typename)

    @cached_property
    def arity(self) -> int:
        return self.main.arity + 1

    @property
    def is_function(self) -> bool:
        return True

    @property
    def parenthesis_typename(self) -> str:
        return '{' + f'{self.typename}' + '}'

    def flatten(self) -> List[Union[CCGSyntaxType, Tuple[CCGSyntaxType, CCGCompositionDirection]]]:
        ret = self.main.flatten()
        ret.append((self.sub, self.direction))
        return ret


def _forward_application(lhs, rhs):
    if isinstance(lhs, CCGComposedSyntaxType):
        if lhs.direction == CCGCompositionDirection.RIGHT:
            if lhs.sub == rhs:
                return lhs.main
    with get_ccg_composition_context().exc(CCGSyntaxCompositionError):
        raise CCGSyntaxCompositionError(f'Cannot make forward application of {lhs} and {rhs}.')


def _backward_application(lhs, rhs):
    if isinstance(rhs, CCGComposedSyntaxType):
        if rhs.direction == CCGCompositionDirection.LEFT:
            if rhs.sub == lhs:
                return rhs.main
    with get_ccg_composition_context().exc(CCGSyntaxCompositionError):
        raise CCGSyntaxCompositionError(f'Cannot make backward application of {lhs} and {rhs}.')


def _coordination(lhs, conj, rhs):
    if lhs == rhs and isinstance(conj, CCGConjSyntaxType):
        return conj(lhs, rhs)
    with get_ccg_composition_context().exc(CCGSyntaxCompositionError):
        raise CCGSyntaxCompositionError(f'Cannot make coordination of {lhs} {conj} {rhs}.')


class CCGSyntaxSystem(object):
    """A data structure that keeps track of a set of primitive and conjunction syntax types allowed in a grammar."""

    def __init__(self):
        self.types = dict()

    def define_primitive_type(self, stype: Union[CCGSyntaxType, str]):
        """Define a primitive syntax type.

        Args:
            stype: The syntax type to be defined.
        """
        if isinstance(stype, CCGSyntaxType):
            self.types[stype.typename] = stype
        elif isinstance(stype, str):
            self.types[stype] = CCGPrimitiveSyntaxType(stype)
        else:
            raise TypeError(f'Invalid type: {stype}.')

    def define_conj_type(self, stype: Union[CCGSyntaxType, str]):
        """Define a conj syntax type.

        Args:
            stype: The syntax type to be defined.
        """
        if isinstance(stype, CCGSyntaxType):
            self.types[stype.typename] = stype
        elif isinstance(stype, str):
            self.types[stype] = CCGConjSyntaxType(stype)
        else:
            raise TypeError(f'Invalid type: {stype}.')

    def __getitem__(self, item: Optional[Union[CCGSyntaxType, str]]) -> CCGSyntaxType:
        """A syntax sugar for `parse_syntax_type`.

        - When the string is `None`, return `None`.
        - When the string is a `CCGSyntaxType`, return the type itself.

        Args:
            item: The string to be parsed.

        Returns:
            CCGSyntaxType: The parsed syntax type.
        """
        if item is None:
            return CCGSyntaxType(None)
        if isinstance(item, CCGSyntaxType):
            return item
        return parse_syntax_type(item, syntax_system=self)

    def __str__(self) -> str:
        return 'CCGSyntaxSystem(' + ', '.join([str(x) for x in self.types.keys()]) + ')'

    __repr__ = __str__

    def format_sumamry(self) -> str:
        fmt = 'Primitive and Conjunction types:\n'
        for type in self.types.values():
            fmt += '  ' + str(type) + '\n'
        fmt = 'CCGSyntaxSystem:\n' + indent_text(fmt.rstrip())
        return fmt

    def print_summary(self):
        print(self.format_sumamry())


def parse_syntax_type(string: str, syntax_system: Optional[CCGSyntaxSystem] = None) -> CCGSyntaxType:
    """Parse a string to a syntax type.

    Args:
        string: The string to be parsed.
        syntax_system: The syntax system to be used. Defaults to None.

    Returns:
        CCGSyntaxType: The parsed syntax type.
    """

    def parse_inner(current):
        if current == '':
            raise CCGSyntaxTypeParsingError('Invalid syntax type string (got empty type): {}.'.format(string))
        nr_parenthesis = 0
        last_op = None
        for i, c in enumerate(current):
            if c in r'\/':
                if nr_parenthesis == 0:
                    last_op = i
            if c == '(':
                nr_parenthesis += 1
            elif c == ')':
                nr_parenthesis -= 1
                if nr_parenthesis < 0:
                    raise CCGSyntaxTypeParsingError('Invalid parenthesis (extra ")"): {}.'.format(string))
        if nr_parenthesis != 0:
            raise CCGSyntaxTypeParsingError('Invalid parenthesis (extra "("): {}.'.format(string))

        if last_op is None:
            if current[0] == '(' and current[-1] == ')':
                return parse_inner(current[1:-1])
            else:
                if syntax_system is None:
                    return CCGSyntaxType(current)
                else:
                    if current in syntax_system.types:
                        return syntax_system.types[current]
                    else:
                        raise CCGSyntaxTypeParsingError('Unknown primitive syntax type {} during parsing {}.'.format(current, string))

        last_op_value = CCGCompositionDirection.RIGHT if current[last_op] == '/' else CCGCompositionDirection.LEFT
        return CCGComposedSyntaxType(
            parse_inner(current[:last_op]),
            parse_inner(current[last_op + 1:]),
            direction=last_op_value
        )

    return parse_inner(string)

