#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : cdl_literal_parser.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/15/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from dataclasses import dataclass
from typing import Any, Union, Tuple, Set, Dict

from lark import Tree
from lark.visitors import Transformer, v_args

from concepts.dm.crow.crow_domain import CrowDomain
from concepts.dsl.dsl_types import BatchedListType, ListType, TypeBase, Variable, VectorValueType

inline_args = v_args(inline=True)

__all__ = ['LiteralValue', 'LiteralList', 'LiteralSet', 'InTypedArgument', 'ArgumentsDef', 'CSList', 'CDLLiteralTransformer']


@dataclass
class LiteralValue(object):
    """A literal value."""
    value: Union[bool, int, float, complex, str]


@dataclass
class LiteralList(object):
    """A list of literals."""
    items: Tuple[Union[bool, int, float, complex, str, 'LiteralList', 'LiteralSet'], ...]


@dataclass
class LiteralSet(object):
    """A set of literals."""
    items: Set[Union[bool, int, float, complex, str, 'LiteralList', 'LiteralSet']]


@dataclass
class InTypedArgument(object):
    """A typed argument defined as `name in value`. This is used in forall/exists statements."""
    name: str
    value: Any


@dataclass
class ArgumentsDef(object):
    arguments: Tuple[Variable, ...]


@dataclass
class CSList(object):
    """A comma-separated list of something."""

    items: Tuple[Any, ...]

    def __str__(self):
        return f'CSList({", ".join(str(item) for item in self.items)})'

    def __repr__(self):
        return self.__str__()


class CDLLiteralTransformer(Transformer):
    """The transformer for literal types. Including:

    - VARNAME, CONSTNAME, BASIC_TYPENAME
    - number, DEC_NUMBER, HEX_NUMBER, BIN_NUMBER, OCT_NUMBER, FLOAT_NUMBER, IMAG_NUMBER
    - boolean, TRUE, FALSE
    - string
    - literal_list
    - literal_set
    - decorator_k, decorator_kwarg, decorator_kwargs
    """

    domain: CrowDomain

    @inline_args
    def typename(self, name: Union[str, TypeBase]) -> TypeBase:
        """Captures typenames including basic types and vector types."""
        return name

    @inline_args
    def sized_vector_typename(self, name: Union[str, TypeBase], size: int) -> VectorValueType:
        """Captures sized vector typenames defined as `vector[typename, size]`."""
        return VectorValueType(self.domain.get_type(name), size)

    @inline_args
    def unsized_vector_typename(self, name: Union[str, TypeBase]) -> VectorValueType:
        """Captures unsized vector typenames defined as `vector[typename]`."""
        return VectorValueType(self.domain.get_type(name))

    @inline_args
    def batched_typename(self, element_dtype: Union[str, TypeBase], indices: Tree) -> BatchedListType:
        """Captures batched typenames defined as `typename[indices]`."""
        element_dtype = self.domain.get_type(element_dtype)
        return BatchedListType(element_dtype, [self.domain.get_type(name) for name in indices.children])

    @inline_args
    def list_typename(self, element_dtype: Union[str, TypeBase]) -> ListType:
        """Captures list typenames defined as `list[typename]`."""
        return ListType(self.domain.get_type(element_dtype))

    @inline_args
    def typed_argument(self, name: str, typename: Union[str, TypeBase]) -> Variable:
        """Captures typed arguments defined as `name: typename`."""
        if isinstance(typename, str):
            typename = self.domain.get_type(typename)
        return Variable(name, typename)

    @inline_args
    def multi_typed_arguments(self, name: Tree, typename: Union[str, TypeBase]) -> 'CSList':
        """Captures multiple typed arguments defined as `name1, name2: typename`."""
        if isinstance(typename, str):
            typename = self.domain.get_type(typename)
        return CSList(tuple(Variable(n, typename) for n in name.children))

    @inline_args
    def is_typed_argument(self, name: str, typename: Union[str, TypeBase]) -> Variable:
        """Captures typed arguments defined as `name is typename`. This is used in forall/exists statements."""
        if isinstance(typename, str):
            typename = self.domain.get_type(typename)
        return Variable(name, typename)

    @inline_args
    def in_typed_argument(self, name: str, value: Any) -> InTypedArgument:
        """Captures typed arguments defined as `name in value`. This is used in forall/exists statements."""
        return InTypedArgument(name, value)

    def arguments_def(self, args):
        """Captures the arguments definition. This is used in function definitions."""
        return ArgumentsDef(tuple(args))

    def VARNAME(self, token):
        """Captures variable names, such as `var_name`."""
        return token.value

    def CONSTNAME(self, token):
        """Captures constant names, such as `CONST_NAME`."""
        return token.value

    def BASIC_TYPENAME(self, token):
        """Captures basic type names (non-vector types), such as `int`, `float`, `bool`, `object`, etc."""
        return token.value

    @inline_args
    def number(self, value: Union[int, float, complex]) -> Union[int, float, complex]:
        """Captures number literals, including integers, floats, and complex numbers."""
        return value

    @inline_args
    def BIN_NUMBER(self, value: str) -> int:
        """Captures binary number literals."""
        return int(value, 2)

    @inline_args
    def OCT_NUMBER(self, value: str) -> int:
        """Captures octal number literals."""
        return int(value, 8)

    @inline_args
    def DEC_NUMBER(self, value: str) -> int:
        """Captures decimal number literals."""
        return int(value)

    @inline_args
    def HEX_NUMBER(self, value: str) -> int:
        """Captures hexadecimal number literals."""
        return int(value, 16)

    @inline_args
    def FLOAT_NUMBER(self, value: str) -> float:
        """Captures floating point number literals."""
        return float(value)

    @inline_args
    def IMAG_NUMBER(self, value: str) -> complex:
        """Captures complex number literals."""
        return complex(value)

    @inline_args
    def boolean(self, value: bool) -> bool:
        """Captures boolean literals."""
        return value

    @inline_args
    def TRUE(self, _) -> bool:
        """Captures the `True` literal."""
        return True

    @inline_args
    def FALSE(self, _) -> bool:
        """Captures the `False` literal."""
        return False

    @inline_args
    def ELLIPSIS(self, _) -> str:
        """Captures the `...` literal."""
        return Ellipsis

    @inline_args
    def string(self, value: str) -> str:
        """Captures string literals."""
        if value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        return str(value)

    @inline_args
    def literal_list(self, *items: Any) -> LiteralList:
        """Captures literal lists, such as `[1, 2, 3, 4]`."""
        return LiteralList(tuple(items))

    @inline_args
    def literal_set(self, *items: Any) -> LiteralSet:
        """Captures literal sets, such as `{1, 2, 3, 4}`."""
        return LiteralSet(set(items))

    @inline_args
    def literal(self, value: Union[bool, int, float, complex, str, LiteralList, LiteralSet]) -> Union[LiteralValue, LiteralList, LiteralSet]:
        """Captures literal values."""
        if isinstance(value, (bool, int, float, complex, str)):
            return LiteralValue(value)
        elif isinstance(value, (LiteralList, LiteralSet)):
            return value
        else:
            raise ValueError(f'Invalid literal value: {value}')

    @inline_args
    def decorator_kwarg(self, k, v: Union[LiteralValue, LiteralList, LiteralSet] = True) -> Tuple[str, Union[bool, int, float, complex, str, LiteralList, LiteralSet]]:
        """Captures the key-value pair of a decorator. This is used in the decorator syntax, such as [[k=True]]."""
        return k, v.value if isinstance(v, LiteralValue) else v

    def decorator_kwargs(self, args) -> Dict[str, Union[bool, int, float, complex, str, LiteralList, LiteralSet]]:
        """Captures the key-value pairs of a decorator. This is used in the decorator syntax, such as [[k=True, k2=123, k3=[1, 2, 3]]]."""
        return {k: v for k, v in args}
