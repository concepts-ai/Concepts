#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : dsl_types.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/09/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Data structures for custom types in a DSL.

All types extend the base class :class:`TypeBase`.

:class:`TypeBase` has four derived types::

    - :class:`ObjectType`: corresponds to an object in the physical world.
    - :class:`ValueType`: corresponds to an encoding of a feature of the object.
    - :class:`~concepts.dsl.dsl_functions.FunctionType`: corresponds to a function over the encodings.
    - :class:`~concepts.dsl.dsl_functions.OverloadedFunctionType`: corresponds to a collection of functions that share the same name.

Furthermore, :class:`ValueType` has four derived types::

    - :class:`ConstantType`: corresponds to a constant.
    - :class:`SequenceType`: corresponds to a sequence of values.
    - :class:`TensorValueTypeBase`: corresponds to a tensor-based encoding.
    - :class:`PyObjValueType`: corresponds to an arbitrary Python object.
"""

import torch
from typing import TYPE_CHECKING, Optional, Union, Iterable, Sequence, Tuple

import jacinle
from jacinle.logging import get_logger
from jacinle.utils.defaults import option_context

if TYPE_CHECKING:
    from concepts.dsl.dsl_functions import FunctionType

logger = get_logger(__file__)

__all__ = [
    'FormatContext', 'get_format_context',
    'TypeBase', 'UnionType', 'ObjectType', 'ValueType',
    'AutoType', 'AnyType', 'UnionTyping', 'is_any_type', 'is_auto_type',  # Special types and type constructors.
    'TensorValueTypeBase', 'PyObjValueType', 'ConstantType', 'SequenceValueType',  # Derived from ValueType.
    'ScalarValueType', 'BOOL', 'INT64', 'FLOAT32', 'VectorValueType', 'NamedTensorValueType',  # Derived from TensorValueTypeBase.
    'QINDEX', 'Variable', 'ObjectConstant', 'UnnamedPlaceholder'  # Placeholder-like objects.
]


class FormatContext(option_context(
    '_FormatContext',
    type_format_cls=False, object_format_type=True, function_format_lambda=False, expr_max_length=120, type_varname_sep=': '
)):
    """FormatContext is a context manager that controls the format of types and objects."""

    type_format_cls: bool
    """If True, the type will be formatted as ``Type[typename]``."""

    object_format_type: bool
    """If True, the object format will include the type."""

    function_format_lambda: bool
    """If True, the function format will be lambda-function styled. Otherwise, it will be python-function styled."""

    expr_max_length: int
    """The maximum length of the expression per line."""

    type_varname_sep: str
    """The separator between the type and the variable name."""


get_format_context = FormatContext.get_default


class TypeBase(object):
    """Base class for all types."""

    def __init__(self, typename: str, alias: Optional[str] = None, parent_type: Optional['TypeBase'] = None):
        """Initialize the type.

        Args:
            typename: The name of the type.
            alias: The alias of the type.
        """

        self._typename = typename
        self._alias = alias
        self._parent_type = parent_type

    @property
    def typename(self) -> str:
        """The (full) typename of the type."""
        return self._typename

    @property
    def alias(self) -> Optional[str]:
        """An optional alias of the type."""
        return self._alias

    @property
    def parent_type(self) -> Optional['TypeBase']:
        """The parent type of the type."""
        return self._parent_type

    @property
    def parent_typename(self):
        """Return the typename of the parent type."""
        return self._parent_type.typename

    def __str__(self) -> str:
        if get_format_context().type_format_cls:
            return self.long_str()
        else:
            return self.short_str()

    __repr__ = jacinle.repr_from_str

    def short_str(self) -> str:
        """Return the short string representation of the type."""
        if self.alias is not None:
            return self.alias
        return self.typename

    def long_str(self) -> str:
        """Return the long string representation of the type."""
        return f'Type[{self.short_str()}]'

    def downcast_compatible(self, other: 'TypeBase') -> bool:
        """Check if the type is downcast-compatible with another type.
        That is, if the type is a subtype of the other type.
        """
        return self.typename == other.typename or self == AnyType or self == AutoType

    def __eq__(self, other: 'TypeBase') -> bool:
        return self.typename == other.typename

    def __ne__(self, other: 'TypeBase') -> bool:
        return not (self == other)

    def __hash__(self) -> int:
        return hash(self.typename)


"""AnyType corresponds to the union of all types."""
AnyType = TypeBase('AnyType')


"""AutoType means the type will be automatically inferred later."""
AutoType = TypeBase('AutoType')


def is_any_type(t: TypeBase) -> bool:
    """Check if the type is AnyType."""
    return t == AnyType


def is_auto_type(t: TypeBase) -> bool:
    """Check if the type is AutoType."""
    return t == AutoType


class UnionType(TypeBase):
    """The UnionType is a type that is the union of multiple types."""

    def __init__(self, *types: TypeBase, alias: Optional[str] = None):
        """Initialize the union type.

        Args:
            types: The types in the union.
            alias: The alias of the union type.
        """
        self.types = tuple(types)
        super().__init__(self.long_str(), alias=alias)

    types: Tuple[TypeBase, ...]
    """The underlying types of the union type."""

    def short_str(self) -> str:
        return ' | '.join(t.short_str() for t in self.types)

    def long_str(self) -> str:
        return 'Union[' + ', '.join(t.long_str() for t in self.types) + ']'

    def downcast_compatible(self, other: TypeBase) -> bool:
        raise NotImplementedError('Cannot downcast to a union type.')


class _UnionTypingSugar(object):
    def __getitem__(self, item: Tuple[TypeBase]):
        assert isinstance(item, tuple)
        return UnionType(*item)


UnionTyping = _UnionTypingSugar()


class ObjectType(TypeBase):
    """The ObjectType corresponds to the type of "real-world" objects."""

    def __init__(self, typename: str, parent_types: Optional[Sequence['ObjectType']] = None, alias: Optional[str] = None):
        """Initialize the object type.

        Args:
            typename: The name of the object type.
            alias: The alias of the object type.
        """
        super().__init__(typename, alias=alias)

        self.parent_types = tuple(parent_types) if parent_types is not None else tuple()

    parent_types: Tuple['ObjectType', ...]
    """The parent types of the object type."""

    def iter_parent_types(self) -> Iterable['ObjectType']:
        """Iterate over all parent types.

        Yields:
            the parent types following the inheritance order.
        """
        for parent_type in self.parent_types:
            yield parent_type
            yield from parent_type.iter_parent_types()

    def long_str(self) -> str:
        if len(self.parent_types) == 0:
            return f'OT[{self.typename}]'

        return f'OT[{self.typename}, parent={", ".join(t.typename for t in self.parent_types)}]'


class ValueType(TypeBase):
    """The ValueType corresponds to a value of a certain type."""

    def long_str(self) -> str:
        return f'VT[{self.typename}]'

    def assignment_type(self) -> 'ValueType':
        """Return the value type for assignment."""
        return self


class TensorValueTypeBase(ValueType):
    """A value type refers to a type of some features associated with physical objects."""

    def __init__(self, typename: str, quantized: bool, alias: Optional[str] = None):
        super().__init__(typename, alias)
        self._quantized = quantized

    @property
    def quantized(self) -> bool:
        """Whether the tensor is quantized."""
        return self._quantized

    def ndim(self) -> int:
        """Return the number of dimensions of the value type."""
        raise NotImplementedError()

    def size(self) -> int:
        """Return the total size of the value type."""
        raise NotImplementedError()

    def size_tuple(self) -> Tuple[int]:
        """Return the size of the value type as a tuple."""
        raise NotImplementedError()

    def tensor_dtype(self) -> torch.dtype:
        """Return the corresponding PyTorch tensor dtype."""
        raise NotImplementedError()

    def is_intrinsically_quantized(self) -> bool:
        """Return whether the value type can be quantized.
        In this project, quantizable means we can transform the value into a single integer scalar.
        Therefore, only BOOL-valued and INTxx-valued tensors are quantizable.

        Returns:
            True if the value type can be quantized.
        """
        raise NotImplementedError()

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> 'TensorValueTypeBase':
        """Create a value type from a pytorch tensor.

        Args:
            tensor: the tensor.

        Returns:
            the tensor value type.
        """
        if tensor.dtype in (torch.float32, torch.float64):
            return FLOAT32
        elif tensor.dtype in (torch.int32, torch.int64):
            return INT64
        elif tensor.dtype == torch.bool:
            return BOOL
        else:
            raise TypeError(f'Unsupported tensor type: {tensor.dtype}.')


class PyObjValueType(ValueType):
    def __init__(self, pyobj_type: Union[type, str], alias: Optional[str] = None):
        self.pyobj_type = pyobj_type
        super().__init__(str(pyobj_type), alias=alias)

    pyobj_type: Union[type, str]
    """The underlying Python object type."""


class SequenceValueType(ValueType):
    """The SequenceValueType corresponds to a sequence of values of a certain type."""

    def __init__(self, value_type: ValueType, alias: Optional[str] = None):
        """Initialize the sequence value type.

        Args:
            value_type: The value type of the sequence.
        """
        super().__init__(f'Sequence[{value_type.typename}]', alias=alias)
        self.value_type = value_type

    value_type: ValueType
    """The value type of the sequence."""

    def long_str(self) -> str:
        return f'SequenceT[{self.value_type.typename}]'

    def assignment_type(self) -> 'ValueType':
        """Return the value type for assignment."""
        return self


class ConstantType(ValueType):
    """The ConstantType corresponds to a constant value."""

    def long_str(self) -> str:
        return f'CT[{self.typename}]'


class ScalarValueType(TensorValueTypeBase):
    TENSOR_DTYPE_MAPPING = {
        'int32': torch.int32,
        'int64': torch.int64,
        'float32': torch.float32,
        'float64': torch.float64,
        'bool': torch.int64,
    }

    TENSOR_DTYPE_QUANTIZED = {
        'int32': True,
        'int64': True,
        'float32': False,
        'float64': False,
        'bool': True
    }

    def __init__(self, typename):
        if typename not in type(self).TENSOR_DTYPE_MAPPING:
            raise TypeError(f'Unknown scalar value type: {typename}.')
        super().__init__(typename, quantized=type(self).TENSOR_DTYPE_QUANTIZED[typename])

    def ndim(self) -> int:
        return 0

    def size(self) -> int:
        return 1

    def size_tuple(self) -> Tuple[int]:
        return tuple()

    def tensor_dtype(self) -> torch.dtype:
        return type(self).TENSOR_DTYPE_MAPPING[self.typename]

    INTRINSICALLY_QUANTIZED_DTYPES = ['bool', 'int32', 'int64']

    def is_intrinsically_quantized(self) -> bool:
        return self.typename in self.INTRINSICALLY_QUANTIZED_DTYPES

    def __eq__(self, o: object) -> bool:
        if isinstance(o, ScalarValueType):
            return self.typename == o.typename
        return False

    def __hash__(self):
        return hash(self.typename)


BOOL = ScalarValueType('bool')
INT64 = ScalarValueType('int64')
FLOAT32 = ScalarValueType('float32')


class VectorValueType(TensorValueTypeBase):
    def __init__(self, dtype: ScalarValueType, dim: int = 0, choices: int = 0, factors: int = 1, alias: Optional[str] = None):
        assert isinstance(dtype, ScalarValueType), 'Currently only support 1D vectors.'
        self.dtype = dtype
        self.dim = dim
        self.choices = choices
        self.factors = factors

        typename = f'vector[{self.dtype}, dim={self.dim}, choices={self.choices}, factors={self.factors}]'
        quantized = choices > 0
        super().__init__(typename, quantized=quantized, alias=alias)

    dtype: ScalarValueType
    """The scalar value type of the vector."""

    dim: int
    """The dimension of the vector."""

    choices: int
    """The number of choices for vector values per factor."""

    factors: int
    """The number of factors of the vector."""

    def ndim(self) -> int:
        return 1

    def size(self) -> int:
        return self.dim * self.factors

    def size_tuple(self) -> Tuple[int]:
        return (self.size(), )

    def assignment_type(self) -> 'VectorValueType':
        if self._quantized:
            return VectorValueType(FLOAT32, self.choices, 0, self.factors)
        return self

    def tensor_dtype(self) -> torch.dtype:
        return self.dtype.tensor_dtype()

    def is_intrinsically_quantized(self) -> bool:
        return False

    def long_str(self):
        if self.alias is not None:
            return self.alias + f'({self.typename})'
        return self.typename


class NamedTensorValueType(TensorValueTypeBase):
    def __init__(self, typename: str, base_type: TensorValueTypeBase):
        super().__init__(typename, base_type._quantized)
        self.base_type = base_type

    base_type: TensorValueTypeBase
    """The base tensor value type."""

    def ndim(self) -> int:
        return self.base_type.ndim()

    def size(self) -> int:
        return self.base_type.size()

    def size_tuple(self) -> Tuple[int]:
        return self.base_type.size_tuple()

    def assignment_type(self) -> TensorValueTypeBase:
        return self

    def tensor_dtype(self) -> torch.dtype:
        return self.base_type.tensor_dtype()

    def is_intrinsically_quantized(self) -> bool:
        return self.base_type.is_intrinsically_quantized()

    def __eq__(self, o: object) -> bool:
        if isinstance(o, NamedTensorValueType):
            return self.typename == o.typename
        return False

    def __hash__(self):
        return hash(self.typename)


QINDEX = slice(None)
"""The "quantified index" object, used to indicate a quantified dimension in a tensor value type. Internally it is `slice(None)`."""


class _Placeholder(object):
    def __init__(self, name: str, dtype: Union[ObjectType, ValueType, 'FunctionType']):
        self.name = name
        self.dtype = dtype

    name: str
    """The name of the placeholder."""

    dtype: Union[ObjectType, ValueType, 'FunctionType']
    """The data type of the placeholder."""

    @property
    def typename(self):
        return self.dtype.typename

    def __str__(self):
        sep = get_format_context().type_varname_sep
        return f'{self.name}{sep}{self.dtype}'

    __repr__ = jacinle.repr_from_str


class Variable(_Placeholder):
    """The class representing a variable in a function."""

    name: str
    """The name of the variable."""

    dtype: Union[ObjectType, ValueType, 'FunctionType']
    """The data type of the variable."""

    quantifier_flag: Optional[str] = None
    """Additional quantifier flag for the variable. This flag will be set by the parser indicating the quantifier scope of this variable.
    Currently this is only used in PDSketch for "quantified variable" in regression rules. """

    def set_quantifier_flag(self, flag: str):
        self.quantifier_flag = flag

    def __str__(self):
        if self.quantifier_flag is None:
            return super().__str__()
        else:
            return f'({self.quantifier_flag} {super().__str__()})'


class ObjectConstant(_Placeholder):
    """The class representing a constant object in a DSL."""

    name: str
    """The name of the constant."""

    dtype: ObjectType
    """The data type of the constant."""


class UnnamedPlaceholder(_Placeholder):
    def __init__(self, dtype: Union[ObjectType, ValueType]):
        super().__init__('_', dtype)

    name: str
    """The name of the unnamed placeholder. Always equal to '_'."""

    dtype: Union[ObjectType, ValueType]
    """The data type of the unnamed placeholder."""

    def __str__(self):
        return '??' + str(self.dtype)

