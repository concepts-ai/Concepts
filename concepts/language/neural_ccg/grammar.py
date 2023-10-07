#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : grammar.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/07/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""The grammar components for neural CCGs, including syntax types, semantic forms, lexicon, and the parser.
There are a few debug options that can be enabled by setting the corresponding flags to True, see the
:class:`foptions` variable for details.
"""

import math
import heapq
import itertools

from typing import TYPE_CHECKING, TypeVar, Any, Optional, Union, Iterable, Sequence, Tuple, List, Set, Dict, Callable
from collections import defaultdict, Counter
from dataclasses import dataclass
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.nn.functional as F
import jacinle
import jactorch

from jacinle.config.environ_v2 import configs, def_configs
from jacinle.utils.cache import cached_property
from jacinle.utils.meta import repr_from_str

from concepts.dsl.dsl_types import FormatContext, TypeBase, ObjectType, ValueType, ConstantType
from concepts.dsl.dsl_functions import FunctionType, Function
from concepts.dsl.value import Value
from concepts.dsl.expression import ConstantExpression, FunctionApplicationExpression, get_type
from concepts.dsl.function_domain import FunctionDomain
from concepts.dsl.executors.function_domain_executor import FunctionDomainExecutor
from concepts.language.ccg.composition import CCGCompositionType, CCGCompositionDirection, CCGCompositionSystem, CCGComposable, CCGCoordinationImmNode, CCGCompositionResult
from concepts.language.ccg.composition import CCGCompositionError, get_ccg_composition_context, CCGCompositionContext
from concepts.language.ccg.syntax import CCGSyntaxType, CCGSyntaxCompositionError
from concepts.language.ccg.semantics import CCGSemantics, CCGSemanticsLazyValue
from concepts.language.ccg.grammar import CCGNode, Lexicon

if TYPE_CHECKING:
    from concepts.language.neural_ccg.search import NeuralCCGLexiconSearchResult

T = TypeVar('T')


__all__ = [
    'foptions',
    'LinearizationTuple', 'parse_linearization_string', 'NeuralCCGSyntaxType', 'NeuralCCGConjSyntaxType',
    'NeuralCCGSemanticsPartialTypeLex', 'NeuralCCGSemantics',
    'NeuralCCGGroundingFunction', 'NeuralCCGSemanticsExecutionBuffer', 'NeuralCCGConjGroundingFunction', 'NeuralCCGSimpleConjFunction',
    'NeuralCCGNode', 'compose_neural_ccg_node', 'NeuralCCG',
]


logger = jacinle.get_logger(__file__)
_profile = getattr(__builtins__, 'profile', lambda x: x)


with def_configs():
    configs.neural_ccg.debug_print = False
    configs.neural_ccg.debug_stat_types = False
    configs.neural_ccg.training_use_lazy_semantics = True
    configs.neural_ccg.compose_function_note = False


foptions = configs.neural_ccg
"""Options for the CCG parser. Use ``foptions.xxx`` to access the options.

- ``debug_print``: print debug information.
- ``debug_stat_types``: print statistics about the types.
- ``training_use_lazy_semantics``: whether to use lazy semantics during training.
- ``compose_function_note``: whether to compose the function notes of semantics.
    Specifically, if this is set to True, then every :class:`NeuralCCGGroundingFunction` instance will compose a note
    that records the trace of the fucntions that has been called.
"""


@dataclass
class LinearizationTuple(object):
    """A pair of (index, direction) for representing linearization of function arguments."""

    index: int
    """The index of the argument."""

    direction: CCGCompositionDirection
    """The direction of the linearization (left or right)."""

    def __str__(self) -> str:
        if self.direction is CCGCompositionDirection.LEFT:
            return f'{self.index}<'
        else:
            return f'{self.index}>'


def parse_linearization_string(string: str) -> List[LinearizationTuple]:
    r"""Parse a linearization string (e.g., ``/0\1``) into a list of tuples.

    Args:
        string: the linearization string.

    Returns:
        A list of tuples, each tuple is a pair of ``(index, direction)``.
    """
    if string == '':
        return []
    string = string.replace('/', '@/').replace('\\', '@\\')
    assert string.startswith('@')
    parts = string.split('@')[1:]
    linearization = list()
    for p in parts:
        linearization.append(LinearizationTuple(
            int(p[1:]),
            CCGCompositionDirection.RIGHT if p[0] == '/' else CCGCompositionDirection.LEFT
        ))
    return linearization


class NeuralCCGSyntaxType(CCGComposable):
    """Data structure for neural CCG syntax types.
    NeuralCCGSyntaxType implements a different version of the syntax type than the CCGSyntaxType.

    Specifically, the :class:`NeuralCCGSyntaxType` define syntaxes based on the signature of its meaning. It has two parts:

    1. the signature of the semantic form, including the argument types and the return type.
    2. the linearization of the semantics.

    IMPORTANT: Arguments are combined from right to left. Thus, argument_types[-1] is the first argument.
    There is one possible confusion about linearization:

    Although LinearizationTuple has two fields: index and direction, only type will be used during computation.
    LinearizationTuple.index is used to record the mapping from the original "function" definition, most likely
    produced by iter_from_function.

    Example: Suppose we have a function definition def func(string, int) -> string.
    Running iter_from_function(func) will get you six possible syntax types:

        string/0:string/1:int, string\\0:string/1:int, string\\0:string\\1:int,
        string/1:int/0:string, string\\1:int/0:string, string\\1:int\\0:string

    Consider a specifc type string\\0:int/1:string. It has the following fields:

    - return_type: string
    - argument_types: [int, string]
    - linearization: [LinearizationTuple(0, CCGCompositionDirection.LEFT), LinearizationTuple(1, CCGCompositionDirection.RIGHT)]

    Here, the first argument should come from fapp, of type string. The second argument should come from bapp,
    of type int. The linearization[i].index is never used in the computation. However, it does record the argument
    order mapping from the original "func(string, int) -> string".

    See also :meth:`NeuralCCGSyntaxType.iter_from_function`.

    See also :meth:`concepts.language.neural_ccg.search.NeuralCCGLexiconSearcher.gen_lexicons`.
    """

    return_type: Union[None, str, ObjectType, ValueType]
    """The return type of the syntax."""

    argument_types: Tuple[Union[ObjectType, ValueType, FunctionType], ...]
    """The argument types of the syntax."""

    linearization: Tuple[LinearizationTuple, ...]
    """The linearization of arguments. See the docstring for details."""

    lang_syntax_type: Optional[CCGSyntaxType]
    """Optionally, a :class:`~concepts.language.ccg.CCGSyntaxType` can be associated with the :class:`NeuralCCGSyntaxType`."""

    def __init__(
        self,
        return_type: Union[None, str, ObjectType, ValueType],
        argument_types: Optional[Iterable[Union[ObjectType, ValueType, FunctionType]]] = None,
        linearization: Optional[Iterable[LinearizationTuple]] = None,
        lang_syntax_type: Optional[CCGSyntaxType] = None,
        function_typename: Optional[str] = None
    ):
        self.return_type = return_type
        self.argument_types = tuple(argument_types) if argument_types is not None else tuple()
        self.linearization = tuple(linearization) if linearization is not None else tuple()
        self.lang_syntax_type = lang_syntax_type
        self._function_typename = function_typename

        if self.is_function and self._function_typename is None:
            self._function_typename = FunctionType(self.argument_types, self.return_type).typename

    def derive_lang_syntax_type(self, lang_syntax_type: CCGSyntaxType) -> 'NeuralCCGSyntaxType':
        """Create a new syntax type that has an associated standard linguistic syntax type.

        Args:
            lang_syntax_type: the syntax type.

        Returns:
            A new :class:`NeuralCCGSyntaxType` with the same signature and linearization, but with an associated
            standard linguistic syntax type.
        """
        return NeuralCCGSyntaxType(self.return_type, self.argument_types, self.linearization, lang_syntax_type)

    @classmethod
    def iter_from_function(
        cls, function: Function, nr_used_arguments: Optional[int] = None
    ) -> Iterable[Tuple['NeuralCCGSyntaxType', Function]]:
        """
        Create all possible linearization of the corresponding function.

        Args:
            function: the function.
            nr_used_arguments: do enumeration only for the first N arguments. Useful when the function has constant
                arguments that will be bound with word types.
        Yield:
            all possible :class:`NeuralCCGSyntaxType` derived from the given function signature.
        """
        rtype = function.ftype.return_type
        if nr_used_arguments is None:
            nr_used_arguments = len(function.ftype.argument_types)
        argtypes = function.ftype.argument_types[:nr_used_arguments]

        for indices in itertools.permutations(range(len(argtypes))):
            for nr_left in range(0, 1 + len(argtypes)):
                linearization = list()
                linearization.extend([
                    LinearizationTuple(x, CCGCompositionDirection.LEFT)
                    for x in indices[:nr_left]
                ])
                linearization.extend([
                    LinearizationTuple(x, CCGCompositionDirection.RIGHT)
                    for x in indices[nr_left:]
                ])
                permuted_argtypes = [argtypes[i] for i in indices]
                new_func = function.remap_arguments(
                    [x.index for x in linearization[::-1]] +
                    list(range(nr_used_arguments, function.ftype.nr_arguments))
                )
                syntax_type = NeuralCCGSyntaxType(rtype, permuted_argtypes, linearization, function_typename=new_func.ftype.typename)
                yield syntax_type, new_func

    @cached_property
    def typename(self) -> str:
        """The functional typename. Does NOT include the linearization.

        Returns:
            The functional typename.
        """
        if self.is_none:
            return '<None>'
        elif self.is_function:
            # NB(Jiayuan Mao @ 04/05): this line should be synced up with typing.py FunctionType::_gen_typename().
            return self._function_typename
        elif self.is_value:
            return self.return_type.typename
        elif self.is_conj:
            return self.return_type

    # This property is inherited from CCGComposable.
    @property
    def is_none(self) -> bool:
        return self.return_type is None

    # This property is inherited from CCGComposable.
    @property
    def is_conj(self) -> bool:
        return False

    @property
    def arity(self) -> int:
        """The arity of the syntax type. That is, the number of arguments it needs to combine before it becomes a primitive syntax type."""
        if self.is_function:
            return len(self.argument_types)
        elif self.is_value:
            return 0
        else:
            raise AttributeError('Cannot get the arity of None syntax.')

    @property
    def is_function(self) -> bool:
        """Whether the syntax type is a function type. That is, whether it can do function application with another syntax type."""
        return not self.is_none and not self.is_conj and self.argument_types is not None and len(self.argument_types) > 0

    @property
    def is_value(self) -> bool:
        """Whether the syntax type is a value type. That is, whether it is a primitive syntax type."""
        return not self.is_none and not self.is_conj and (self.argument_types is None or len(self.argument_types) == 0)

    @_profile
    def _fapp(self, rhs: 'NeuralCCGSyntaxType') -> 'NeuralCCGSyntaxType':
        new_syntax_type = None
        if self.lang_syntax_type is not None:
            assert rhs.lang_syntax_type is not None
            new_syntax_type = self.lang_syntax_type.fapp(rhs.lang_syntax_type)

        if self.is_function:
            if self.argument_types[-1].typename == rhs.typename and self.linearization[-1].direction is CCGCompositionDirection.RIGHT:
                return NeuralCCGSyntaxType(self.return_type, self.argument_types[:-1], self.linearization[:-1], lang_syntax_type=new_syntax_type)
        raise CCGCompositionError()

    @_profile
    def _bapp(self, lhs: 'NeuralCCGSyntaxType') -> 'NeuralCCGSyntaxType':
        new_syntax_type = None
        if self.lang_syntax_type is not None:
            assert lhs.lang_syntax_type is not None
            new_syntax_type = self.lang_syntax_type.bapp(lhs.lang_syntax_type)

        if self.is_function:
            if self.argument_types[-1].typename == lhs.typename and self.linearization[-1].direction is CCGCompositionDirection.LEFT:
                return NeuralCCGSyntaxType(self.return_type, self.argument_types[:-1], self.linearization[:-1], lang_syntax_type=new_syntax_type)
        raise CCGCompositionError()

    def _coord3(self, lhs: 'NeuralCCGSyntaxType', rhs: 'NeuralCCGSyntaxType') -> 'NeuralCCGSyntaxType':
        raise NotImplementedError('Coordination is not supported for primitive neural CCG syntax types. Use NeuralCCGConjSyntaxType instead.')

    def __str__(self) -> str:
        if self.is_none:
            return '<None>'
        if isinstance(self.return_type, str):
            fmt = self.return_type
        else:
            fmt = self.return_type.short_str()
        if self.is_function:
            fmt = self.return_type.short_str()
            for arg, lin in zip(self.argument_types, self.linearization):
                if lin.direction is CCGCompositionDirection.LEFT:
                    fmt = fmt + '\\' + arg.short_str()
                else:
                    fmt = fmt + '/' + arg.short_str()
        if self.lang_syntax_type is not None:
            fmt += '[' + str(self.lang_syntax_type) + ']'
        return fmt

    __repr__ = repr_from_str

    def __eq__(self, other: 'NeuralCCGSyntaxType') -> bool:
        return (
            self.return_type == other.return_type and
            self.argument_types == other.argument_types and
            self.linearization == other.linearization
        )

    def __hash__(self):
        return hash(str(self))

    def __lt__(self, other: 'NeuralCCGSyntaxType') -> bool:
        a, b = str(self), str(other)
        return (a.count('/') + a.count('\\'), a) < (b.count('/') + b.count('\\'), b)


class NeuralCCGConjSyntaxType(NeuralCCGSyntaxType):
    """Data structure for conjunction syntax types. Conjunction syntax types are represented as a single string (stored in ``return_type``."""

    return_type: str
    """The return type of the syntax type. For conjunction syntax types, this is the string name of the conjunction."""

    argument_types: Tuple[TypeBase, ...]

    linearization: Tuple[LinearizationTuple, ...]

    lang_syntax_type: Optional[CCGSyntaxType]

    def __init__(self, typename: str):
        super().__init__(typename)

    @property
    def is_conj(self):
        return True

    def __call__(self, lhs: NeuralCCGSyntaxType, rhs: NeuralCCGSyntaxType) -> NeuralCCGSyntaxType:
        """Perform conjunction of two syntax types."""
        return lhs

    @_profile
    def _coord3(self, lhs: NeuralCCGSyntaxType, rhs: NeuralCCGSyntaxType) -> NeuralCCGSyntaxType:
        if lhs == rhs:
            return self(lhs, rhs)
        raise CCGSyntaxCompositionError()


@dataclass
class NeuralCCGSemanticsPartialTypeLex(object):
    """A data structure that will be used in representing the semantic type of a partially applied function."""

    candidate_lexicon_index: int
    """The index of the candidate lexicon entry."""

    word_index: Optional[int]
    """The index of the word in the sentence. This will only be set if the lexicon entry contains constants."""

    def __str__(self):
        return f'Lex({self.candidate_lexicon_index}, word_binding={self.word_index})'

    def __repr__(self):
        return str(self)


class NeuralCCGSemantics(CCGSemantics):
    """
    We implement the execution result of the current subtree in the following way:
    For primitive types (syntax.is_value == True), the execution_buffer field is a list containing a single element
    recording the execution result.
    For function types (syntax.is_function == True), the execution_buffer stores a partially bound function application.
    It is stored as a list, where the first element is the function and the rest are bound arguments (the order of
    these elements is decided by syntax.linearization).

    There are two auxiliary properties of this class:

    - `partial_type`: It summarizes the intermediate execution result of the composition. Specifically, if there is only one string in the list,
        the string is the type of the semantics LF. Otherwise, the first element is of type NeuralCCGSemanticsPartialTypeLex,
        which records the candidate lexicon index and the word index (which can be None).
    - `nr_execution_steps`: the number of forwarding steps that have been executed during the construction of this subtree.
    """

    def __init__(
        self,
        value: Union[None, Callable, Function, ConstantExpression, FunctionApplicationExpression, CCGSemanticsLazyValue],
        execution_buffer: Optional[List[Union[None, Callable, Value]]] = None,
        partial_type: Optional[List[str]] = None,
        nr_execution_steps: int = 0,
        is_conj: bool = False
    ):
        """Initialize a neural CCG semantics object.

        Args:
            value: the value of the semantics.
            execution_buffer: the buffer for partially executed values. See the docstring for details.
            partial_type: the partial type of the semantics. See the docstring for details.
            nr_execution_steps: the number of execution steps that have been executed.
            is_conj: whether the semantics is a conjunction.
        """
        super().__init__(value, is_conj=is_conj)

        self.execution_buffer = execution_buffer
        self.partial_type = partial_type
        self.nr_execution_steps = nr_execution_steps

    execution_buffer: Optional[List[Union[None, Callable, Value]]]
    """The execution buffer for partially executed values."""

    partial_type: Optional[List[str]]
    """The partial type of the semantics."""

    nr_execution_steps: int
    """The number of execution steps that have been executed."""

    @_profile
    def _fapp(self, rhs: 'NeuralCCGSemantics') -> 'NeuralCCGSemantics':
        output: NeuralCCGSemantics = super()._fapp(rhs)
        output.execution_buffer = _merge_list(self.execution_buffer, rhs.execution_buffer)
        output.partial_type = _merge_list(self.partial_type, rhs.partial_type)
        output.nr_execution_steps = self.nr_execution_steps + rhs.nr_execution_steps
        return output.canonize_execution_buffer()

    @_profile
    def _bapp(self, lhs: 'NeuralCCGSemantics') -> 'NeuralCCGSemantics':
        output: NeuralCCGSemantics = super()._fapp(lhs)
        output.execution_buffer = _merge_list(self.execution_buffer, lhs.execution_buffer)
        output.partial_type = _merge_list(self.partial_type, lhs.partial_type)
        output.nr_execution_steps = self.nr_execution_steps + lhs.nr_execution_steps
        return output.canonize_execution_buffer()

    @_profile
    def _coord3(self, lhs: 'NeuralCCGSemantics', rhs: 'NeuralCCGSemantics') -> 'NeuralCCGSemantics':
        output: NeuralCCGSemantics = super()._coord3(lhs, rhs)
        assert isinstance(self.execution_buffer[0], NeuralCCGConjGroundingFunction)
        ret = self.execution_buffer[0](
            lhs.execution_buffer,
            lhs.partial_type,
            rhs.execution_buffer,
            rhs.partial_type
        )
        output.execution_buffer = ret.execution_buffer
        output.partial_type = ret.partial_type
        output.nr_execution_steps = lhs.nr_execution_steps + rhs.nr_execution_steps + ret.nr_execution_steps
        return output

    def canonize_execution_buffer(self) -> 'NeuralCCGSemantics':
        """Canonize the execution buffer. Specifically, if the execution buffer has all argument values to the function,
        we execute the function and replace the execution buffer with the result."""
        if self.execution_buffer is not None and len(self.execution_buffer) > 0:
            r0 = self.execution_buffer[0]
            if isinstance(r0, NeuralCCGGroundingFunction):
                if len(self.execution_buffer) - 1 == r0.nr_arguments:
                    self.execution_buffer = [r0(*self.execution_buffer[1:])]
                    self.partial_type = [get_type(self.execution_result).typename]
                    self.nr_execution_steps += 1

        return self

    def has_execution_result(self):
        """Check whether the semantics has a fully-executed result."""
        if self.execution_buffer is None:
            return False
        if len(self.execution_buffer) != 1:
            return False
        return True

    @property
    def execution_result(self):
        """Get the fully-executed result. If the result is not fully executed, raise an exception."""
        if self.execution_buffer is None:
            return None
        if len(self.execution_buffer) != 1:
            raise ValueError('The result being queried is not a single output.')
        return self.execution_buffer[0]

    def set_execution_result(self, value):
        """Set the fully-executed result.

        Args:
            value: the value to be set.
        """
        assert self.execution_buffer is not None
        if len(self.execution_buffer) != 1:
            raise ValueError('The result being queried is not a single output.')
        self.execution_buffer[0] = value


class NeuralCCGGroundingFunction(object):
    """A data structure contains the implementation of a function (for neural CCG semantics)."""

    def __init__(
        self,
        expression: Union[Callable, ConstantExpression, Function, FunctionApplicationExpression],
        executor: FunctionDomainExecutor,
        nr_arguments: Optional[int] = None,
        constant_arg_types: Optional[Iterable[ConstantType]] = None,
        bound_constant: Optional[str] = None,
        note: Any = '<anonymous>'
    ):
        """Initialize a neural CCG function.

        Args:
            expression: the function application expression, the function, or a python Callable function representing the semantic form of the function.
            executor: the executor for the domain.
            nr_arguments: the number of arguments to the function.
            constant_arg_types: the types of the constant arguments. Note that constant-typed arguments always come last.
            bound_constant: the constant that is bound to the function. During execution, constant-typed arguments will be filled with this constant.
            note: the note for the function. This is used for debugging.
        """
        self.expression = expression
        self.executor = executor
        if nr_arguments is None:
            if isinstance(expression, (ConstantExpression, FunctionApplicationExpression)):
                self.nr_arguments = 0
            elif isinstance(expression, Function):
                self.nr_arguments = expression.ftype.nr_arguments
            else:
                raise ValueError('The number of arguments must be specified when the expression is a Python callable.')
        else:
            self.nr_arguments = nr_arguments
        self.constant_arg_types = tuple(constant_arg_types) if constant_arg_types is not None else tuple()
        self.bound_constant = bound_constant
        self.note = note

    expression: Union[Callable, Function, FunctionApplicationExpression]
    """The function application expression, the function, or a python Callable function representing the semantic form of the function."""

    executor: FunctionDomainExecutor
    """The executor for the domain."""

    nr_arguments: int
    """The number of arguments to the function."""

    constant_arg_types: Tuple[ConstantType]
    """The types of the constant arguments. Note that constant-typed arguments always come last."""

    bound_constant: Optional[str]
    """The constant that is bound to the function. During execution, constant-typed arguments will be filled with this constant."""

    note: Any
    """The note for the function. This is used for debugging."""

    def bind_constants(self, constant: str) -> 'NeuralCCGGroundingFunction':
        """Bind a constant to the function. This function derives a new function with the same underlying expression.

        Args:
            constant: the constant to be bound.

        Returns:
            the new function with the constant bound.
        """
        return type(self)(
            self.expression, self.executor,
            self.nr_arguments, self.constant_arg_types,
            bound_constant=constant, note=self.note
        )

    def __call__(self, *args: Union[Callable, Value]) -> Value:
        """Execute the function with the given arguments.

        Args:
            *args: the arguments to the function.

        Returns:
            the result of the function.
        """
        args = list(args)
        if len(self.constant_arg_types) > 0:
            assert self.bound_constant is not None
        for arg in self.constant_arg_types:
            args.append(Value(arg, self.bound_constant))

        for i, arg in enumerate(args):
            if isinstance(arg, NeuralCCGGroundingFunction):
                args[i] = arg.expression

        if isinstance(self.expression, Function):
            retval = self.executor.execute_function(self.expression, *args)
        elif isinstance(self.expression, Callable):
            expression = self.expression(*args)
            retval = self.executor.execute(expression)
        else:
            assert len(args) == 0
            retval = self.executor.execute(self.expression)

        if foptions.compose_function_note:
            retval.note = [self.note]
            if len(self.constant_arg_types) > 0:
                retval.note.append('const:' + self.bound_constant)
            for arg in args:
                if hasattr(arg, 'note'):
                    retval.note.append(arg.note)

        return retval


@dataclass
class NeuralCCGSemanticsExecutionBuffer(object):
    """A small data structure that contains the execution buffer, partial type, and number of execution steps.
    This is used in the underlying implementation of the conjunction-typed semantics."""

    execution_buffer: Optional[List[Union[None, Callable, Value]]] = None
    partial_type: Optional[List[str]] = None
    nr_execution_steps: int = 0


class NeuralCCGConjGroundingFunction(object):
    """A wrapper for the underlying implementation of a conjunction-typed semantic form (e.g., AND)."""

    def __call__(
        self,
        left_buffer: List[Union[None, Callable, Value]],
        left_type: List[str],
        right_buffer: List[Union[None, Callable, Value]],
        right_type: List[str],
    ) -> NeuralCCGSemanticsExecutionBuffer:
        """Execute the conjunction-typed semantic form."""
        raise NotImplementedError()


class NeuralCCGSimpleConjFunction(NeuralCCGConjGroundingFunction):
    """A simple implementation for conjunction-typed semantics.

    This function takes a function that works for :class:`~concepts.dsl.value.Value` and automatically
    converts it to a function that works for :class:`NeuralCCGGroundingFunction`.

    The initializer takes a function that takes two arguments and returns a :class:`~concepts.dsl.value.Value`.
    When the conjunction lexicon is combined with two semantic forms which has a value type, this combination function
    will be called to produce the result. When the conjunction lexicon is combined with two semantic forms which has
    a function type, this combination function will be called to produce a new function:

    .. code-block:: python

        def new_function(*args):
            return combination_function(left_function(*args), right_function(*args))
    """

    def __init__(self, executor: FunctionDomainExecutor, primitive_function: Function, note='<anonymous conj>'):
        """Initialize the conjunction-typed semantics.

        Args:
            executor: the executor for the domain.
            primitive_function: the primitive function that is used to implement the conjunction-typed semantics.
            note: the note for the function. This is used for debugging.
        """

        self.executor = executor
        self.primitive_function = primitive_function
        self.note = note

    executor: FunctionDomainExecutor
    """The executor for the domain."""

    primitive_function: Function
    """The primitive function that is used to execute the conjunction-typed semantics at the primitive level."""

    note: Any
    """The note for the function. This is used for debugging."""

    def __call__(
        self,
        left_buffer: List[Union[None, Callable, Value]],
        left_type: List[str],
        right_buffer: List[Union[None, Callable, Value]],
        right_type: List[str],
    ) -> NeuralCCGSemanticsExecutionBuffer:

        lfunc = left_buffer[0]
        rfunc = right_buffer[0]
        if isinstance(lfunc, NeuralCCGGroundingFunction) and isinstance(rfunc, NeuralCCGGroundingFunction):
            nr_left = lfunc.nr_arguments - len(left_buffer) + 1
            nr_right = rfunc.nr_arguments - len(right_buffer) + 1
            assert nr_left == nr_right

            def body(*args):
                assert len(args) == nr_left == nr_right
                lval = lfunc(*left_buffer[1:], *args)
                rval = rfunc(*right_buffer[1:], *args)
                return self.executor.execute_function(self.primitive_function, lval, rval)

            return NeuralCCGSemanticsExecutionBuffer(
                [NeuralCCGGroundingFunction(body, self.executor, nr_left, note=self.note)],
                [self.note],
                0
            )

        if (
            len(left_buffer) == 1 and len(right_buffer) == 1 and
            isinstance(left_buffer[0], (Value, FunctionApplicationExpression)) and
            isinstance(right_buffer[0], (Value, FunctionApplicationExpression))
        ):
            lval, rval = left_buffer[0], right_buffer[0]
            ret = self.executor.execute_function(self.primitive_function, lval, rval)
            return NeuralCCGSemanticsExecutionBuffer([ret], [get_type(ret).typename], 1)

        raise CCGCompositionError('NeuralCCGConjunction failed.')


class NeuralCCGNode(CCGNode):
    """A node in the Neural CCG parsing tree."""

    def __init__(
        self, composition_system: CCGCompositionSystem,
        syntax: NeuralCCGSyntaxType, semantics: NeuralCCGSemantics, composition_type: CCGCompositionType,
        lexicon: Optional[Lexicon] = None,
        lhs: Optional['NeuralCCGNode'] = None, rhs: Optional['NeuralCCGNode'] = None,
        weight: Optional[Union[float, torch.Tensor]] = None,
        composition_str: str = None,
    ):
        """Initialize a CCG node.

        Args:
            composition_system: the composition system.
            syntax: the syntax of the node.
            semantics: the semantics of the node.
            composition_type: the composition type of the node.
            lexicon: the lexicon of the node (only if the composition type is LEXICON).
            lhs: the left child of the node (when available).
            rhs: the right child of the node (when available).
            weight: the weight of the node.
            composition_str: a string representation of the composition, which can be used to remove duplicated trees.
        """
        super().__init__(
            composition_system, syntax, semantics, composition_type,
            lexicon=lexicon, lhs=lhs, rhs=rhs,
            weight=weight
        )

        self.composition_str = composition_str
        if self.composition_str is None:
            self.composition_str = self._build_composition_str()
        self._used_lexicon_entries: Optional[Dict[str, Set[Lexicon]]] = None

    def _build_composition_str(self):
        if self.composition_type is CCGCompositionType.LEXICON:
            if self.is_none:
                return ''
            else:
                return f'{self.lexicon.extra[1]}:{self.lexicon.extra[2]}'

        if isinstance(self.rhs.syntax, CCGCoordinationImmNode):
            return f'({self.lhs.composition_str} {self.rhs.lhs.composition_str}, {self.rhs.rhs.composition_str})'

        if self.lhs.syntax.is_none:
            return self.rhs.composition_str
        if self.rhs.syntax.is_none:
            return self.lhs.composition_str
        return f'({self.lhs.composition_str} {self.rhs.composition_str})'

    composition_system: CCGCompositionSystem
    syntax: Optional[NeuralCCGSyntaxType]
    semantics: Optional[NeuralCCGSemantics]
    composition_type: CCGCompositionType
    lexicon: Optional[Lexicon]
    lhs: Optional['NeuralCCGNode']
    rhs: Optional['NeuralCCGNode']
    weight: Union[float, torch.Tensor]
    composition_str: str
    """A string representation of the composition inside the tree."""

    @property
    def is_none(self):
        """Whether the node is a None node."""
        return self.syntax.is_none

    @property
    def execution_result(self):
        """The execution result of the node."""
        return self.semantics.execution_result

    # NB(Jiayuan Mao @ 04/11) interface for the neural_ccg_soft implementation.
    @property
    def used_lexicon_entries(self) -> Dict[str, Set[Lexicon]]:
        """A set of lexicon entries used in the derivation."""
        return self._used_lexicon_entries

    def set_used_lexicon_entries(self, used_lexicon_entries: Dict[str, Set[Lexicon]]):
        """Set the used lexicon entries."""
        assert self._used_lexicon_entries is None
        self._used_lexicon_entries = used_lexicon_entries

    def compose_check(self, rhs: 'NeuralCCGNode', composition_type: CCGCompositionType):
        super().compose_check(rhs, composition_type)
        if composition_type is CCGCompositionType.FORWARD_APPLICATION:
            if self.syntax.is_function and isinstance(self.syntax.argument_types[-1], FunctionType):
                if rhs.composition_type is not CCGCompositionType.LEXICON:
                    raise CCGCompositionError('Does not support combination with non-lexical functor type.')
        elif composition_type is CCGCompositionType.BACKWARD_APPLICATION:
            if rhs.syntax.is_function and isinstance(rhs.syntax.argument_types[-1], FunctionType):
                if self.composition_type is not CCGCompositionType.LEXICON:
                    raise CCGCompositionError('Does not support combination with non-lexical functor type.')

    def compose_guess(self, rhs: 'NeuralCCGNode') -> Tuple[CCGCompositionType]:
        if isinstance(self.syntax, CCGCoordinationImmNode):
            return tuple()
        if isinstance(rhs.syntax, CCGCoordinationImmNode):
            return (CCGCompositionType.COORDINATION, )
        if self.syntax.is_conj:
            return (CCGCompositionType.COORDINATION, )
        if rhs.syntax.is_conj:
            return tuple()

        if self.syntax.is_none or rhs.syntax.is_none:
            return (CCGCompositionType.NONE, )
        if self.syntax.is_function:
            if self.syntax.argument_types[-1].typename == rhs.syntax.typename and self.syntax.linearization[-1].direction is CCGCompositionDirection.RIGHT:
                return (CCGCompositionType.FORWARD_APPLICATION, )
        if rhs.syntax.is_function:
            if rhs.syntax.argument_types[-1].typename == self.syntax.typename and rhs.syntax.linearization[-1].direction is CCGCompositionDirection.LEFT:
                return (CCGCompositionType.BACKWARD_APPLICATION, )

        return tuple()


def compose_neural_ccg_node(lhs: NeuralCCGNode, rhs: NeuralCCGNode, composition_type: Optional[CCGCompositionType] = None) -> Union[NeuralCCGNode, CCGCompositionResult]:
    """Compose two neural CCG nodes.

    Args:
        lhs: the left node.
        rhs: the right node.

    Returns:
        The composed node.
    """
    return lhs.compose(rhs, composition_type=composition_type)


class NeuralCCG(nn.Module):
    """The neural CCG grammar and the implementation for parsing."""

    training: bool

    def __init__(
        self,
        domain: FunctionDomain,
        executor: FunctionDomainExecutor,
        candidate_lexicon_entries: Iterable['NeuralCCGLexiconSearchResult'],
        composition_system: Optional[CCGCompositionSystem] = None,
        joint_execution: bool = True,  # execute the partial programs during CKY.
        allow_none_lexicon: bool = False,
        reweight_meaning_lex: bool = False
    ):
        """Initialize the neural CCG grammar.

        Args:
            domain: the function domain of the grammar.
            executor: the executor for the function domain.
            candidate_lexicon_entries:  a list of candidate lexicon entries.
            composition_system: the composition system. If None, the default composition system will be used.
            joint_execution: whether to execute the partial programs during CKY.
            allow_none_lexicon: whether to allow None lexicon.
            reweight_meaning_lex: whether to reweight the meaning lexicon entries. Specifically, if there are two parsings
                share the same set of lexicon entries (i.e., caused by ambiguities in combination), specifying this flag
                will reweight both of them to be 1 / (number of parsings that used this set of lexicon entries).
        """
        super().__init__()

        if composition_system is None:
            composition_system = CCGCompositionSystem.make_default()

        self.domain = domain
        self.executor = executor
        self.composition_system = composition_system
        self.candidate_lexicon_entries = tuple(candidate_lexicon_entries)
        self.joint_execution = joint_execution
        self.allow_none_lexicon = allow_none_lexicon
        self.reweight_meaning_lex = reweight_meaning_lex

    domain: FunctionDomain
    """The function domain."""

    executor: FunctionDomainExecutor
    """The executor for the domain."""

    composition_system: CCGCompositionSystem
    """The composition system."""

    candidate_lexicon_entries: Tuple['NeuralCCGLexiconSearchResult', ...]
    """The candidate lexicon entries."""

    joint_execution: bool
    """Whether to execute the partial programs during CKY."""

    allow_none_lexicon: bool
    """Whether to allow None lexicon."""

    reweight_meaning_lex: bool
    """Whether to reweight the meaning lexicon entries."""

    @property
    def nr_candidate_lexicon_entries(self):
        return len(self.candidate_lexicon_entries)

    @_profile
    def parse(
        self,
        words: Union[Sequence[str], str],
        distribution_over_lexicon_entries: torch.Tensor,
        used_lexicon_entries: Optional[Dict[str, Set[int]]] = None,
        acceptable_rtypes: Optional[Sequence[TypeBase]] = None,
        max_research: int = 0
    ) -> List[NeuralCCGNode]:
        """Parse the sentence using the CKY algorithm. The function maintains a "chart" for all possible spans of the
        sentence, and for each span, it maintains a list of possible CCG nodes. When ``max_research`` is set to a positive
        integer, for each span, the function will keep a tuple of lists of CCG nodes, where ``dp[i][j][k]`` corresponds to
        candidate CCG nodes for span ``[i, j]`` with ``k`` words being re-searched.

        Args:
            words: the words of the sentence. If the input is a string, it will be tokenized using the default ``.split()`` method.
            distribution_over_lexicon_entries: the distribution over the lexicon entries for each word.
            used_lexicon_entries: the used lexicon entries for each word. If None, it will be computed from the distribution.
                This is a dictionary mapping from the word to the set of lexicon entry indices.
            acceptable_rtypes: the acceptable return types. If None, it will accept all return types.
            max_research: the maximum number of words whose lexicon entries can be re-searched (i.e., the ``used_lexicon_entries`` for this word will be ignored).

        Returns:
            The parsing result.
        """

        if isinstance(words, str):
            words = words.split()

        if not self.training:
            max_research = 0

        length = distribution_over_lexicon_entries.size(0)
        assert length == len(words)

        # Initialize the chart.
        if max_research == 0:
            dp = [[list() for _ in range(length + 1)] for _ in range(length)]
        else:
            dp = [[tuple([list() for _ in range(max_research + 1)] for _ in range(length + 1)) for _ in range(length)]]

        if foptions.debug_print:
            print('Parsing: ', words)
            print('-' * 120)

        use_lazy_semantics = False
        if self.training:
            use_lazy_semantics = foptions.training_use_lazy_semantics

        syntax_only = False
        if get_ccg_composition_context().semantics is False:
            syntax_only = True

        for i in range(length):
            dp[i][i + 1] = self._gen_lexicons(words, i, distribution_over_lexicon_entries, used_lexicon_entries, max_research=max_research)

            if self.training:
                if max_research == 0:
                    dp[i][i + 1] = self._unique_lexicon(dp[i][i + 1], syntax_only=syntax_only)
                else:
                    dp[i][i + 1] = tuple(self._unique_lexicon(x, syntax_only=syntax_only) for x in dp[i][i + 1])

            # NB(Jiayuan Mao @ 03/01): Remove "select max" at the lexical level to implement SFINAE.
            # if not self.training:
            #     dp[i][i+1] = [max(dp[i][i+1], key=lambda node: node.weight.item())] if len(dp[i][i+1]) > 0 else []
            # dp[i][i+1] = [max(dp[i][i+1], key=lambda node: node.weight.item())] if len(dp[i][i+1]) > 0 else []

            # Block :: Debug Print {{{
            if foptions.debug_print:
                print(f'Span: [{i}, {i+1})', words[i])
                with FormatContext(expr_max_length=-1).as_default():
                    for node in sorted(_maybe_flatten_list(dp[i][i + 1]), key=lambda x: x.syntax):
                        print(jacinle.indent_text(
                            f'Weight: {node.weight.item():.4f}; Semantics: {str(node.expression)}; Syntax: {str(node.syntax)}'
                        ))
            # }}} End Block :: Debug Print

        with CCGCompositionContext(semantics_lazy_composition=use_lazy_semantics, exc_verbose=False).as_default():
            for phrase_length in range(2, length + 1):
                for i in range(0, length + 1 - phrase_length):
                    j = i + phrase_length
                    for k in range(i + 1, j):
                        if max_research == 0:
                            dp[i][j].extend(self._merge(dp[i][k], dp[k][j]))
                        else:
                            for mri in range(max_research + 1):
                                for mrj in range(max_research + 1):
                                    if mri + mrj <= max_research:
                                        dp[i][j][mri + mrj].extend(self._merge(dp[i][k][mri], dp[k][j][mrj]))

                    # Block :: Debug Print {{{
                    if foptions.debug_print:
                        print(f'Span: [{i}, {j}) :: Before Unique', words[i:j])
                        with FormatContext(expr_max_length=-1).as_default():
                            for node in sorted(_maybe_flatten_list(dp[i][j]), key=lambda x: x.syntax):
                                print(jacinle.indent_text(
                                    f'Weight: {node.weight.item():.4f}; Semantics: {str(node.expression)}; Syntax: {str(node.syntax)}'
                                ))
                    # }}} End Block :: Debug Print

                    if not self.training:
                        assert max_research == 0
                        dp[i][j] = [max(dp[i][j], key=lambda node: node.weight.item())] if len(dp[i][j]) > 0 else []
                    else:
                        if max_research == 0:
                            dp[i][j] = self._unique(dp[i][j], syntax_only=syntax_only)
                        else:
                            dp[i][j] = tuple(self._unique(x, syntax_only=syntax_only) for x in dp[i][j])

                    # Block :: Debug Print {{{
                    if foptions.debug_print:
                        print(f'Span: [{i}, {j}) :: After Unique', words[i:j])
                        with FormatContext(expr_max_length=-1).as_default():
                            for node in sorted(_maybe_flatten_list(dp[i][j]), key=lambda x: x.syntax):
                                print(jacinle.indent_text(
                                    f'Weight: {node.weight.item():.4f}; Semantics: {str(node.expression)}; Syntax: {str(node.syntax)}'
                                ))
                    # }}} End Block :: Debug Print

                    if foptions.debug_stat_types:
                        count = defaultdict(int)
                        for node in _maybe_flatten_list(dp[i][j]):
                            count[str(node.syntax)] += 1
                        print(f'Span: [{i}, {j}) ::', count)

        ret = _maybe_flatten_list(dp[0][length])
        ret = self._select_acceptable_parsings(ret, acceptable_rtypes)
        if self.reweight_meaning_lex:
            ret = self._reweight_parse_trees(ret)
        ret = sorted(ret, key=lambda node: node.weight.item(), reverse=True)

        if foptions.debug_print:
            input('Press enter to continue...')
            import ipdb; ipdb.set_trace()

        return ret

    @_profile
    def parse_beamsearch(
        self,
        words: Union[Sequence[str], str],
        distribution_over_lexicon_entries: torch.Tensor,
        used_lexicon_entries: Optional[Dict[str, Set[int]]] = None,
        acceptable_rtypes: Optional[List[TypeBase]] = None,
        beam: int = 5, lexical_beam: int = 0,
        out_of_vocab_weight: Optional[float] = None
    ) -> List[NeuralCCGNode]:
        """Parse the sentence with the CKY algorithm, but with beam search. Note that this function does not support re-search for used lexicon entries,
        and it can only be used when the model is in ``eval`` mode, because the beam-search will break the correctness of the gradient computation.

        Args:
            words: the words of the sentence. If the input is a string, it will be tokenized using the default ``.split()`` method.
            distribution_over_lexicon_entries: the distribution over the lexicon entries for each word.
            used_lexicon_entries: the used lexicon entries for each word. If None, it will be computed from the distribution.
                This is a dictionary mapping from the word to the set of lexicon entry indices.
            acceptable_rtypes: the acceptable return types. If None, it will accept all return types.
            beam: the beam size.
            lexical_beam: the lexical beam size. If 0, we will keep all the lexical entries.
            out_of_vocab_weight: the weight for out-of-vocabulary words. Specifically, if this value is set, we will consider all lexical entries
                (instead of only those in the ``used_lexicon_entries``). However, the corresponding weights for those entries will be set to
                ``out_of_vocab_weight``.

        Returns:
            The parsing result, as a list of parsing trees.
        """
        assert not self.training

        length = distribution_over_lexicon_entries.size(0)
        dp = [[list() for _ in range(length + 1)] for _ in range(length)]

        if lexical_beam is None:
            lexical_beam = beam

        for i in range(length):
            dp[i][i + 1] = self._gen_lexicons(words, i, distribution_over_lexicon_entries, used_lexicon_entries, out_of_vocab_weight=out_of_vocab_weight)
            if lexical_beam > 0:
                dp[i][i + 1] = heapq.nlargest(lexical_beam, dp[i][i + 1], key=lambda node: node.weight.item()) if len(dp[i][i + 1]) > 0 else list()

        with CCGCompositionContext(semantics_lf=False, exc_verbose=False).as_default():
            for span_length in range(2, length + 1):
                for i in range(0, length + 1 - span_length):
                    j = i + span_length
                    for k in range(i + 1, j):
                        dp[i][j].extend(self._merge(dp[i][k], dp[k][j]))

                    if span_length != length:
                        if len(dp[i][j]) > beam:
                            dp[i][j] = heapq.nlargest(beam, dp[i][j], key=lambda node: node.weight.item()) if len(dp[i][j]) > 0 else []

        ret = self._select_acceptable_parsings(dp[0][length], acceptable_rtypes)
        ret = sorted(ret, key=lambda node: node.weight.item(), reverse=True)
        return ret

    def _gen_lexicons(
        self,
        words: Sequence[str],
        word_index: int,
        distribution_over_lexicon_entries: torch.Tensor,
        used_lexicon_entries: Optional[Dict[str, Set[int]]] = None,
        out_of_vocab_weight: Optional[float] = None,
        max_research: int = 0
    ) -> Union[List[NeuralCCGNode], Tuple[List[NeuralCCGNode], ...]]:
        """Generate the lexicon nodes for the given word.

        Args:
            words: the words of the sentence.
            word_index: the index of the word.
            distribution_over_lexicon_entries: the distribution over the lexicon entries for each word.
            used_lexicon_entries: the used lexicon entries for each word. It should be a dictionary mapping from the word to the set of lexicon entry indices.
            out_of_vocab_weight: the weight for out-of-vocab words. This will only take effect in the evaluation mode. See :meth:`parse_beamsearch` for more details.
            max_research: the maximum number of re-search for used lexicon entries. See :meth:`parse` for more details.

        Returns:
            When ``max_research`` is 0, it will return a list of lexicon nodes. Otherwise, it will return a tuple of lists of lexicon nodes.
        """
        word = words[word_index]
        use_used_lexicon_record = used_lexicon_entries is not None and word in used_lexicon_entries

        # When ``out_of_vocab_weight`` is None, we will still consider using all candidate lexicon entries, but
        # for candidates that are not in the ``used_lexicon_entries`` record, we will set their weight to be
        # ``out_of_vocab_weight``.
        if not self.training and out_of_vocab_weight is not None:
            use_used_lexicon_record = False

        # When ``max_research`` is not 0, we will still consider using all candidate lexicon entries.
        if self.training and max_research != 0:
            use_used_lexicon_record = False

        if use_used_lexicon_record:
            lexicon_entry_indices = used_lexicon_entries[word]
        else:
            lexicon_entry_indices = range(distribution_over_lexicon_entries.size(1))

        if max_research == 0:
            lexicon_entries = list()
        else:
            lexicon_entries = tuple(list() for _ in range(max_research + 1))

        ctx = get_ccg_composition_context()

        for j in lexicon_entry_indices:
            candidate = self.candidate_lexicon_entries[j]
            syn, sem, exe = candidate.syntax, candidate.semantics, candidate.executor

            if ctx.semantics:
                is_conj = syn.is_conj
                sem = self._bind_constants(sem, exe, word, word_index=word_index, candidate_lexicon_entry_index=j, is_conj=is_conj)
                sem = sem.canonize_execution_buffer()
            else:
                sem = None

            weight = distribution_over_lexicon_entries[word_index, j]
            if not self.training and out_of_vocab_weight is not None:
                if used_lexicon_entries is not None and word in used_lexicon_entries:
                    if j not in used_lexicon_entries[word]:
                        weight = out_of_vocab_weight

            lexicon = Lexicon(syn, sem, weight, extra=(words[word_index], word_index, j))
            node = NeuralCCGNode(
                self.composition_system,
                lexicon.syntax, lexicon.semantics, CCGCompositionType.LEXICON, lexicon=lexicon
            )

            if max_research == 0:
                lexicon_entries.append(node)
            else:
                research_index = 0
                if used_lexicon_entries is not None and words[word_index] in used_lexicon_entries:
                    if j not in used_lexicon_entries[words[word_index]]:
                        research_index = 1
                lexicon_entries[research_index].append(node)

        return lexicon_entries

    def _bind_constants(
        self,
        expression: Union[None, Callable, ConstantExpression, Function, FunctionApplicationExpression],
        compiled_executor: Optional[Callable],
        word: str,
        word_index: int,
        candidate_lexicon_entry_index: int,
        is_conj: bool
    ) -> NeuralCCGSemantics:
        """Bind a constant to a semantic form.

        Args:
            expression: the semantic form.
            compiled_executor: the compiled executor. When ``self.joint_execution`` is set to False, this argument will be None.
            word: the word.
            word_index: the word index in the sentence. This will only be used in compiling the partial type.
            candidate_lexicon_entry_index: the index of the candidate lexicon entry. This will only be used in compiling the partial type.
            is_conj: whether the semantic form is a conjunction.

        Returns:
            The semantic form with the constant bound.
        """
        if expression is None:
            return NeuralCCGSemantics(None)

        new_expression = expression
        new_buffer = compiled_executor
        new_partial_type = None
        nr_execution_steps = 0

        # NB(Jiayuan Mao @ 2022/12/11): this function does need to handle "execution" because a ``canonize_execution_buffer`` function
        # will be called in ``_gen_lexicons`` immediately after this function.
        if isinstance(expression, (ConstantExpression, FunctionApplicationExpression)):
            pass
        elif isinstance(expression, Function):
            bindings = dict()
            for denotation, arg in enumerate(expression.ftype.argument_types):
                if isinstance(arg, ConstantType):
                    bindings['#' + str(denotation)] = Value(arg, word)

            if len(bindings) == 0:
                new_partial_type = [NeuralCCGSemanticsPartialTypeLex(candidate_lexicon_entry_index, None)]
            else:
                new_expression = expression.partial(**bindings, execute_fully_bound_functions=True)
                new_buffer = compiled_executor.bind_constants(word) if compiled_executor is not None else None
                new_partial_type = [NeuralCCGSemanticsPartialTypeLex(candidate_lexicon_entry_index, new_partial_type)]
        else:
            new_partial_type = [NeuralCCGSemanticsPartialTypeLex(candidate_lexicon_entry_index, None)]

        return NeuralCCGSemantics(
            new_expression,
            execution_buffer=[new_buffer],
            partial_type=new_partial_type,
            nr_execution_steps=nr_execution_steps,
            is_conj=is_conj
        )

    def _merge(self, list1, list2):
        """The merge function for chart parsing."""
        # TODO(Jiayuan Mao @ 2020/01/11): implement the merge sort.
        output_list = list()
        for node1, node2 in itertools.product(list1, list2):
            try:
                node = compose_neural_ccg_node(node1, node2).result
                output_list.append(node)
            except CCGCompositionError:
                pass

        return output_list

    def _unique_lexicon(self, nodes: List[NeuralCCGNode], syntax_only: bool) -> List[NeuralCCGNode]:
        """Unique the lexicon entries during the chart parsing."""
        if syntax_only:
            return self._unique_syntax_only(nodes)
        return nodes

    def _unique(self, nodes: List[NeuralCCGNode], syntax_only: bool) -> List[NeuralCCGNode]:
        """Unique the nodes for intermediate results during the chart parsing. This will only be triggered when the system allows None lexicon entriese, in which
        case different set of leaf lexicon entries will lead to the same parsing tree."""
        # NB(Jiayuan Mao @ 2020/02/10): perform a unique() operation.
        # We do not need to sum up the weights here, since the duplicates occur because of the empty semantic forms.
        if syntax_only:
            return self._unique_syntax_only(nodes)

        if self.allow_none_lexicon:
            nodes_by_composition_string = defaultdict(list)
            for v in nodes:
                nodes_by_composition_string[v.composition_str].append(v)
            nodes = [v[0] for v in nodes_by_composition_string.values()]
        return nodes

    def _unique_syntax_only(self, nodes: List[NeuralCCGNode]) -> List[NeuralCCGNode]:
        """Perform unique operation based on the syntax only. When there are multiple nodes with the same syntax, we will only keep the semantic forms for the one with the highest weight."""
        output_nodes = list()

        # a dictionary mapping from syntax types to (semantics, weight) pairs.
        nodes_by_syntax = defaultdict(lambda: (list(), list()))
        for node in nodes:
            rec = nodes_by_syntax[str(node.syntax)]
            rec[0].append(node)
            rec[1].append(node.weight)

        for typename, (this_nodes, this_weights) in nodes_by_syntax.items():
            if len(this_nodes) > 1:
                sample_node = this_nodes[0]
                weights_tensor = torch.stack(this_weights, dim=0)
                total_weights = jactorch.logsumexp(weights_tensor)
                output_nodes.append(NeuralCCGNode(
                    composition_system=sample_node.composition_system,
                    syntax=sample_node.syntax,
                    semantics=sample_node.expression,
                    composition_type=sample_node.composition_type,
                    lexicon=sample_node.lexicon, lhs=sample_node.lhs, rhs=sample_node.rhs,
                    composition_str=sample_node.composition_str,
                    weight=total_weights
                ))
                output_nodes[-1].set_used_lexicon_entries(gen_used_lexicon_entries(this_nodes))
            else:
                output_nodes.append(this_nodes[0])
        return output_nodes

    def _select_acceptable_parsings(self, parsings: List[NeuralCCGNode], acceptable_rtypes: Sequence[TypeBase]) -> List[NeuralCCGNode]:
        """Select a subset of the parsing trees that are whose final types are in the acceptable_rtypes."""
        if acceptable_rtypes is None:
            return [node for node in parsings if not node.syntax.is_none]
        parsings_filtered = list()
        for node in parsings:
            if not isinstance(node.syntax, CCGCoordinationImmNode) and not node.syntax.is_none and node.syntax.is_value and node.syntax.return_type in acceptable_rtypes:
                parsings_filtered.append(node)
        return parsings_filtered

    def _reweight_parse_trees(self, parsings: List[NeuralCCGNode]) -> List[NeuralCCGNode]:
        """Reweight the parse trees based on lexicon entries. Specifically, if there are multiple parsing trees with the same leaf-level lexicon
        entries, their weight will be divided by the number of such trees."""
        lexicon_strs = list()
        for p in parsings:
            lexicon_strs.append(p.composition_str.replace('(', '').replace(')', ''))
        counter = Counter(lexicon_strs)
        for p, ls in zip(parsings, lexicon_strs):
            if counter[ls] > 1:
                # p.weight = p.weight * 1/(counter[ls])
                p.weight = p.weight - math.log(counter[ls])
        return parsings

    def format_lexicon_table_sentence(self, words, distribution_over_lexicon_entries, max_entries=None):
        return self.format_lexicons_table(None, distribution_over_lexicon_entries_words=distribution_over_lexicon_entries, words=words, max_entries=max_entries)

    def format_lexicons_table(
        self,
        vocab,
        distribution_over_lexicon_entries=None,
        distribution_over_lexicon_entries_words=None,
        *,
        used_lexicon_weights=None, words=None, full=False,
        max_entries=None, print_grad=True
    ):
        if words is None:
            if full or used_lexicon_weights is None:
                words = vocab
            else:
                words = used_lexicon_weights.keys()

        table = list()

        if distribution_over_lexicon_entries is not None:
            has_grad = print_grad and distribution_over_lexicon_entries.grad is not None
        else:
            assert distribution_over_lexicon_entries_words is not None
            has_grad = print_grad and distribution_over_lexicon_entries_words.grad is not None

        # if print_grad and not has_grad:
        #     logger.warning('print_grad is specified for lexicon table but gradient information is not available.')

        for word_idx, word in enumerate(words):
            if word[0] == '<' and word[-1] == '>':
                continue

            if used_lexicon_weights is not None and word in used_lexicon_weights:
                lexicon_indices = list(used_lexicon_weights[word])
            else:
                lexicon_indices = range(self.nr_candidate_lexicon_entries)

            if distribution_over_lexicon_entries is not None:
                lexicon_weights = F.softmax(torch.tensor([
                    distribution_over_lexicon_entries[vocab.map(word), i].item()
                    for i in lexicon_indices
                ]), dim=-1)

                # TODO(Jiayuan Mao @ 02/10): take a softmax for visualization?
                all_lexicons = [(
                    i,
                    self.candidate_lexicon_entries[i],
                    lexicon_weights[j],
                    distribution_over_lexicon_entries.grad[vocab.map(word), i].item() if has_grad else None
                ) for j, i in enumerate(lexicon_indices)]
            else:
                lexicon_weights = F.softmax(torch.tensor([
                    distribution_over_lexicon_entries_words[word_idx, i].item()
                    for i in lexicon_indices
                ]), dim=-1)

                # TODO(Jiayuan Mao @ 02/10): take a softmax for visualization?
                all_lexicons = [(
                    i,
                    self.candidate_lexicon_entries[i],
                    lexicon_weights[j],
                    distribution_over_lexicon_entries_words.grad[word_idx, i].item() if has_grad else None
                ) for j, i in enumerate(lexicon_indices)]

            all_lexicons.sort(key=lambda x: x[2], reverse=True)
            if max_entries is not None:
                all_lexicons = all_lexicons[:max_entries]

            first = word
            for idx, lexicon, score, grad in all_lexicons:
                if print_grad:
                    table.append((first, idx, '{:3.4f}'.format(score), ('g: {:3.4f}'.format(grad) if grad is not None else 'g: None'), str(lexicon.syntax), str(lexicon.semantics)))
                else:
                    table.append((first, idx, '{:3.4f}'.format(score), str(lexicon.syntax), str(lexicon.semantics)))

                first = ''

        if print_grad:
            return tabulate(table, headers=['', 'i', 'weight', 'w_grad', 'syntax', 'semantics'])
        else:
            return tabulate(table, headers=['', 'i', 'weight', 'syntax', 'semantics'])


def _merge_list(*lists: Optional[List[T]]) -> List[T]:
    output = list()
    for x in lists:
        if x is not None:
            output.extend(x)
    return output


def _maybe_flatten_list(tuple_of_list: Union[List[T], Tuple[List[T], ...]]) -> List[T]:
    if isinstance(tuple_of_list, list):
        return tuple_of_list
    output = list()
    for x in tuple_of_list:
        output.extend(x)
    return output


def gen_used_lexicon_entries(parsings: Iterable[NeuralCCGNode]) -> Dict[str, Set[int]]:
    """Generate a dictionary of used lexicon entries from a list of parsing results. The result is a mapping from words to lexicon entry indices.

    Args:
        parsings: a list of parsing results.

    Returns:
        A dictionary of used lexicon entries.
    """
    used_lexicon_entries = defaultdict(set)

    def dfs(ccg_node: NeuralCCGNode):
        nonlocal used_lexicon_entries

        # NB(Jiayuan Mao @ 04/11): handles compressed nodes generated by neural soft ccg.
        if ccg_node.used_lexicon_entries is not None:
            for k, v in ccg_node.used_lexicon_entries.items():
                used_lexicon_entries[k].update(v)
        else:
            if ccg_node.composition_type is CCGCompositionType.LEXICON:
                word, _, index = ccg_node.lexicon.extra
                used_lexicon_entries[word].add(index)
            else:
                dfs(ccg_node.lhs)
                dfs(ccg_node.rhs)

    for p in parsings:
        dfs(p)
    return used_lexicon_entries


def count_nr_execution_steps(parsings: Iterable[NeuralCCGNode]) -> int:
    """Count the number of execution steps from a list of parsing results."""
    answer = 0

    for p in parsings:
        answer += p.expression.nr_execution_steps
    return answer

