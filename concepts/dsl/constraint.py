#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : constraint.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/10/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Data structures for Constraint Satisification Problems (CSPs).

In general, a CSP can be represented as a tuple of variables, domains (for each variable), and constraints.
Here, we only consider constraints of the following form: ``function(*arguments) == rv``. Internally:

- Each variable can either be a determined :class:`~concepts.dsl.tensor_value.TensorValue` or an :class:`OptimisticValue` (i.e., undetermined value).
    For :class:`TensorValue`, the representation contains its data type and the actual value.
    For :class:`OptimisticValue`, the representation contains its data type and a single integer, representing the index of the variable.
- Each constraint is represented as a :class:`Constraint` object, which contains the function, arguments, and the return value.
    The function is represented as a string (only for equality constraints), or BoolOpType (and, or, not), or QuantificationOpType (forall, exists),
        or :class:`~concepts.dsl.dsl_functions.Function` objects. Both arguments and return values can be :class:`DeterminedValue` or :class:`OptimisticValue`.
- The index of optimistic values starts from -2147483647 (i.e., ``OPTIM_MAGIC_NUMBER``). Therefore, they can be directly stored in tensors without any conversion,
    which is helpful for implementing CSP-based expression executors. There are a few helper functions to convert between optimistic values and their indices.
"""

from enum import IntEnum
from dataclasses import dataclass
from typing import Any, Optional, Union, Sequence, Tuple, List, Set, Dict

import torch
import jacinle
from jacinle.utils.enum import JacEnum

from concepts.dsl.dsl_types import BOOL, INT64, TensorValueTypeBase, NamedTensorValueType, PyObjValueType, Variable
from concepts.dsl.dsl_functions import Function
from concepts.dsl.tensor_value import TensorValue
from concepts.dsl.expression import BoolOpType, QuantificationOpType

__all__ = [
    'OPTIM_MAGIC_NUMBER', 'OPTIM_MAGIC_NUMBER_UPPER', 'OPTIM_MAGIC_NUMBER_MAGIC',
    'is_optimistic_value', 'optimistic_value_id', 'is_optimistic_value',
    'OptimisticValue', 'cvt_opt_value', 'SimulationFluentConstraintFunction',
    'Constraint', 'EqualityConstraint', 'ConstraintSatisfactionProblem',
    'AssignmentType', 'Assignment', 'AssignmentDict', 'print_assignment_dict', 'ground_assignment_value'
]


OPTIM_MAGIC_NUMBER = -2147483647
OPTIM_MAGIC_NUMBER_UPPER = (-2147483648) / 2
OPTIM_MAGIC_NUMBER_MAGIC = -2147483648


def is_optimistic_value(v: Union[int, torch.Tensor]):
    """Check if a value is an optimistic identifier."""
    return v < OPTIM_MAGIC_NUMBER_UPPER


def optimistic_value_id(v: Union[int, 'OptimisticValue']):
    """Get the optimistic identifier from an optimistic value."""
    if isinstance(v, OptimisticValue):
        return v.identifier - OPTIM_MAGIC_NUMBER
    return v - OPTIM_MAGIC_NUMBER


def maybe_optimistic_string(v: int):
    """If `v` is an optimistic value, return a string representation of the optimistic identifier, otherwise return `str(v)`."""
    if is_optimistic_value(v):
        return '@' + str(optimistic_value_id(v))
    return str(v)


class OptimisticValue(object):
    """An optimistic value object holds a pair of data type and an optimistic identifier."""

    def __init__(self, dtype: TensorValueTypeBase, identifier: int):
        """Initialize the OptimisticValue object.

        Args:
            dtype: the type of the value.
            identifier: the optimistic identifier.
        """

        if not isinstance(dtype, (NamedTensorValueType, PyObjValueType)) and dtype != BOOL and dtype != INT64:
            raise TypeError('OptimisticValue only supports NamedTensorValueType, PyObjValueType, BOOL, and INT64.')

        self.dtype = dtype
        self.identifier = identifier
        self.__post_init__()

    dtype: TensorValueTypeBase
    """The dtype of the optimistic value."""

    identifier: int
    """The optimistic identifier."""

    # TODO(Jiayuan Mao @ 2023/08/16): remove this check.
    def __post_init__(self):
        assert isinstance(self.identifier, int)

    def __str__(self) -> str:
        return f'O[{self.dtype}]{{@{optimistic_value_id(self.identifier)}}}'

    def __repr__(self) -> str:
        return self.__str__()


def cvt_opt_value(value: Union[OptimisticValue, TensorValue, bool, int], dtype: Optional[TensorValueTypeBase] = None) -> Union[TensorValue, OptimisticValue]:
    """Convert a value to OptimisticValue or a (determined) TensorValue. Acceptable types are:

    - OptimisticValue
    - TensorValue, which will be converted to OptimisticValue if it corresponds to an optimistic value.
    - bool, which will be converted to TensorValue.
    - int, which will be converted to OptimisticValue if it is an optimistic identifier, otherwise TensorValue.

    Args:
        value: the value to be converted.
        dtype: the data type of the value. If not specified (and the input is a TensorValue), the data type of the value will be used.

    Returns:
        The converted value.
    """

    if isinstance(value, OptimisticValue):
        return value
    elif isinstance(value, TensorValue):
        assert dtype is None or value.dtype == dtype
        v = value.single_elem()
        if isinstance(v, OptimisticValue):
            return v
        return value
    elif isinstance(value, bool):
        assert dtype is None or dtype == BOOL
        return TensorValue.from_scalar(value, BOOL)
    elif isinstance(value, int):
        if is_optimistic_value(value):
            assert dtype is not None, 'dtype must be specified for optimistic value'
            return OptimisticValue(dtype, value)
        if dtype == BOOL:
            raise RuntimeError('Should not use int to represent bool. This could be a bug in the code; report to the developers.')
            # return TensorValue.from_scalar(bool(value), BOOL)
        assert dtype == INT64
        return TensorValue.from_scalar(value, INT64)
    else:
        raise TypeError('Unknown value type: {} (type={}).'.format(value, type(value)))


@dataclass
class SimulationFluentConstraintFunction(object):
    """SimulationConstraint is a special kind of constraint that asserts the return value of the function is the grounding
    of a predicate after executing action ``action_index`` in the simulation. This is a special kind of constraint that
    has to be evaluated in a simulation environment. Therefore it is listed separately from the other domain-general constraints.
    """

    action_index: int
    """The index of the action that is executed in the simulation."""

    predicate: Function
    """The predicate to be grounded."""

    arguments: Tuple[int, ...]
    """The arguments to the predicate."""

    @property
    def name(self) -> str:
        return f'SimulationConstraint({self.action_index}, {self.predicate}, {self.arguments})'


class Constraint(object):
    """A constraint is basically a equality expression: ``function(*arguments) == rv``."""

    EQUAL = '__EQ__'
    """Magic name for equality constraints."""

    def __init__(
        self, function: Union[str, BoolOpType, QuantificationOpType, Function, SimulationFluentConstraintFunction],
        arguments: Sequence[Union[TensorValue, OptimisticValue]],
        rv: Union[TensorValue, OptimisticValue],
        note: Any = None
    ):
        """Initialize a constraint. Each constraint take the form of:

        .. code-block:: python

            function(*arguments) == rv

        Args:
            function: the function name, or a BoolOpType or QuantifierType, or a Function object.
            arguments: the arguments of the function.
            rv: the expected return value of the function.
            note: an optional note for the constraint.
        """

        self.function = function
        self.arguments = tuple(map(cvt_opt_value, arguments))
        self.rv = cvt_opt_value(rv)
        self.note = note

    function: Union[str, BoolOpType, QuantificationOpType, Function, SimulationFluentConstraintFunction]
    """The function identifier: either a string (currently only for equality constraints), or a BoolOpType or QuantificationOpType (for Boolean expressions), or a Function object."""

    arguments: Tuple[Union[TensorValue, OptimisticValue], ...]
    """The arguments to the function."""

    rv: Union[TensorValue, OptimisticValue]
    """The expected return value of the function."""

    note: Any
    """An optional note for the constraint."""

    def constraint_str(self):
        """Return the string representation of the constraint."""
        argument_strings = [x.format(short=True) if isinstance(x, TensorValue) else str(x) for x in self.arguments]
        if self.is_equal_constraint and isinstance(self.rv, TensorValue):
            if self.rv.item():
                return '__EQ__(' + ', '.join(argument_strings) + ')'
            else:
                return '__NEQ__(' + ', '.join(argument_strings) + ')'
        else:
            if isinstance(self.function, (str, JacEnum)):
                name = str(self.function)
            else:
                name = self.function.name
            return name + '(' + ', '.join(argument_strings) + ') == ' + str(self.rv)

    def __str__(self):
        ret = self.constraint_str()
        if self.note is not None:
            ret += '  # ' + str(self.note)
        return ret

    __repr__ = jacinle.repr_from_str

    @property
    def is_equal_constraint(self) -> bool:
        """Check if the constraint is an equality constraint."""
        return isinstance(self.function, str) and self.function == Constraint.EQUAL

    @classmethod
    def from_function(cls, function: Function, args: Sequence[Union[bool, int, torch.Tensor, TensorValue, Any]], rv: Union[bool, int, torch.Tensor, TensorValue, Any]) -> 'Constraint':
        """Create a constraint given a function, arguments, and return value.

        Args:
            function: the function object.
            args: the arguments of the function. The arguments can be bool, int, or torch.Tensor.
            rv: the return value of the function. The return value can be bool, int, or torch.Tensor.

        Returns:
            The created constraint.
        """
        _cvt = cvt_opt_value
        args = [_cvt(x, var.dtype if isinstance(var, Variable) else var) for x, var in zip(args, function.arguments)]
        rv = _cvt(rv, function.return_type)
        return cls(function, args, rv)


class EqualityConstraint(Constraint):
    """A special constraint for equality constraints. It is equivalent to:

    .. code-block:: python

        Constraint(Constraint.EQUAL, [left, right], rv)

    Basically, it states

    .. code-block:: python

        (left == right) == rv

    Therefore, when rv is True, it states that left and right are equal, and when rv is False, it states that left and right are not equal.
    """

    def __init__(self, left: Union[TensorValue, OptimisticValue], right: Union[TensorValue, OptimisticValue], rv: Optional[Union[TensorValue, OptimisticValue]] = None):
        """Initialize an equality constraint.

        Args:
            left: the left hand side of the equality constraint.
            right: the right hand side of the equality constraint.
            rv: the expected return value of the equality constraint. If None, it will be set to True (i.e., left == right).
        """
        super().__init__(Constraint.EQUAL, [left, right], rv if rv is not None else cvt_opt_value(True, BOOL))

    function: str
    """The function identifier, which is always ``Constraint.EQUAL``."""

    arguments: Tuple[Union[TensorValue, OptimisticValue], Union[TensorValue, OptimisticValue]]
    rv: Union[TensorValue, OptimisticValue]
    note: Any

    @classmethod
    def from_bool(cls, left: Union[bool, int, OptimisticValue], right: Union[bool, int, OptimisticValue], rv: Optional[Union[bool, int, OptimisticValue]] = None) -> 'EqualityConstraint':
        """Create an equality constraint from bool or optimistic values (represented as integers).

        Args:
            left: the left hand side of the equality constraint.
            right: the right hand side of the equality constraint.
            rv: the expected return value of the equality constraint. If None, it will be set to True (i.e., left == right).

        Returns:
            The created equality constraint.
        """

        def _cvt(x):
            if x is None:
                return x
            if isinstance(x, OptimisticValue):
                return x
            elif isinstance(x, bool):
                return cvt_opt_value(x, BOOL)
            else:
                assert isinstance(x, int) and is_optimistic_value(x)
                return OptimisticValue(BOOL, x)
        return cls(_cvt(left), _cvt(right), _cvt(rv))


class ConstraintSatisfactionProblem(object):
    """A constraint satisfaction problem.

    A constraint satisfaction problem is a set of constraints, and a set of variables.
    The solution to a constraint satisfaction problem is to find a set of values for the variables that satisfy all the constraints.
    """

    def __init__(
        self,
        index2actionable: Optional[Dict[int, bool]] = None,
        index2type: Optional[Dict[int, TensorValueTypeBase]] = None,
        index2domain: Optional[Dict[int, Set[Any]]] = None,
        constraints: Optional[List[Constraint]] = None,
        counter: int = 0,
    ):
        """Initialize a constraint satisfaction problem.

        Args:
            counter: the counter for generating new variable indices.
            index2actionable: a mapping from variable indices to whether the variable is actionable.
            index2type: a mapping from variable indices to the type of the variable.
            index2domain: a mapping from variable indices to the domain of the variable.
            constraints: a list of constraints.
        """

        self.index2actionable = index2actionable if index2actionable is not None else dict()
        self.index2type = index2type if index2type is not None else dict()
        self.index2domain = index2domain if index2domain is not None else dict()
        self.constraints = constraints if constraints is not None else list()
        self._optim_var_counter = counter

    index2actionable: Dict[int, bool]
    """A mapping from variable indices to whether the variable is actionable."""

    index2type: Dict[int, Union[TensorValueTypeBase, PyObjValueType]]
    """A mapping from variable indices to the type of the variable."""

    index2domain: Dict[int, Set[Any]]
    """A mapping from variable indices to the domain of the variable."""

    constraints: List[Constraint]
    """A list of constraints."""

    def clone(self, constraints: Optional[List[Constraint]] = None) -> 'ConstraintSatisfactionProblem':
        """Clone the constraint satisfaction problem.

        Args:
            constraints: the constraints to be replaced into the cloned constraint satisfaction problem. If None, the constraints of the original constraint satisfaction problem will be used.
        """

        if constraints is None:
            constraints = self.constraints.copy()
        return type(self)(self.index2actionable.copy(), self.index2type.copy(), self.index2domain.copy(), constraints, self._optim_var_counter)

    def new_actionable_var(self, dtype: Union[TensorValueTypeBase, PyObjValueType], domain: Optional[Set[Any]] = None, wrap: bool = False) -> Union[int, OptimisticValue]:
        """Create a new actionable variable.

        Args:
            dtype: the type of the variable.
            domain: the domain of the variable. If None, it will be assumed to the full domain of the type.
            wrap: whether to wrap the variable index into an OptimisticValue.
        """
        identifier = self.new_var(dtype, domain)
        self.index2actionable[identifier] = True
        if wrap:
            return OptimisticValue(dtype, identifier)
        return identifier

    def new_var(self, dtype: Union[TensorValueTypeBase, PyObjValueType], domain: Optional[Set[Any]] = None, wrap: bool = False) -> Union[int, OptimisticValue]:
        """Create a new variable.

        Args:
            dtype: the type of the variable.
            domain: the domain of the variable. If None, it will be assumed to the full domain of the type.
            wrap: whether to wrap the variable index into an OptimisticValue.

        Returns:
            The index of the new variable (int) if wrap is False, or the wrapped OptimisticValue if wrap is True.
        """

        identifier = OPTIM_MAGIC_NUMBER + self._optim_var_counter
        self.index2type[identifier] = dtype
        self._optim_var_counter += 1
        if domain is not None:
            self.index2domain[identifier] = domain

        if wrap:
            return OptimisticValue(dtype, identifier)
        return identifier

    def get_type(self, identifier: int) -> Union[TensorValueTypeBase, PyObjValueType]:
        """Get the type of a variable."""
        return self.index2type[identifier]

    def get_domain(self, identifier: int) -> Optional[Set[Any]]:
        """Get the domain of a variable."""
        return self.index2domain.get(identifier, None)

    def add_domain_value(self, identifier: int, value: Any):
        """Add a value to the domain of a variable."""
        if identifier not in self.index2domain:
            self.index2domain[identifier] = set()
        self.index2domain[identifier].add(value)

    def add_constraint(self, c: Constraint, note: Optional[Any] = None):
        """Add a constraint to the constraint satisfaction problem.

        Args:
            c: the constraint to be added.
            note: the note of the constraint.
        """
        if note is not None:
            c.note = note
        self.constraints.append(c)

    def __str__(self) -> str:
        fmt = 'ConstraintSatisfactionProblem{\n'
        fmt += '  Actionable Variables:\n    ' + '\n    '.join([f'@{x - OPTIM_MAGIC_NUMBER} - {self.index2type[x]}' for x in self.index2actionable]) + '\n'
        fmt += '  Constraints:\n'
        for c in self.constraints:
            fmt += f'    {c}\n'
        fmt += '}'
        return fmt

    def __repr__(self):
        return self.__str__()


class AssignmentType(IntEnum):
    """See class `Assignment`."""
    EQUAL = 0
    VALUE = 1
    IGNORE = 2


@dataclass
class Assignment(object):
    """An assignment of a variable."""

    t: AssignmentType
    """the type of the assignment. There are three types:

        EQUAL: The variable is equal to another variable.
        VALUE: The variable is equal to a value.
        IGNORE: The variable is ignored. This happens when the variable does not appear in any constraint.
    """

    d: Union[bool, int, None, TensorValue]
    """the value of the variable.

        - EQUAL: The variable is equal to another variable.
        - VALUE: The variable is equal to a value.
        - IGNORE: The variable is ignored. This happens when the variable does not appear in any constraint.
    """

    generator_index: Optional[Tuple[str, int]] = None

    # TODO(Jiayuan Mao @ 2023/08/16): remove this check.
    def __post_init__(self):
        assert self.d is None or isinstance(self.d, (TensorValue, bool)), f'Invalid type of d: {type(self.d)}'

    @property
    def assignment_type(self):
        """Alias of `assignment.t`."""
        return self.t

    @property
    def data(self):
        """Alias of `assignment.d`."""
        return self.d


AssignmentDict = Dict[int, Assignment]
"""A mapping from variable indices to assignment values."""


def print_assignment_dict(assignments: AssignmentDict):
    """Print an assignment dictionary.

    Args:
        assignments: the assignment dictionary.
    """
    print('AssignmentDict{')
    for k, v in assignments.items():
        while v.t is AssignmentType.EQUAL and v.d in assignments:
            v = assignments[v.d]
        if v.t is AssignmentType.VALUE:
            print('  @{}: {}'.format(optimistic_value_id(k), v.d))
        elif v.t is AssignmentType.EQUAL:
            print('  @{}: {}'.format(optimistic_value_id(k), f'@{optimistic_value_id(v.d)}'))
        elif v.t is AssignmentType.IGNORE:
            print('  @{}: {}'.format(optimistic_value_id(k), 'IGNORE'))
        else:
            raise ValueError(f'Unknown assignment type: {v.t}')
    print('}')


def ground_assignment_value(assignments: AssignmentDict, identifier: int) -> Any:
    """Get the value of a variable based on the assignment dictionary. It will follow the EQUAL assignment types.

    Args:
        assignments: the assignment dictionary.
        identifier: the identifier of the variable.

    Returns:
        the value of the variable in the assignment dict.
    """
    while identifier in assignments and assignments[identifier].t is AssignmentType.EQUAL:
        identifier = assignments[identifier].d
    assert identifier in assignments
    return assignments[identifier].d

