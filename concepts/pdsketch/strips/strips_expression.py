#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : strips_expression.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/26/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""
This file defines a set of classes for representing STRIPS expressions, including:

- Boolean constant expressions.
- Boolean predicate expressions.
- SAS predicate expressions.
- And/Or expressions.
- Not expressions.
- Forall/Exists expressions.
- SAS expressions.
- Single predicate assignment expressions.
- Conditional assignment expressions.
- Deictic assignment expressions.

At the highest level, the STRIPS expressions are categorized into two types:

- StripsValueOutputExpression, which outputs a value. There is a special instantiation of this type, StripsBooleanOutputExpression, which outputs a boolean value.
- StripsVariableAssignmentExpression, which assigns a value to a state variable.
"""

import jacinle
from copy import deepcopy
from abc import abstractmethod, ABC
from typing import Optional, Union, Iterable, Sequence, Tuple, Set, FrozenSet, Dict

from concepts.dsl.dsl_types import Variable, ObjectConstant, BOOL
from concepts.dsl.expression import FunctionApplicationExpression, VariableExpression, ObjectConstantExpression, BoolExpression, BoolOpType

__all__ = [
    'SPredicateName', 'SProposition', 'make_sproposition',
    'SState', 'SStateCompatible', 'SStateDict',
    'SExpression', 'SValueOutputExpression', 'SVariableAssignmentExpression', 'SBoolOutputExpression',
    'SBoolConstant', 'SBoolPredicateApplicationExpression', 'SSASPredicateApplicationExpression',
    'SSimpleBoolExpression', 'SBoolNot', 'SQuantificationExpression', 'SSASExpression',
    'SAssignExpression', 'SConditionalAssignExpression', 'SDeicticAssignExpression'
]

"""The name of predicates, represented as strings."""
SPredicateName = str

"""The name of propositions. A proposition is a predicate grounded on a set of arguments."""
SProposition = str

# """The representation of a SAS proposition. It is a tuple of (predicate name, SAS value)."""
# StripsSASProposition = Tuple[str, int]


def _variable_or_constant_to_str(x: Union[str, Variable, ObjectConstant]) -> str:
    if isinstance(x, str):
        return x
    elif isinstance(x, int):
        return str(x)
    elif isinstance(x, Variable):
        return x.name
    elif isinstance(x, ObjectConstant):
        return x.name
    else:
        raise TypeError(f'Unknown type: {type(x)}.')


def make_sproposition(name: SPredicateName, *args: Union[str, Variable, ObjectConstantExpression]) -> SProposition:
    """
    Compose a proposition from a predicate name and a list of arguments.
    """
    if len(args) == 0:
        return name
    return '{} {}'.format(name, ' '.join(_variable_or_constant_to_str(x) for x in args))


def make_sproposition_from_function_application(expr: FunctionApplicationExpression, objects: Optional[Dict[str, str]] = None) -> SProposition:
    """
    Compose a proposition from a function application expression.
    """
    name = expr.function.name

    arguments = list()
    for arg in expr.arguments:
        if not isinstance(arg, ObjectConstantExpression):
            raise ValueError(f'Expected object constant, got {arg}.')
        if objects is not None and arg.name not in objects:
            raise ValueError(f'Unknown object {arg.name}.')
        assert arg.name in objects, f'Unknown object {arg.name}.'
        arguments.append(arg.name)

    return make_sproposition(name, *arguments)


class SState(frozenset, FrozenSet[SProposition]):
    """The representation of a STRIPS state, which is a set of propositions."""
    pass


class SStateDict(dict, Dict[str, Set[Tuple[Union[int, str], ...]]]):
    def add(self, predicate_name: SPredicateName, arguments: Sequence[Union[int, str]]):
        if predicate_name not in self:
            self[predicate_name] = set()
        self[predicate_name].add(tuple(arguments))

    def remove(self, predicate_name: SPredicateName, arguments: Sequence[Union[int, str]]):
        if predicate_name in self:
            self[predicate_name].discard(tuple(arguments))

    def contains(self, predicate_name: SPredicateName, arguments: Sequence[Union[int, str]], negated: bool = False, check_negation: bool = False) -> bool:
        """Check whether the state contains the given proposition.

        Args:
            predicate_name: the name of the predicate.
            arguments: the arguments of the predicate, as a tuple of integers or strings.
            negated: whether the proposition is negated. If True, the function will check whether the state does not contain the proposition.
            check_negation: whether the function should also check "{predicate_name}_not" in the state. This will only be used when `negated` is True.
                This is useful for delete-relaxed planning.

        Returns:
            True if the state contains the proposition, False otherwise.
        """
        if not check_negation:
            if predicate_name in self:
                return (arguments in self[predicate_name]) ^ negated
            return negated  # if the predicate is not in the state and it is not negated, we return False.
        else:
            if not negated:
                if predicate_name in self:
                    return arguments in self[predicate_name]
                return False
            else:
                true_set = self.get(predicate_name, None)
                false_set = self.get(f'{predicate_name}_not', None)
                return (false_set is not None and arguments in false_set) or (true_set is None) or (true_set is not None and arguments not in true_set)

    def clone(self):
        return deepcopy(self)

    def as_state(self) -> SState:
        return SState([f'{predicate_name} {" ".join(map(str, arguments))}' for predicate_name, list_of_arguments in self.items() for arguments in list_of_arguments])


SStateCompatible = Union[SState, Set[SProposition]]


# class StripsSASState(dict, Dict[Tuple[StripsPredicateName, str], int]):
#     """The representation of a SAS state, which is a mapping from (predicate name, SAS value) to the number of occurrences."""
#     """StripsSASState is a mapping from (predicate name, arguments_str) to value."""
#     pass


class SExpression(ABC):
    """The base class for STRIPS expressions."""
    __repr__ = jacinle.repr_from_str

    def ground(self, variable_dict: Dict[str, str], state: Optional[SStateCompatible] = None):
        """Return a new expression with all variables grounded according to the given variable dictionary."""
        raise NotImplementedError()

    @abstractmethod
    def iter_precondition_predicates(self) -> Iterable[SPredicateName]:
        """Iterate over the precondition predicate names in the expression."""
        raise NotImplementedError()

    @abstractmethod
    def iter_effect_predicates(self) -> Iterable[SPredicateName]:
        """Iterate over the effect predicate names in the expression."""
        raise NotImplementedError()


class SValueOutputExpression(SExpression, ABC):
    """The base class for STRIPS expressions that output a value."""

    @abstractmethod
    def iter_precondition_predicates(self) -> Iterable[SPredicateName]:
        """Iterate over the precondition predicate names in the expression."""
        raise NotImplementedError()

    def iter_effect_predicates(self) -> Iterable[SPredicateName]:
        """Iterate over the effect predicate names in the expression."""
        return set()


class SVariableAssignmentExpression(SExpression, ABC):
    """The base class for STRIPS expressions that assign a value to a state variable."""

    def iter_precondition_predicates(self) -> Iterable[SPredicateName]:
        """Iterate over the precondition predicate names in the expression."""
        return set()

    @abstractmethod
    def iter_effect_predicates(self) -> Iterable[SPredicateName]:
        """Iterate over the effect predicate names in the expression."""
        raise NotImplementedError()


class SBoolOutputExpression(SValueOutputExpression, ABC):
    """The base class for STRIPS expressions that output a boolean value."""
    pass


class SBoolConstant(SBoolOutputExpression):
    """The representation of a boolean constant."""

    def __init__(self, constant: bool):
        """Initialize a boolean constant.

        Args:
            constant: the value of the constant.
        """

        self.constant = constant

    def ground(self, variable_dict: Dict[str, str], state: Optional[SStateCompatible] = None):
        from concepts.pdsketch.strips.strips_grounded_expression import GSBoolConstantExpression
        return GSBoolConstantExpression(self.constant)

    def iter_precondition_predicates(self) -> Iterable[SPredicateName]:
        return set()

    def __str__(self) -> str:
        return 'true' if self.constant else 'false'


class SBoolPredicateApplicationExpression(SBoolOutputExpression):
    """The base class for STRIPS expressions that output a boolean value based on a predicate."""

    def __init__(self, name: SPredicateName, negated: bool, arguments: Sequence[Union[Variable, str]]):
        """Initialize a boolean predicate expression.

        Args:
            name: the name of the predicate.
            negated: whether the predicate is negated.
            arguments: the arguments of the predicate. Either variables or str (constants).
        """

        self.name = name
        self.arguments = tuple(arguments)
        self.negated = negated

    name: SPredicateName
    """The name of the predicate."""

    arguments: Tuple[Union[Variable, str]]
    """The arguments of the predicate."""

    negated: bool
    """Whether the predicate is negated."""

    def ground(self, variable_dict: Dict[str, str], state: Optional[SStateCompatible] = None, negated: bool = False, return_proposition: bool = False):
        """Ground the expression according to the given variable dictionary.

        Args:
            variable_dict: the variable dictionary.
            state: the state to ground the expression on. If None, the expression will be grounded without considering the state.
            negated: whether the predicate is negated.
            return_proposition: whether to return a SProposition instead of a GSSimpleBoolExpression

        Returns:
            the grounded expression. Will be a GSSimpleBoolExpression if `return_proposition` is False, otherwise a SProposition.
        """
        from concepts.pdsketch.strips.strips_grounded_expression import GSSimpleBoolExpression
        identifier = self.name + '_not' if (self.negated ^ negated) else self.name
        proposition = make_sproposition(identifier, *tuple(variable_dict[argument.name] if isinstance(argument, Variable) else argument for argument in self.arguments))

        if return_proposition:
            return proposition
        return GSSimpleBoolExpression({proposition})

    def iter_precondition_predicates(self) -> Iterable[SPredicateName]:
        return {self.name}

    def __str__(self) -> str:
        if len(self.arguments) == 0:
            fmt = f'({self.name})'
        else:
            argument_str = ' '.join(x.name if isinstance(x, Variable) else x for x in self.arguments)
            fmt = f'({self.name} {argument_str})'
        if self.negated:
            return f'(not {fmt})'
        return fmt

    @classmethod
    def from_function_application_expression(cls, expression: Union[FunctionApplicationExpression, BoolExpression], negated: bool = False):
        assert isinstance(expression, (FunctionApplicationExpression, BoolExpression)), f'Invalid expression type: {type(expression)}.'

        if isinstance(expression, BoolExpression):
            assert expression.bool_op is BoolOpType.NOT
            assert len(expression.arguments) == 1
            expression = expression.arguments[0]
            assert isinstance(expression, FunctionApplicationExpression)
            negated = not negated
            return cls.from_function_application_expression(expression, negated)

        assert expression.function.return_type == BOOL
        new_arguments = list()
        for arg in expression.arguments:
            if isinstance(arg, VariableExpression):
                new_arguments.append(arg.variable)
            elif isinstance(arg, ObjectConstantExpression):
                new_arguments.append(arg.constant.name)
            else:
                raise TypeError(f'Invalid argument type: {type(arg)}.')
        return cls(expression.function.name, negated, new_arguments)


class SSASPredicateApplicationExpression(SBoolPredicateApplicationExpression):
    """The representation for an SAS predicate expression. It is composed of a predicate name and an SAS index."""

    def __init__(self, sas_name: SPredicateName, sas_index: Optional[int], negated: bool, arguments: Sequence[Variable]):
        """Initialize an SAS predicate expression.

        Args:
            sas_name: the name of the SAS predicate.
            sas_index: the index of the SAS predicate.
            negated: whether the predicate is negated.
            arguments: the arguments of the predicate.
        """
        if sas_index is None:
            super().__init__(sas_name, negated, arguments)
        else:
            super().__init__(sas_name + '@' + str(sas_index), negated, arguments)

        self.sas_name = sas_name
        self.sas_index = sas_index

    name: SPredicateName
    arguments: Tuple[Variable]
    negated: bool

    sas_name: SPredicateName
    """The name of the SAS predicate."""

    sas_index: Optional[int]
    """The index of the SAS predicate. If None, it is a normal predicate."""


class SSimpleBoolExpression(SBoolOutputExpression):
    """The representation of a boolean expression. Note that since the negation is recorded in the raw :class:`StripsBoolPredicateApplicationExpression`,
    we do not need to record it here. Therefore, in this class, we only need to record whether the expression is an AND or an OR."""

    def __init__(self, arguments: Sequence[SBoolOutputExpression], is_disjunction: bool):
        """Initialize a boolean expression.

        Args:
            arguments: the arguments of the expression.
            is_disjunction: whether the expression is a disjunction.
        """
        self.arguments = arguments
        self.is_disjunction = is_disjunction

    arguments: Sequence[SBoolOutputExpression]
    """The arguments of the expression."""

    is_disjunction: bool
    """Whether the expression is a disjunction."""

    def ground(self, variable_dict: Dict[str, str], state: Optional[SStateCompatible] = None):
        from concepts.pdsketch.strips.strips_grounded_expression import gs_compose_bool_expressions
        return gs_compose_bool_expressions(
            [argument.ground(variable_dict) for argument in self.arguments],
            is_disjunction=self.is_disjunction,
        )

    @property
    def is_conjunction(self) -> bool:
        """Whether the expression is a conjunction."""
        return not self.is_disjunction

    def iter_precondition_predicates(self) -> Iterable[SPredicateName]:
        return set.union(*(x.iter_precondition_predicates() for x in self.arguments))

    def __str__(self) -> str:
        arguments_str = [str(arg) for arg in self.arguments]
        if sum(len(s) for s in arguments_str) > 120:
            arguments_str = [jacinle.indent_text(s) for s in arguments_str]
            fmt = '\n'.join(arguments_str)
            return f'(or\n{fmt}\n)' if self.is_disjunction else f'(and\n{fmt}\n)'
        return '({} {})'.format('or' if self.is_disjunction else 'and', ' '.join(arguments_str))


class SBoolNot(SBoolOutputExpression):
    """The representation of a boolean NOT expression. Note that this class is usually only used as a temporary expression during parsing.
    At the end, the negation is recorded in the raw :class:`StripsBoolPredicateApplicationExpression`."""

    def __init__(self, expr: SBoolOutputExpression):
        """Initialize a boolean NOT expression.

        Args:
            expr: the expression to be negated.
        """
        self.expr = expr

    expr: SBoolOutputExpression
    """The expression to be negated."""

    def ground(self, variable_dict: Dict[str, str], state: Optional[SStateCompatible] = None):
        if isinstance(self.expr, SBoolPredicateApplicationExpression):
            return self.expr.ground(variable_dict, negated=True)
        raise NotImplementedError()

    def iter_precondition_predicates(self) -> Iterable[SPredicateName]:
        return self.expr.iter_precondition_predicates()

    def __str__(self) -> str:
        return '(not {})'.format(str(self.expr))


class SQuantificationExpression(SBoolOutputExpression):
    """The representation of a quantification expression."""

    def __init__(self, variable: Variable, expr: SBoolOutputExpression, is_disjunction: bool):
        """Initialize a quantification expression.

        Args:
            variable: the variable to be quantified.
            expr: the expression to be quantified.
            is_disjunction: whether the expression is a disjunction (EXISTS quantification).
        """

        self.variable = variable
        self.expr = expr
        self.is_disjunction = is_disjunction

    variable: Variable
    """The variable to be quantified."""

    expr: SBoolOutputExpression
    """The expression to be quantified."""

    is_disjunction: bool
    """Whether the expression is a disjunction (EXISTS quantification)."""

    @property
    def is_conjunction(self) -> bool:
        """Whether the expression is a conjunction (FORALL quantification)."""
        return not self.is_disjunction

    @property
    def is_forall(self) -> bool:
        """Whether the expression is a conjunction (FORALL quantification)."""
        return not self.is_disjunction

    @property
    def is_exists(self) -> bool:
        """Whether the expression is a disjunction (EXISTS quantification)."""
        return self.is_disjunction

    def ground(self, variable_dict: Dict[str, str], state: Optional[SStateCompatible] = None):
        raise NotImplementedError()

    def iter_precondition_predicates(self) -> Iterable[SPredicateName]:
        return self.expr.iter_precondition_predicates()

    def __str__(self) -> str:
        return '({} ({}) {})'.format('exists' if self.is_disjunction else 'forall', str(self.variable), str(self.expr))


class SSASExpression(SValueOutputExpression):  # For all external functions.
    """The representation of an SAS expression. The return value of the expression is an SAS index, therefore it can be represented as a dictionary,
    mapping from Boolean expressions to SAS indices. The execution procedure is to first evaluate all Boolean expressions, and then set the SAS index.
    Suggested implementation is:

    .. code-block:: python

        for sas_index, expr in self.mappings.items():
            if evaluate(expr, state):
                return sas_index
    """

    def __init__(self, mappings: Dict[int, SBoolOutputExpression]):
        """Initialize an SAS expression.

        Args:
            mappings: the mappings from SAS indices to Boolean expressions.
        """
        self.mappings: Dict[int, SBoolOutputExpression] = mappings

    mappings: Dict[int, SBoolOutputExpression]
    """The mappings from SAS indices to Boolean expressions."""

    def ground(self, variable_dict: Dict[str, str], state: Optional[SStateCompatible] = None):
        raise NotImplementedError()

    def __str__(self) -> str:
        return '(SAS\n{}\n)'.format('\n'.join('  ' + str(i) + ' <- ' + str(self.mappings[i]) for i in self.mappings))


class SAssignExpression(SVariableAssignmentExpression):
    """The representation of an assignment expression."""

    def __init__(self, predicate: Union[SBoolPredicateApplicationExpression, SSASPredicateApplicationExpression], value: Union[SBoolOutputExpression, SSASExpression]):
        """Initialize an assignment expression.

        Args:
            predicate: the predicate in the state representation to be assigned.
            value: the value to be assigned.
        """
        self.predicate = predicate
        self.value = value

    predicate: Union[SBoolPredicateApplicationExpression, SSASPredicateApplicationExpression]
    """The predicate in the state representation to be assigned."""

    value: Union[SBoolOutputExpression, SSASExpression]
    """The value to be assigned."""

    def iter_effect_predicates(self) -> Iterable[SPredicateName]:
        return self.predicate.iter_precondition_predicates()

    def __str__(self) -> str:
        return '({} <- {})'.format(str(self.predicate), str(self.value))


class SConditionalAssignExpression(SVariableAssignmentExpression):
    """The representation of a conditional assignment expression. Note that the inner assignment expression is always a :class:`StripsAssignment`."""

    def __init__(self, assign_op: SAssignExpression, condition: SBoolOutputExpression):
        """Initialize a conditional assignment expression.

        Args:
            assign_op: the assignment expression.
            condition: the condition expression.
        """
        self.assign_op = assign_op
        self.condition = condition

    assign_op: SAssignExpression
    """The assignment expression."""

    condition: SBoolOutputExpression
    """The condition expression."""

    @property
    def predicate(self):
        """The predicate in the state representation to be assigned."""
        return self.assign_op.predicate

    @property
    def value(self):
        """The value to be assigned, if the condition is satisfied."""
        return self.assign_op.value

    def iter_precondition_predicates(self) -> Iterable[SPredicateName]:
        return self.condition.iter_precondition_predicates()

    def iter_effect_predicates(self) -> Iterable[SPredicateName]:
        return self.assign_op.iter_effect_predicates()

    def __str__(self) -> str:
        return '({} if {})'.format(str(self.assign_op), str(self.condition))


class SDeicticAssignExpression(SVariableAssignmentExpression):
    """The representation of a deictic assignment expression."""

    def __init__(self, variable: Variable, expression: SVariableAssignmentExpression):
        """Initialize a deictic assignment expression.

        Args:
            variable: the deictic variable.
            expression: the inner assignment expression.
        """
        self.variable = variable
        self.expression = expression

    variable: Variable
    """The deictic expression."""

    expression: SVariableAssignmentExpression
    """The inner assignment expression."""

    def iter_precondition_predicates(self) -> Iterable[SPredicateName]:
        return self.expression.iter_precondition_predicates()

    def iter_effect_predicates(self) -> Iterable[SPredicateName]:
        return self.expression.iter_effect_predicates()

    def __str__(self) -> str:
        return '(foreach ({}) {})'.format(self.variable, str(self.expression))

