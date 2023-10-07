#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : strips_grounded_expression.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/27/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""This file contains the implementation of grounded STRIPS expressions.
Here, the word "grounded" means that the expressions have been instantiated on a specific state. Therefore, the expressions
will not contain any variables (e.g., quantification or deictic expressions). This will make the representations significantly simpler.
For example, the :class:`~concepts.pdsketch.strips.strips_expression.SQuantificationExpression` will be translated into a simple
:class:`GSSimpleBoolExpression` (explicit conjunction or disjunction).

To maximize the execution efficiency of grounded expressions (because they will typically be used in the inner loop of the planner for heuristic evaluation),
we use a compilation strategy to convert each grounded expression into a function that can be called directly. Therefore, each grounded expression class
will have a ``compile`` method that returns a function that takes a state as input and returns a Boolean value.

Note that since we want to support delete-relaxation heuristics, the compiled function will support a feature called "forward difference." This means,
when we compute a Boolean expression on a state, we will also keep track of the branches that makes the expression true. Specifically, we will keep track
of all state variable-level propositions that have contributed to the "true" value of the expression. During the backtracking stage of the heuristic
computation, we will use this information to mark corresponding state variables. See the documentation of :meth:`GSBoolExpression.enable_forward_diff_ctx` for more details.
"""

import warnings
import contextlib

from abc import ABC, abstractmethod
from typing import Any, Optional, Union, Callable, Iterable, Sequence, Tuple, List, Set, FrozenSet, Dict
from dataclasses import dataclass

import jacinle
from concepts.pdsketch.strips.strips_expression import SPredicateName, SProposition, SState, SStateCompatible

__all__ = [
    'GSOptimisticStaticObjectType', 'GS_OPTIMISTIC_STATIC_OBJECT', 'GSBoolForwardDiffReturn',
    'GSExpression', 'GSBoolOutputExpression', 'GSBoolConstantExpression', 'GSSimpleBoolExpression', 'GSComplexBoolExpression',
    'GSVariableAssignmentExpression', 'GSSimpleBoolAssignExpression', 'GSConditionalAssignExpression', 'GSSASAssignExpression', 'GStripsDerivedPredicate',
    'gs_compose_bool_expressions',
    'gs_is_constant_true', 'gs_is_constant_false', 'gs_is_empty_bool_expression', 'gs_is_simple_conjunctive_classifier',
]


class GSOptimisticStaticObjectType(object):
    """The underlying type for :class:`GS_OPTIMISTIC_STATIC_OBJECT`."""
    pass


GS_OPTIMISTIC_STATIC_OBJECT = GSOptimisticStaticObjectType()
"""OptimisticObject only occurs when the arguments to an operator is a complex-typed (Tensor or PyObject) value."""


@dataclass
class GSBoolForwardDiffReturn(object):
    """The return type of the forward difference function of a Boolean expression."""

    rv: bool
    """The return value of the expression."""

    propositions: Union[FrozenSet[SProposition], Set[SProposition]]
    """The set of propositions that have contributed to the return value."""


class GSExpression(ABC):
    """The base class for all grounded STRIPS expressions."""

    @abstractmethod
    def iter_propositions(self) -> Iterable[SProposition]:
        """Iterate over all propositions that are used in this expression."""
        raise NotImplementedError()

    @abstractmethod
    def filter_propositions(self, propositions: SStateCompatible, state: Optional[SState] = None) -> 'GSExpression':
        """Filter the propositions in this expression. Only the propositions in the given set will be kept. Note that this function also takes a state as input,
        essentially, the state is the initial state of the environment, and the `propositions` contains all propositions that will be possibly changed
        by actions. Therefore, for propositions outside the `propositions` set, their value will stay as the same value as in the initial state. See, for example,
        the implementation for :meth:`GSSimpleBoolExpression.filter_propositions` for more details.

        Args:
            propositions: the propositions that should be kept.
            state: the initial state, default to None.

        Returns:
            the filtered expression.
        """
        raise NotImplementedError()

    @abstractmethod
    def compile(self) -> Callable[[SState], Any]:
        """Compile the expression into a function that takes a state as input and returns a boolean value."""
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()

    __repr__ = jacinle.repr_from_str


class GSBoolOutputExpression(GSExpression, ABC):
    """The base class for all Boolean expressions."""

    FORWARD_DIFF = False
    """A static variable that indicates whether we want to enable forward difference for expressions."""

    @staticmethod
    def set_forward_diff(value: bool = True):
        """Set the forward difference flag.
        If the forward difference flag is set to True, the compiled function will return a tuple of (bool, Set[SProposition]) instead of a bool.

        Args:
            value: the value to set, default to True.
        """
        GSBoolOutputExpression.FORWARD_DIFF = value

    @staticmethod
    @contextlib.contextmanager
    def enable_forward_diff_ctx():
        """A context manager that enables the forward difference flag. Example:

        .. code-block:: python

            state = SState({'a', 'b'})
            expr = GSSimpleBoolExpression({'a', 'b'})
            compiled_function = expr.compile()
            with GSBoolExpression.enable_forward_diff_ctx():
                rv = compiled_function(state)
                assert isinstance(rv, GSBoolForwardDiffReturn)
                assert rv.rv is True
                assert rv.propositions == {'a', 'b'}
        """
        old_value = GSBoolOutputExpression.FORWARD_DIFF
        GSBoolOutputExpression.FORWARD_DIFF = True
        yield
        GSBoolOutputExpression.FORWARD_DIFF = old_value

    def compile(self) -> Callable[[SState], Union[bool, GSBoolForwardDiffReturn]]:
        """Compile the expression into a function that takes a state as input."""
        raise NotImplementedError()

    def forward(self, state: SState) -> Union[bool, GSBoolForwardDiffReturn]:
        """Compute the expression on the given state.

        Args:
            state: the state to compute the expression on.

        Returns:
            the return value of the expression.
        """

        if hasattr(self, '_compiled_function'):
            compiled_function = self._compiled_function
        else:
            compiled_function = self._compiled_function = self.compile()

        return compiled_function(state)

    def filter_propositions(self, propositions: SStateCompatible, state: Optional[SState] = None) -> 'GSBoolOutputExpression':
        raise NotImplementedError()


class GSVariableAssignmentExpression(GSExpression, ABC):
    """The base class for assignment expressions."""

    @abstractmethod
    def compile(self) -> Callable[[SState], SState]:
        """Compile the assignment expression into a function. By default, the function will return the state with the add and delete effects applied."""
        raise NotImplementedError()

    @abstractmethod
    def relax(self) -> Union['GSVariableAssignmentExpression', List['GSVariableAssignmentExpression']]:
        """Relax the expression (a.k.a. delete relaxation)."""
        raise NotImplementedError()

    def filter_propositions(self, propositions: SStateCompatible, state: Optional[SState] = None) -> 'GSVariableAssignmentExpression':
        raise NotImplementedError()


class GSBoolConstantExpression(GSBoolOutputExpression):
    """A constant Boolean expression."""

    def __init__(self, constant: bool):
        """Initialize the constant expression.

        Args:
            constant: the constant value.
        """
        super().__init__()
        self.constant = constant

    constant: bool
    """The constant value."""

    def compile(self) -> Callable[[SState], Union[bool, GSBoolForwardDiffReturn]]:
        def function(state: SState, constant=self.constant):
            if GSBoolOutputExpression.FORWARD_DIFF:
                return GSBoolForwardDiffReturn(constant, set())
            return constant
        return function

    def iter_propositions(self) -> Iterable[SProposition]:
        return tuple()

    def filter_propositions(self, propositions: Set[SProposition], state: Optional[SState] = None) -> 'GSBoolConstantExpression':
        """Filter the propositions in this expression. Since this is a constant expression, the result will always be the same expression.

        Args:
            propositions: the propositions that should be kept.
            state: the initial state, default to None.

        Returns:
            the Boolean expression after filtering.
        """
        return self

    def __str__(self):
        return '{}'.format(self.constant).upper()


class GSSimpleBoolExpression(GSBoolOutputExpression):
    """A simple Boolean expression. Here, a simple Boolean expression is a conjunction or disjunction of propositions. Therefore, internally,
    we use a frozen set to store the propositions. This also accelerates the testing process."""

    def __init__(self, propositions: Union[Sequence[SProposition], FrozenSet[str]], is_disjunction: bool = False):
        """Initialize the simple Boolean expression.

        Args:
            propositions: the propositions in the expression.
            is_disjunction: whether the expression is a disjunction, default to False (a.k.a. conjunction).
        """
        super().__init__()

        self.propositions = frozenset(propositions)
        if len(self.propositions) > 1:
            self.is_disjunction = is_disjunction
        else:
            self.is_disjunction = False  # prefer to represent a single prop. as a conjunction.

    propositions: FrozenSet[SProposition]
    """The propositions in the expression."""

    is_disjunction: bool
    """Whether the expression is a disjunction."""

    @property
    def is_conjunction(self) -> bool:
        """Whether the expression is a conjunction."""
        return not self.is_disjunction

    def compile(self) -> Callable[[SState], Union[bool, GSBoolForwardDiffReturn]]:
        """Compile the expression into a function. When forward diff is enabled, the function will use the following strategy:

        1. If the expression is a conjunction, the function will return True if all propositions are true, and return all propositions as the forward diff.
        2. If the expression is a disjunction, the function will return True if any proposition is true, and return the first true proposition as the forward diff.
        """
        if self.is_disjunction:
            def function(state: SState, classifier=self.propositions) -> Optional[Union[bool, GSBoolForwardDiffReturn]]:
                if GSBoolOutputExpression.FORWARD_DIFF:
                    intersection = classifier.intersection(state)
                    if intersection:
                        return GSBoolForwardDiffReturn(True, {next(iter(intersection))})
                    else:
                        return GSBoolForwardDiffReturn(False, None)
                return len(classifier.intersection(state)) > 0
            return function
        else:
            def function(state: SState, classifier=self.propositions) -> Optional[Union[bool, GSBoolForwardDiffReturn]]:
                if GSBoolOutputExpression.FORWARD_DIFF:
                    return GSBoolForwardDiffReturn(classifier <= state, classifier)
                return classifier <= state
            return function

    def iter_propositions(self) -> Iterable[SProposition]:
        yield from iter(self.propositions)

    def filter_propositions(self, propositions: Set[SProposition], state: Optional[SState] = None) -> 'GSBoolOutputExpression':
        """Filter the propositions in this expression. Only the propositions in the given set will be kept. Note that this function will automatically
        handle cases where the filtered expression is a constant expression. The rules are:

        1. If the expression is a conjunction, denote the set of propositions that will be filtered out as `A`, and the set of propositions that
        are True in the initial state as `B`:

            1. If `A` is a not subset of `B`, then the filtered expression is a constant False expression (i.e., there are propositions that are
            not True in the initial state but will never be changed by actions).

            2. Otherwise, if the remaining set of propositions is empty, then the filtered expression is a constant True expression. Otherwise,
            the filtered expression is a conjunction of the remaining propositions.

        2. If the expression is a disjunction, denote the set of propositions that will be filtered out as `A`, and the set of propositions that
        are True in the initial state as `B`:

            1. If `A` and `B` has non-empty intersection, then the filtered expression is a constant True expression (i.e., there are propositions that are
            True in the initial state and will never be changed by actions).

            2. Otherwise, if the remaining set of propositions is empty, then the filtered expression is a constant False expression (i.e., all propositions
            are False in the initial state and will never be changed by actions). Otherwise, the filtered expression is a disjunction of the remaining propositions.

        Args:
            propositions: the propositions that should be kept.
            state: the initial state, default to None.

        Returns:
            the Boolean expression after filtering.
        """
        if state is None:
            state = set()

        diff = self.propositions - propositions
        new_classifiers = frozenset(self.propositions & propositions)

        if len(diff) == 0:
            return self
        else:
            if self.is_disjunction:
                if len(diff & state) > 0:
                    return GSBoolConstantExpression(True)
                else:
                    if len(new_classifiers) == 0:
                        return GSBoolConstantExpression(False)
                    else:
                        return GSSimpleBoolExpression(new_classifiers, self.is_disjunction)
            else:
                if not diff <= state:
                    return GSBoolConstantExpression(False)
                else:
                    if len(new_classifiers) == 0:
                        return GSBoolConstantExpression(True)
                    else:
                        return GSSimpleBoolExpression(new_classifiers, self.is_disjunction)

    def __str__(self) -> str:
        name = 'CONJ' if not self.is_disjunction else 'DISJ'
        classifier_str = ', '.join([str(x) for x in self.propositions])
        return f'{name}({classifier_str})'


class GSComplexBoolExpression(GSBoolOutputExpression):
    """A complex Boolean expression. Here, a complex Boolean expression is a conjunction or disjunction of Boolean sub-expressions.
    In most of the scenarios, you should directly call the constructor for this class. Instead, if you want to compose multiple Boolean
    expressions, you should use the function :func:`gs_compose_bool_expressions` instead."""

    def __init__(self, expressions: Sequence[GSBoolOutputExpression], is_disjunction: Optional[bool] = False):
        """Initialize the complex Boolean expression.

        Args:
            expressions: the sub-expressions.
            is_disjunction: whether the expression is a disjunction, default to False (a.k.a. conjunction).
        """
        super().__init__()
        self.expressions = tuple(expressions)
        self.is_disjunction = is_disjunction

    expressions: Tuple[GSBoolOutputExpression]
    """The sub-expressions."""

    is_disjunction: bool
    """Whether the expression is a disjunction."""

    @property
    def is_conjunction(self) -> bool:
        """Whether the expression is a conjunction."""
        return not self.is_disjunction

    def compile(self) -> Callable[[SState], Union[bool, GSBoolForwardDiffReturn]]:
        """Compile the expression into a function. When forward diff is enabled, the function will use the following strategy:

        1. If the expression is a conjunction, the function will return True if all sub-expressions are true, and return the union of propositions as the forward diff.
        2. If the expression is a disjunction, the function will return True if any sub-expression is true, and return the first true sub-expression as the forward diff.
        """
        compiled_functions = tuple(e.compile() for e in self.expressions)
        if self.is_disjunction:
            def function(state: SState, functions=compiled_functions) -> Union[bool, GSBoolForwardDiffReturn]:
                if GSBoolOutputExpression.FORWARD_DIFF:
                    for f in functions:
                        rv = f(state)
                        result, propositions = rv.rv, rv.propositions
                        if result:
                            return GSBoolForwardDiffReturn(True, propositions)
                    return GSBoolForwardDiffReturn(False, set())

                return any(f(state) for f in functions)
        else:
            def function(state: SState, functions=compiled_functions) -> Union[bool, GSBoolForwardDiffReturn]:
                if GSBoolOutputExpression.FORWARD_DIFF:
                    all_propositions = list()
                    for f in functions:
                        rv = f(state)
                        result, propositions = rv.rv, rv.propositions
                        if not result:
                            return GSBoolForwardDiffReturn(False, set())
                        all_propositions.append(propositions)
                    propositions = frozenset.union(*all_propositions)
                    return GSBoolForwardDiffReturn(True, propositions)

                return all(f(state) for f in functions)
        return function

    def iter_propositions(self) -> Iterable[SProposition]:
        for e in self.expressions:
            yield from e.iter_propositions()

    def filter_propositions(self, propositions: Set[SProposition], state: Optional[SState] = None) -> 'GSBoolOutputExpression':
        """Filter the given propositions from the expression. See the documentation of :func:`gstrips_compose_classifiers` for more details.

        Args:
            propositions: the propositions that should be kept.
            state: the initial state, default to None.

        Returns:
            the Boolean expression after filtering.
        """
        expressions = [e.filter_propositions(propositions, state=state) for e in self.expressions]
        return gs_compose_bool_expressions(expressions, self.is_disjunction, propagate_implicit_propositions=False)

    def __str__(self):
        name = 'ComplexAND' if not self.is_disjunction else 'ComplexOR'
        expressions = [str(x) for x in self.expressions]
        if sum([len(x) for x in expressions]) > 120:
            return f'{name}(\n' + ',\n'.join([jacinle.indent_text(jacinle.stformat(x)).rstrip() for x in expressions]) + '\n)'
        return f'{name}(' + ', '.join([str(x) for x in self.expressions]) + ')'


class GSSimpleBoolAssignExpression(GSVariableAssignmentExpression):
    """A simple assignment expression. Here, a simple assignment expression is represented by the set of add effects and the set of delete effects, both
    of which are a set of propositions."""

    def __init__(self, add_effects: Iterable[SProposition], del_effects: Iterable[SProposition]):
        """Initialize the simple assignment expression.

        Args:
            add_effects: the set of add effects.
            del_effects: the set of delete effects.
        """
        super().__init__()
        self.add_effects = frozenset(add_effects)
        self.del_effects = frozenset(del_effects)

    add_effects: FrozenSet[SProposition]
    """The set of add effects."""

    del_effects: FrozenSet[SProposition]
    """The set of delete effects."""

    def compile(self) -> Callable[[SState], SState]:
        def function(state: SState, del_effects=self.del_effects, add_effects=self.add_effects) -> SState:
            new_state = (state - del_effects) | add_effects
            return SState(new_state)
        return function

    def iter_propositions(self) -> Iterable[SProposition]:
        yield from iter(self.add_effects)
        yield from iter(self.del_effects)

    def filter_propositions(self, propositions: Set[SProposition], state: Optional[SState] = None) -> 'GSSimpleBoolAssignExpression':
        """Filter the given propositions from the expression. For simple assignment expressions, the filtering is done simply by removing irrelevant propositions
        from the add and delete effects."""
        return GSSimpleBoolAssignExpression(propositions & self.add_effects, propositions & self.del_effects)

    def relax(self) -> 'GSSimpleBoolAssignExpression':
        """Delete relaxation of a simple assignment. Essentially, it removes all delete effects."""
        return GSSimpleBoolAssignExpression(self.add_effects, set())

    def __str__(self) -> str:
        return f'EFF[add={self.add_effects}, del={self.del_effects}]'


class GSConditionalAssignExpression(GSVariableAssignmentExpression):
    """A conditional assignment expression. Note that the inner expression must be a simple assignment expression.
    Therefore , a conditional assignment expression is represented by a condition, a set of add effects, and a set of delete effects."""

    def __init__(self, condition: GSBoolOutputExpression, assignment: GSSimpleBoolAssignExpression):
        """Initialize the conditional assignment expression.

        Args:
            condition: the condition.
            assignment: the simple assignment expression.
        """

        super().__init__()
        self.condition = condition
        self.assignment = assignment
        self.condition_func = None
        self.assignment_func = None

    condition: GSBoolOutputExpression
    """The condition of the conditional assignment expression."""

    assignment: GSSimpleBoolAssignExpression
    """The inner (simple) assignment expression of the conditional assignment expression."""

    condition_func: Optional[Callable[[SState], Union[bool, GSBoolForwardDiffReturn]]]
    """The compiled condition function."""

    assignment_func: Optional[Callable[[SState], SState]]
    """The compiled assignment function."""

    @property
    def add_effects(self) -> FrozenSet[SProposition]:
        """The add effects of the conditional assignment expression."""
        assert isinstance(self.assignment, GSSimpleBoolAssignExpression)
        return self.assignment.add_effects

    @property
    def del_effects(self) -> FrozenSet[SProposition]:
        """The delete effects of the conditional assignment expression."""
        assert isinstance(self.assignment, GSSimpleBoolAssignExpression)
        return self.assignment.del_effects

    def compile(self) -> Callable[[SState], SState]:
        condition_func = self.condition.compile()
        assignment_func = self.assignment.compile()

        def function(state: SState, condition_func=condition_func, assignment=assignment_func) -> SState:
            if condition_func(state):
                return assignment(state)
            return state
        self.condition_func = condition_func
        self.assignment_func = assignment_func
        return function

    def applicable(self, state: SState) -> Union[bool, GSBoolForwardDiffReturn]:
        """Check if the conditional assignment expression is applicable in the given state."""
        return self.condition_func(state)

    def apply(self, state: SState) -> SState:
        """Apply the conditional assignment expression in the given state."""
        return self.assignment_func(state)

    def iter_propositions(self) -> Iterable[SProposition]:
        yield from self.assignment.iter_propositions()

    def filter_propositions(self, propositions: Set[SProposition], state: Optional[SState] = None) -> 'GSConditionalAssignExpression':
        """Filter the given propositions from the expression. For conditional assignment expressions, the filtering is done by filtering the inner simple
        assignment expression and the condition.

        Args:
            propositions: the propositions that should be kept.
            state: the initial state, default to None.

        Returns:
            the conditional assignment expression after filtering.
        """
        return GSConditionalAssignExpression(
            self.condition.filter_propositions(propositions, state=state),
            self.assignment.filter_propositions(propositions, state=state)
        )

    def relax(self) -> 'GSConditionalAssignExpression':
        """Delete relaxation of a conditional assignment. Essentially, it removes all delete effects for the inner simple assignment expression.

        Returns:
            the delete relaxed conditional assignment expression.
        """
        return GSConditionalAssignExpression(self.condition, self.assignment.relax())

    def __str__(self) -> str:
        if isinstance(self.assignment, GSSimpleBoolAssignExpression):
            return f'CONDEFF[{self.condition} => add={self.assignment.add_effects}, del={self.assignment.del_effects}]'
        else:
            return f'CONDEFF[{self.condition} => {self.assignment}]'


class GSSASAssignExpression(GSVariableAssignmentExpression):
    """A SAS assignment expression. It is represented as the name for the SAS predicate, the size of the SAS predicate, a list of assignments,
    and a mapping from Boolean expressions to integers (the value of the SAS assignment)."""

    def __init__(
        self,
        sas_name: SPredicateName, sas_size: int, arguments: Sequence[str],
        expression: Dict[int, GSBoolOutputExpression]
    ):
        """Initialize the SAS assignment expression.

        Args:
            sas_name: the name of the SAS predicate.
            sas_size: the size of the SAS predicate.
            arguments: the arguments of the SAS predicate.
            expression: the expressions for SAS assignments.
        """

        super().__init__()
        self.sas_name = sas_name
        self.sas_size = sas_size
        self.arguments = arguments
        self.arguments_str = ' '.join(str(arg) for arg in arguments)
        self.all_bool_predicates = frozenset({f'{sas_name}@{i} {self.arguments_str}' for i in range(sas_size)})
        self.all_neg_bool_predicates = frozenset({f'{sas_name}@{i}_not {self.arguments_str}' for i in range(sas_size)})
        self.expression = expression

    sas_name: SPredicateName
    """The name of the SAS predicate."""

    sas_size: int
    """The size of the SAS predicate."""

    arguments: Sequence[str]
    """The arguments of the SAS predicate."""

    arguments_str: str
    """A string format for the arguments of the SAS predicate."""

    all_bool_predicates: FrozenSet[SProposition]
    """All the Boolean propositions of the SAS predicate."""

    all_neg_bool_predicates: FrozenSet[SProposition]
    """All the negated Boolean propositions of the SAS predicate."""

    expression: Dict[int, GSBoolOutputExpression]
    """The expressions for SAS assignments, which is a mapping from integers to Boolean expressions."""

    def compile(self) -> Callable[[SState], SState]:
        """Compile the SAS assignment expression to a function.

        .. warning::

            This method is deprecated and will be removed in the future. Use the :meth:`to_conditional_assignments` or :meth:`relax` methods instead.

        Returns:
            the compiled function.
        """
        warnings.warn('SAS assignments are not supported. Run to_conditional_assignments() before compilation.', RuntimeWarning)
        compiled_expressions = tuple((i, e.compile()) for i, e in self.expression.items())

        def function(
            state: SState, *,
            compiled_expressions=compiled_expressions,
            sas_name=self.sas_name, arguments_str=self.arguments_str, all_bool_predicates=self.all_bool_predicates,
        ) -> SState:
            new_value = None
            for k, v in compiled_expressions:
                if v(state):
                    new_value = k
                    break
            if new_value is not None:
                current_value = state.intersection(all_bool_predicates)
                diff: FrozenSet[SProposition] = state - all_bool_predicates
                current_value_not: Set[SProposition] = set()
                for v in current_value:
                    a = v.split()
                    a[0] = f'{a[0]}_not'
                    current_value_not.add(' '.join(a))
                new_value_set: Set[SProposition] = {f'{sas_name}@{new_value} {arguments_str}'}

                state: SState = SState(diff | current_value_not | new_value_set)
                return state
            return state
        return function

    def iter_propositions(self) -> Iterable[SProposition]:
        yield from self.all_bool_predicates
        for v in self.expression.values():
            yield from v.iter_propositions()

    def filter_propositions(self, propositions: Set[SProposition], state: Optional[SState] = None) -> 'GSSASAssignExpression':
        return GSSASAssignExpression(
            self.sas_name, self.sas_size, self.arguments,
            expression={k: v.filter_propositions(propositions, state=state) for k, v in self.expression.items()}
        )

    def to_conditional_assignments(self) -> List[GSVariableAssignmentExpression]:
        """Convert the SAS assignment to a list of conditional assignments.

        Returns:
            a list of conditional assignments.
        """
        rvs = list()
        for k, v in self.expression.items():
            this_add = {f'{self.sas_name}@{k} {self.arguments_str}'} | self.all_neg_bool_predicates - {f'{self.sas_name}@{k}_not {self.arguments_str}'}
            this_del = self.all_bool_predicates | {f'{self.sas_name}@{k}_not {self.arguments_str}'} - {f'{self.sas_name}@{k} {self.arguments_str}'}
            rvs.append(GSConditionalAssignExpression(v, GSSimpleBoolAssignExpression(this_add, this_del)))
        return rvs

    def relax(self) -> List['GSConditionalAssignExpression']:
        """Delete relaxation of a SAS assignment. Essentially, it removes all delete effects for the inner simple assignment expression.
        Note that this method returns a list of conditional assignments, instead of a single SASAssignment.

        Returns:
            a list of delete-relaxed SAS assignment expression.
        """
        rvs = list()
        for k, v in self.expression.items():
            add_effects = {f'{self.sas_name}@{k} {self.arguments_str}'} | {f'{self.sas_name}@{i}_not {self.arguments_str}' for i in range(self.sas_size) if i != k}
            rvs.append(GSConditionalAssignExpression(v, GSSimpleBoolAssignExpression(add_effects, set())))
        return rvs

    def __str__(self) -> str:
        expression_str = jacinle.stformat(self.expression).rstrip()
        return f'SAS[target={self.sas_name} {self.arguments_str}, value={expression_str}]'


class GStripsDerivedPredicate(GSExpression):
    """Grounded STRIPS version of derived predicates."""

    def __init__(
        self, name: str, arguments: Sequence[str],
        expression_true: Optional[GSBoolOutputExpression] = None, expression_false: Optional[GSBoolOutputExpression] = None, is_relaxed: bool = False,
        effects: Optional[Iterable[GSConditionalAssignExpression]] = None
    ):
        """Initialize the grounded derived predicate.

        Args:
            name: the name of the derived predicate.
            arguments: the arguments of the derived predicate.
            expression_true: the expression for the true case. Optional if ``effects`` is provided.
            expression_false: the expression for the false case. Optional if ``effects`` is provided.
            is_relaxed: whether the derived predicate has been delete-relaxed.
            effects: the effects of the derived predicate. Optional if ``expression_true`` and ``expression_false`` are provided.
        """

        super().__init__()
        self.name = name
        self.arguments = tuple(arguments)

        self.pos_name = self.name + ' ' + ' '.join(str(x) for x in self.arguments)
        self.neg_name = self.name + '_not ' + ' '.join(str(x) for x in self.arguments)

        if effects is None:
            assert expression_true is not None and expression_false is not None
            self.pos_effect = GSConditionalAssignExpression(expression_true, GSSimpleBoolAssignExpression({self.pos_name}, {self.neg_name} if not is_relaxed else set()))
            self.neg_effect = GSConditionalAssignExpression(expression_false, GSSimpleBoolAssignExpression({self.neg_name}, {self.pos_name} if not is_relaxed else set()))
            self.effects = (self.pos_effect, self.neg_effect)
        else:
            self.effects = tuple(effects)
            assert len(self.effects) == 2

    name: str
    """The name of the derived predicate."""

    arguments: Tuple[str, ...]
    """The arguments of the derived predicate."""

    pos_name: str
    """The name of the positive proposition corresponding to this derived predicate."""

    neg_name: str
    """The name of the negative proposition corresponding to this derived predicate."""

    pos_effect: GSConditionalAssignExpression
    """The positive effect of the derived predicate."""

    neg_effect: GSConditionalAssignExpression
    """The negative effect of the derived predicate."""

    effects: Tuple[GSConditionalAssignExpression, ...]
    """A tuple of the positive and negative effects of the derived predicate."""

    def compile(self) -> Callable[[SState], SState]:
        effects_func = tuple(effect.compile() for effect in self.effects)

        def function(state: SState, effects_func=effects_func) -> SState:
            for func in effects_func:
                state = func(state)
            return state
        return function

    def iter_propositions(self) -> Iterable[SProposition]:
        for eff in self.effects:
            yield from eff.iter_propositions()

    def filter_propositions(self, propositions: Set[SProposition], state: Optional[SState] = None) -> 'GStripsDerivedPredicate':
        return GStripsDerivedPredicate(
            self.name,
            self.arguments,
            effects=[eff.filter_propositions(propositions, state=state) for eff in self.effects]
        )

    def relax(self) -> 'GStripsDerivedPredicate':
        """Delete relaxation of a derived predicate. Essentially, it removes all delete effects for the inner simple assignment expression."""
        return GStripsDerivedPredicate(self.name, self.arguments, effects=[eff.relax() for eff in self.effects])

    def __str__(self) -> str:
        effects_str = '\n'.join(jacinle.indent_text(str(eff)) for eff in self.effects)
        return f'DERIVED[\n{effects_str}\n]'


# @jacinle.log_function
def _compose_strips_classifiers_inner(classifiers: Sequence[GSBoolOutputExpression], is_disjunction: Optional[bool] = False) -> GSBoolOutputExpression:
    new_classifiers = list()
    visited = [False for _ in classifiers]

    for i in range(len(classifiers)):
        if not visited[i]:
            c = classifiers[i]
            if gs_is_constant_true(c):
                visited[i] = True
                if is_disjunction:
                    return GSBoolConstantExpression(True)
            elif gs_is_constant_false(c):
                visited[i] = True
                if not is_disjunction:
                    return GSBoolConstantExpression(False)
            elif c == GS_OPTIMISTIC_STATIC_OBJECT:
                visited[i] = True

    new_set = set()

    def add_simple_classifier(c: GSSimpleBoolExpression):
        if c.is_disjunction == is_disjunction or len(c.propositions) == 1:
            new_set.update(c.propositions)
            return True
        return False

    for i in range(len(classifiers)):
        if not visited[i]:
            c = classifiers[i]
            if isinstance(c, GSSimpleBoolExpression):
                visited[i] = add_simple_classifier(c)

    complex = list()
    for i in range(len(classifiers)):
        if not visited[i]:
            c = classifiers[i]
            assert isinstance(c, (GSComplexBoolExpression, GSSimpleBoolExpression))
            if c.is_disjunction == is_disjunction:
                assert isinstance(c, GSComplexBoolExpression)
                for e in c.expressions:
                    if isinstance(e, GSSimpleBoolExpression):
                        if add_simple_classifier(e):
                            continue
                        else:
                            complex.append(e)
                    else:
                        complex.append(e)
            else:
                complex.append(c)

    if len(new_set) > 0:
        new_classifiers.append(GSSimpleBoolExpression(new_set, is_disjunction))
    new_classifiers.extend(complex)

    if len(new_classifiers) == 0:
        return GSBoolConstantExpression(True if not is_disjunction else False)
    elif len(new_classifiers) == 1:
        return new_classifiers[0]
    else:
        return GSComplexBoolExpression(new_classifiers, is_disjunction)


def gs_compose_bool_expressions(
    expressions: Union[Sequence[Tuple[GSBoolOutputExpression, Set[SProposition]]], Sequence[GSBoolOutputExpression]],
    is_disjunction: Optional[bool] = False,
    propagate_implicit_propositions: bool = True
) -> Union[Tuple[GSBoolOutputExpression, Set[SProposition]], GSBoolOutputExpression]:
    """Compose a list of Boolean expressions into a single expression, by taking the conjunction or disjunction of them.
    This function will automatically handles possible merging of simple expressions to reduce the complexity of the final expression.
    For example, if we are taking the conjunction of two simple conjunctive expressions, then the two expressions will be merged into one single conjunction.

    This function takes an additional argument `propagate_implicit_propositions`, which controls whether the implicit propositions should be propagated to the final expression.
    Specifically, when `propagate_implicit_propositions` is set to `True`, the argument `expression` should be a list of tuples,
    where the first element is the Boolean expression and the second element is the set of implicit propositions.
    In this case, the return will also be a tuple, where the first element is the final expression and the second element is the set of implicit propositions.

    When `propagate_implicit_propositions` is set to `False`, the argument `expression` should be a list of Boolean expressions,
    and the return is also a single Boolean expression.

    Args:
        expressions: a list of Boolean expressions.
        is_disjunction: whether to take disjunction of the expressions, default to False (conjunction).
        propagate_implicit_propositions: whether to keep the implicit propositions in the final expression, default to True.
    """
    if propagate_implicit_propositions:
        rv = _compose_strips_classifiers_inner([c[0] for c in expressions], is_disjunction)
        return rv, set.union(*[c[1] for c in expressions]) if len(expressions) > 0 else set()
    else:
        rv = _compose_strips_classifiers_inner(expressions, is_disjunction)
        return rv


def gs_is_constant_true(classifier: GSBoolOutputExpression) -> bool:
    """Check if the given classifier is a constant true expression."""
    return isinstance(classifier, GSBoolConstantExpression) and classifier.constant


def gs_is_constant_false(classifier: GSBoolOutputExpression) -> bool:
    """Check if the given classifier is a constant false expression."""
    return isinstance(classifier, GSBoolConstantExpression) and not classifier.constant


def gs_is_empty_bool_expression(expression: GSBoolOutputExpression) -> bool:
    """Check if the given classifier is an empty Boolean expression."""
    return isinstance(expression, GSSimpleBoolExpression) and len(expression.propositions) == 0


def gs_is_simple_conjunctive_classifier(classifier: GSBoolOutputExpression) -> bool:
    """Check if the given classifier is a simple conjunctive classifier."""
    return isinstance(classifier, GSSimpleBoolExpression) and not classifier.is_disjunction

