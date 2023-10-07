#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : strips_grounding.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/19/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import itertools
from typing import Optional, Union, Callable, Iterable, Sequence, Iterator, Tuple, List, Set, Dict

import jacinle
from jacinle.utils.printing import indent_text

import concepts.dsl.expression as E
from concepts.dsl.dsl_types import BOOL
from concepts.dsl.tensor_value import TensorValue
from concepts.dsl.constraint import OptimisticValue
from concepts.dsl.expression import Expression, ExpressionDefinitionContext
from concepts.dsl.executors.tensor_value_executor import compose_bvdict_args

from concepts.pdsketch.predicate import Predicate, flatten_expression, split_simple_bool
from concepts.pdsketch.domain import Domain, State
from concepts.pdsketch.operator import OperatorApplier, gen_all_partially_grounded_actions, RegressionRuleApplier, gen_all_grounded_regression_rules
from concepts.pdsketch.executor import PDSketchExecutor

from concepts.pdsketch.strips.strips_expression import SPredicateName, SProposition, SState, SStateCompatible, make_sproposition

from concepts.pdsketch.strips.strips_grounded_expression import GSBoolOutputExpression, GSVariableAssignmentExpression, GSBoolForwardDiffReturn
from concepts.pdsketch.strips.strips_grounded_expression import GS_OPTIMISTIC_STATIC_OBJECT, GSOptimisticStaticObjectType
from concepts.pdsketch.strips.strips_grounded_expression import GSBoolConstantExpression, GSSimpleBoolExpression, gs_compose_bool_expressions, gs_is_empty_bool_expression
from concepts.pdsketch.strips.strips_grounded_expression import GSSimpleBoolAssignExpression, GSConditionalAssignExpression, GStripsDerivedPredicate


__all__ = [
    'GStripsOperator', 'GStripsProblem',
    'GStripsTranslatorBase', 'GStripsTranslatorOptimistic',
    'relevance_analysis_v1', 'relevance_analysis_v2'
]


class GStripsOperator(object):
    """A grounded STRIPS operator, the STRIPS version of the OperatorApplier.
    By grounded we mean that all the preconditions / effects are propositions."""

    def __init__(
        self,
        precondition: GSBoolOutputExpression,
        effects: Sequence[GSVariableAssignmentExpression],
        raw_operator: OperatorApplier,
        implicit_propositions: Optional[Set[SProposition]] = None
    ):
        self.precondition = precondition
        self.implicit_propositions = implicit_propositions if implicit_propositions is not None else set()
        self.effects = tuple(effects)
        self.raw_operator = raw_operator
        self.precondition_func: Optional[Callable[[SState], bool]] = None
        self.effects_func: Optional[Tuple[Callable[[SState], SState], ...]] = None

    precondition: GSBoolOutputExpression
    """The precondition for the operator."""

    effects: Tuple[GSVariableAssignmentExpression, ...]
    """The effects for the operator."""

    implicit_propositions: Set[SProposition]
    """Implicit preconditions."""

    raw_operator: OperatorApplier
    """The raw operator.
    Note that since the strips operator is grounded, the raw operator is the :class:`concepts.pdsketch.operator.OperatorApplier` instance."""

    precondition_func: Optional[Callable[[SState], bool]]
    """An compiled version of the precondition tester."""

    effects_func: Optional[Tuple[Callable[[SState], SState], ...]]
    """A list of compiled version of the effect appliers."""

    def compile(self):
        """Generate the compiled version of the operator (preconditions and effects)."""
        self.precondition_func = self.precondition.compile()
        self.effects_func = tuple(e.compile() for e in self.effects)

    def applicable(self, state: Union[SState, Set[SProposition]]) -> Union[bool, GSBoolForwardDiffReturn]:
        """Return whether the operator is applicable to the state.

        Args:
            state: the current state.

        Returns:
            True if the operator is applicable, False otherwise. When the `FORWARD_DIFF` mode is enabled for :class:`~concepts.pdsketch.strips.strips_grounded_expression.GSBoolExpression`,
            this function will return a :class:`~concepts.pdsketch.strips.strips_grounded_expression.GSBoolForwardDiffReturn` object.
        """
        return self.precondition_func(state)

    def apply(self, state: SState) -> SState:
        """Apply the operator to the state. Note that this function does not check the precondition. To do that, use :meth:`applicable` before calling this function.

        Args:
             state: the current state.

        Returns:
             the new state after applying the operator.
        """
        for effect_func in self.effects_func:
            state = effect_func(state)
        return state

    def iter_propositions(self) -> Iterable[SProposition]:
        """Iterate over all the propositions used in the operator, including both preconditions and effects."""
        yield from self.precondition.iter_propositions()
        for e in self.effects:
            yield from e.iter_propositions()

    def filter_propositions(self, propositions: Set[SProposition], state: Optional[SState] = None) -> 'GStripsOperator':
        """Filter the propositions in this expression. Only the propositions in the given set will be kept. Note that this function also takes a state as input,
        essentially, the state is the initial state of the environment, and the `propositions` contains all propositions that will be possibly changed
        by actions. See :meth:`~concepts.pdsketch.strips.strips_grounded_expression.GSExpression.filter_propositions` for more details.

        Args:
            propositions: the propositions to keep.
            state: the initial state for filtering the propositions.

        Returns:
            a new operator with filtered propositions.
        """

        return GStripsOperator(
            self.precondition.filter_propositions(propositions, state=state),
            [e.filter_propositions(propositions, state=state) for e in self.effects],
            self.raw_operator
        )

    def __str__(self) -> str:
        return (
            type(self).__name__
            + '{' + f'{self.raw_operator}' + '}'
            + '{\n' + indent_text(self.precondition)
            + '\n' + '\n'.join(indent_text(x) for x in self.effects) if len(self.effects) > 0 else '<Empty Effect>'
            + '\n}'
        )

    def __repr__(self) -> str:
        return self.__str__()


class GStripsAchieveExpression(object):
    def __init__(self, goal: GSBoolOutputExpression, maintains: GSBoolOutputExpression):
        self.goal = goal
        self.maintains = maintains

    goal: GSBoolOutputExpression
    maintains: GSBoolOutputExpression


class GStripsRegressionRule(object):
    def __init__(self, precondition: GSBoolOutputExpression, preconstraints: GSBoolOutputExpression, goal: GSBoolOutputExpression, body: Sequence[Union[GStripsOperator, GStripsAchieveExpression]]):
        self.precondition = precondition
        self.preconstraints = preconstraints
        self.goal = goal
        self.body = tuple(body)

    precondition: GSBoolOutputExpression
    preconstraints: GSBoolOutputExpression
    goal: GSBoolOutputExpression
    body: Tuple[Union[GStripsOperator, GStripsAchieveExpression], ...]


class GStripsProblem(object):
    """A grounded STRIPS task. It is composed of three parts:

    - The current state, which is a set of propositions.
    - The goal state, which is a grounded STRIPS classifier.
    - The operators, which is a list of applicable grounded STRIPS operators.
    """

    def __init__(
        self,
        state: SStateCompatible,
        goal: GSBoolOutputExpression,
        operators: Sequence[GStripsOperator],
        is_relaxed: bool = False,
        goal_implicit_propositions: Optional[Set[SProposition]] = None,
        regression_rules: Optional[Sequence[GStripsRegressionRule]] = None,
        derived_predicates: Optional[Sequence[GStripsDerivedPredicate]] = None,
        facts: Optional[Set[SProposition]] = None,
    ):
        self.state = SState(state)
        self.goal = goal
        self.operators = tuple(operators)
        self.regression_rules = tuple(regression_rules) if regression_rules is not None else tuple()
        self.derived_predicates = tuple(derived_predicates) if derived_predicates is not None else tuple()
        self.is_relaxed = is_relaxed
        self.goal_implicit_propositions = goal_implicit_propositions if goal_implicit_propositions is not None else set()
        self._facts = facts

    state: SState
    """The initial state."""

    goal: GSBoolOutputExpression
    """The goal expression."""

    operators: Tuple[GStripsOperator]
    """The list of operators."""

    regression_rules: Tuple[GStripsRegressionRule]
    """The list of regression rules."""

    derived_predicates: Tuple[GStripsDerivedPredicate]
    """The list of derived predicates."""

    is_relaxed: bool
    """Whether the task is a delete-relaxed task."""

    goal_implicit_propositions: Set[SProposition]
    """The implicit propositions in the goal."""

    def compile(self) -> 'GStripsProblem':
        """Compile all operators and derived predicates.

        Returns:
            self
        """
        for op in self.operators:
            op.compile()
        for op in self.derived_predicates:
            op.compile()
        return self

    @property
    def facts(self):
        """The relevant facts in the task."""
        return self._facts

    def __str__(self):
        operator_str = '\n'.join(str(op) for op in self.operators)
        derived_predicate_str = '\n'.join(str(dp) for dp in self.derived_predicates)
        return f"""{type(self).__name__}{{
  state: {self.state}
  goal: {self.goal}
  operators:
    {indent_text(operator_str, 2).strip()}
  derived_predicates:
    {indent_text(derived_predicate_str, 2).strip()}
  facts: {self._facts}
}}"""

    def __repr__(self) -> str:
        return self.__str__()


class GStripsTranslatorBase(object):
    def __init__(self, executor: PDSketchExecutor, use_string_name: bool = True, prob_goal_threshold: float = 0.5, use_derived_predicates: bool = False, use_regression_rules: bool = True):
        assert isinstance(executor, PDSketchExecutor)
        self._executor = executor
        self._use_string_name = use_string_name
        self._prob_goal_threshold = prob_goal_threshold
        self._use_derived_predicates = use_derived_predicates
        self._use_regression_rules = use_regression_rules
        self._predicate2index = dict()
        self._init_indices()

    @property
    def executor(self) -> PDSketchExecutor:
        """The executor for the domain."""
        return self._executor

    @property
    def domain(self) -> Domain:
        """The domain."""
        return self._executor.domain

    @property
    def predicate2index(self) -> Dict[Tuple[SPredicateName, str], int]:
        """The mapping from predicate names to index."""
        return self._predicate2index

    def _init_indices(self):
        raise NotImplementedError()

    def compile_expr(self, expr: Union[str, Expression], state: State) -> Tuple[GSBoolOutputExpression, Set[SProposition]]:
        raise NotImplementedError()

    def compile_operator(self, op: OperatorApplier, state: State, is_relaxed=False) -> GStripsOperator:
        raise NotImplementedError()

    def compile_regression_rule(self, op: OperatorApplier, state: State) -> GStripsRegressionRule:
        raise NotImplementedError()

    def compile_derived_predicate(self, dp: Predicate, state: State, is_relaxed=False) -> List[GStripsDerivedPredicate]:
        raise NotImplementedError()

    def compile_state(self, state: State, forward_derived: bool = False) -> SState:
        raise NotImplementedError()

    def relevance_analysis(self, task: GStripsProblem, relaxed_relevance: bool = False, forward: bool = True, backward: bool = True) -> GStripsProblem:
        raise NotImplementedError()

    def define_grounded_predicate(self, name: str, modifier: Optional[str] = None) -> int:
        """Allocate a new identifier for the predicate (with modifier).

        Args:
            name: the name of the predicate.
            modifier: an optional modifier (e.g., not)

        Returns:
            the index of the predicate.
        """
        if not self._use_string_name:
            identifier = len(self.predicate2index)
        else:
            identifier = name + (f'_{modifier}' if modifier is not None else '')
        self.predicate2index[(name, modifier)] = identifier
        return identifier

    def get_grounded_predicate_indentifier(self, name: str, modifier: Optional[str] = None):
        return self.predicate2index[(name, modifier)]

    def compile_task(
        self,
        state: State,
        goal_expr: Union[str, Expression],
        actions: Optional[Sequence[OperatorApplier]] = None,
        regression_rules: Optional[Sequence[RegressionRuleApplier]] = None,
        is_relaxed: bool = False,
        forward_relevance_analysis: bool = True,
        backward_relevance_analysis: bool = True,
        verbose: bool = False
    ) -> GStripsProblem:
        """Compile a grounded STRIPS task.

        Args:
            state: the initial state.
            goal_expr: the goal expression.
            actions: the list of actions. If not specified, all actions in the domain will be used.
            is_relaxed: whether the task is a delete-relaxed task.
            forward_relevance_analysis: whether to perform forward relevance analysis.
            backward_relevance_analysis: whether to perform backward relevance analysis.
            verbose: whether to print verbose information.

        Returns:
            the compiled task.
        """
        with jacinle.cond_with(jacinle.time('compile_task::actions'), verbose):
            if actions is None:
                actions = gen_all_partially_grounded_actions(self._executor, state, filter_static=True)
        with jacinle.cond_with(jacinle.time('compile_task::regression_rules'), verbose):
            if regression_rules is None:
                regression_rules = gen_all_grounded_regression_rules(self._executor, state, filter_static=True)

        with jacinle.cond_with(jacinle.time('compile_task::state'), verbose):
            strips_state = self.compile_state(state)
        with jacinle.cond_with(jacinle.time('compile_task::operators'), verbose):
            strips_operators = [self.compile_operator(op, state, is_relaxed=is_relaxed) for op in actions]
        derived_predicates = list()
        if self._use_derived_predicates:
            with jacinle.cond_with(jacinle.time('compile_task::derived_predicates'), verbose):
                for pred in self.domain.functions.values():
                    if not pred.is_state_variable and pred.is_cacheable and pred.return_type == BOOL and not pred.is_static:
                        derived_predicates.extend(self.compile_derived_predicate(pred, state, is_relaxed=is_relaxed))
        regression_rules = list()
        with jacinle.cond_with(jacinle.time('compile_task::regression_rules'), verbose):
            strips_regression_rules = [self.compile_regression_rule(op, state) for op in regression_rules]
        with jacinle.cond_with(jacinle.time('compile_task::goal'), verbose):
            strips_goal, strips_goal_ip = self.compile_expr(goal_expr, state)
        task = GStripsProblem(strips_state, strips_goal, strips_operators, regression_rules=strips_regression_rules, is_relaxed=is_relaxed, goal_implicit_propositions=strips_goal_ip, derived_predicates=derived_predicates)
        with jacinle.cond_with(jacinle.time('compile_task::relevance_analysis'), verbose):
            if forward_relevance_analysis or backward_relevance_analysis:
                task = self.relevance_analysis(task, forward=forward_relevance_analysis, backward=backward_relevance_analysis)
        return task.compile()

    def recompile_relaxed_task(self, task: GStripsProblem, forward_relevance_analysis: bool = True, backward_relevance_analysis: bool = True) -> GStripsProblem:
        """Recompile a task to a delete-relaxed task.

        Args:
            task: the task to be recompiled.
            forward_relevance_analysis: whether to perform forward relevance analysis.
            backward_relevance_analysis: whether to perform backward relevance analysis.

        Returns:
            the recompiled task.
        """
        new_operators = list(task.operators)
        for i, op in enumerate(new_operators):
            new_effects = []
            for e in op.effects:
                new_e = e.relax()
                if isinstance(new_e, (tuple, list)):  # Note that GSSASAssignment will be relaxed into a tuple.
                    new_effects.extend(new_e)
                else:
                    new_effects.append(new_e)
            new_operators[i] = GStripsOperator(op.precondition, new_effects, op.raw_operator)
        task = GStripsProblem(task.state, task.goal, new_operators, is_relaxed=True, derived_predicates=[dp.relax() for dp in task.derived_predicates], facts=task.facts)
        if forward_relevance_analysis or backward_relevance_analysis:
            task = self.relevance_analysis(task, forward=forward_relevance_analysis, backward=backward_relevance_analysis)
        return task.compile()

    def recompile_task_new_state(
        self,
        task: GStripsProblem, new_state: Union[State, SStateCompatible],
        forward_relevance_analysis: bool = True, backward_relevance_analysis: bool = True,
        forward_derived: bool = False
    ) -> GStripsProblem:
        """
        Compile a new GStripsTask from a new state.

        Args:
            task: the original task.
            new_state: the new state.
            forward_relevance_analysis: whether to perform forward relevance analysis. Defaults to True.
            backward_relevance_analysis: whether to perform backward relevance analysis. Defaults to True.
            forward_derived: whether to forward derived predicates. Defaults to False.

        Returns:
            the new task.
        """
        if isinstance(new_state, State):
            new_state = self.compile_state(new_state.clone(), forward_derived=forward_derived)
        if task.facts is not None:
            new_state = new_state & task.facts
        task = GStripsProblem(new_state, task.goal, task.operators, is_relaxed=task.is_relaxed, derived_predicates=task.derived_predicates, facts=task.facts)
        if forward_relevance_analysis or backward_relevance_analysis:
            task = self.relevance_analysis(task, forward=forward_relevance_analysis, backward=backward_relevance_analysis)
            return task.compile()
        return task


class GStripsTranslatorOptimistic(GStripsTranslatorBase):
    def _init_indices(self):
        for pred in _find_cached_predicates(self.domain):
            if pred.return_type == BOOL:
                self.define_grounded_predicate(pred.name)
                self.define_grounded_predicate(pred.name, 'not')
            else:
                self.define_grounded_predicate(pred.name, 'initial')
                self.define_grounded_predicate(pred.name, 'optimistic')

    def compose_grounded_predicate(
        self, predicate_app: E.FunctionApplicationExpression,
        negated: bool = False, optimistic: Optional[bool] = None, allow_set: bool = False, return_argument_indices: bool = False
    ) -> Union[SProposition, Tuple[SProposition, List[int]]]:
        state = self.executor.state
        arguments = list()
        for arg_index, arg in enumerate(predicate_app.arguments):
            assert isinstance(arg, (E.ObjectConstantExpression, E.VariableExpression))
            if isinstance(arg, E.ObjectConstantExpression):
                arg = state.get_typed_index(arg.name)
                if allow_set:
                    arg = [arg]
            else:
                if arg.variable.name == '??':
                    assert allow_set
                    arg = list(range(state.get_nr_objects_by_type(predicate_app.function.arguments[arg_index].typename)))
                else:
                    arg = self.executor.get_bounded_variable(arg.variable).index
                    if allow_set:
                        arg = [arg]
            assert isinstance(arg, list) if allow_set else isinstance(arg, int)
            arguments.append(arg)

        if predicate_app.return_type == BOOL:
            assert optimistic is None
            if allow_set:
                rv = set(
                    _format_proposition((self.get_grounded_predicate_indentifier(predicate_app.function.name, 'not' if negated else None),) + tuple(args))
                    for args in itertools.product(*arguments)
                )
            else:
                rv = _format_proposition((self.get_grounded_predicate_indentifier(predicate_app.function.name, 'not' if negated else None),) + tuple(arguments))
        else:
            assert not negated and optimistic is not None
            modifier = 'optimistic' if optimistic else 'initial'
            if allow_set:
                rv = set(
                    _format_proposition((self.get_grounded_predicate_indentifier(predicate_app.function.name, modifier),) + tuple(args))
                    for args in itertools.product(*arguments)
                )
            else:
                rv = _format_proposition((self.get_grounded_predicate_indentifier(predicate_app.function.name, modifier),) + tuple(arguments))

        if return_argument_indices:
            return rv, arguments
        return rv

    # @jacinle.log_function
    def compose_bool_expression(self, expr: E.ValueOutputExpression, negated: bool = False) -> Tuple[Union[GSBoolOutputExpression, GSOptimisticStaticObjectType], Set[SProposition]]:
        state = self.executor.state
        predicate_app, this_negated = split_simple_bool(expr, initial_negated=negated)
        if predicate_app is not None:
            # jacinle.log_function.print(predicate_app, this_negated)
            if predicate_app.function.is_static:
                if predicate_app.return_type == BOOL:
                    if predicate_app.function.name in state.features:
                        _, arguments = self.compose_grounded_predicate(predicate_app, this_negated, return_argument_indices=True)
                        init_value = state.features[predicate_app.function.name][tuple(arguments)]
                    else:
                        init_value = self.executor.execute(expr)
                    return GSBoolConstantExpression(bool(init_value) ^ this_negated), set()
                else:
                    return GS_OPTIMISTIC_STATIC_OBJECT, set()

            if predicate_app.return_type == BOOL:
                return GSSimpleBoolExpression(self.compose_grounded_predicate(predicate_app, this_negated, allow_set=True), is_disjunction=True), set()
            else:
                return GSSimpleBoolExpression(self.compose_grounded_predicate(predicate_app, this_negated, optimistic=True, allow_set=True), is_disjunction=True), set()
        elif E.is_not_expr(expr):
            return self.compose_bool_expression(expr.arguments[0], negated=not negated)
        elif E.is_and_expr(expr) and not negated or E.is_or_expr(expr) and negated:
            classifiers = [self.compose_bool_expression(e, negated=negated) for e in expr.arguments]
            rv = gs_compose_bool_expressions(classifiers, is_disjunction=False)
            return rv
        elif E.is_and_expr(expr) and negated or E.is_or_expr(expr) and not negated:
            classifiers = [self.compose_bool_expression(e, negated=negated) for e in expr.arguments]
            return gs_compose_bool_expressions(classifiers, is_disjunction=True)
        elif E.is_forall_expr(expr) and not negated or E.is_exists_expr(expr) and negated:
            classifiers = list()
            for index in range(state.get_nr_objects_by_type(expr.variable.typename)):
                with self.executor.new_bounded_variables({expr.variable: index}):
                    classifiers.append(self.compose_bool_expression(expr.expr, negated=negated))
            return gs_compose_bool_expressions(classifiers, is_disjunction=False)
        elif E.is_forall_expr(expr) and negated or E.is_exists_expr(expr) and not negated:
            classifiers = list()
            for index in range(state.get_nr_objects_by_type(expr.variable.typename)):
                with self.executor.new_bounded_variables({expr.variable: index}):
                    classifiers.append(self.compose_bool_expression(expr.expression, negated=negated))
            return gs_compose_bool_expressions(classifiers, is_disjunction=True)
        elif isinstance(expr, E.DeicticSelectExpression):
            classifiers = list()
            for index in range(state.get_nr_objects_by_type(expr.variable.typename)):
                with self.executor.new_bounded_variables({expr.variable: index}):
                    classifiers.append(self.compose_bool_expression(expr.expr, negated=negated))
            return gs_compose_bool_expressions(classifiers, is_disjunction=True)
        elif isinstance(expr, E.ConditionalSelectExpression):
            classifiers = [
                self.compose_bool_expression(expr.condition),
                self.compose_bool_expression(expr.predicate)
            ]
            return gs_compose_bool_expressions(classifiers, is_disjunction=False)
        elif isinstance(expr, E.PredicateEqualExpression):
            argument_values = [self.compose_bool_expression(arg) for arg in [expr.predicate, expr.value]]
            has_optimistic_object = any(c[0] == GS_OPTIMISTIC_STATIC_OBJECT for c in argument_values)
            if has_optimistic_object:
                if expr.return_type == BOOL:
                    return GSBoolConstantExpression(True), _extract_all_propositions(argument_values)
                else:
                    return GS_OPTIMISTIC_STATIC_OBJECT, _extract_all_propositions(argument_values)

            argument_values = [argv for argv in argument_values if not gs_is_empty_bool_expression(argv)]
            init_value = self.executor.execute(expr)
            if init_value.item() > self._prob_goal_threshold and not negated or init_value.item() < 1 - self._prob_goal_threshold and negated:
                return GSBoolConstantExpression(True), _extract_all_propositions(argument_values)
            else:
                return gs_compose_bool_expressions(argument_values, is_disjunction=True)
        elif isinstance(expr, E.FunctionApplicationExpression):
            argument_values = [self.compose_bool_expression(arg) for arg in expr.arguments]
            # jacinle.log_function.print(argument_values)
            if GS_OPTIMISTIC_STATIC_OBJECT in argument_values:
                if expr.return_type == BOOL:
                    return GSBoolConstantExpression(True), _extract_all_propositions(argument_values)
                else:
                    return GS_OPTIMISTIC_STATIC_OBJECT, _extract_all_propositions(argument_values)

            argument_values = [argv for argv in argument_values if not gs_is_empty_bool_expression(argv)]
            # jacinle.log_function.print('computing initial value.')
            if expr.return_type == BOOL:
                # Theoretically, we can compute these values bottom-up together with the transformation.
                # In practice, this requires much more code to do...
                init_value = self.executor.execute(expr)
                # jacinle.log_function.print('computed initial value:', init_value)
                if init_value.item() > self._prob_goal_threshold and not negated or init_value.item() < 1 - self._prob_goal_threshold and negated:
                    return GSBoolConstantExpression(True), _extract_all_propositions(argument_values)
                else:
                    return gs_compose_bool_expressions(argument_values, is_disjunction=True)
            else:
                return gs_compose_bool_expressions(argument_values, is_disjunction=True)
        elif isinstance(expr, E.VariableExpression):
            assert expr.return_type != BOOL
            assert isinstance(self.executor.get_bounded_variable(expr.variable), TensorValue), 'Most likely you are accessing a non-optimistic object.'
            assert isinstance(self.executor.get_bounded_variable(expr.variable).single_elem(), OptimisticValue)
            return GS_OPTIMISTIC_STATIC_OBJECT, set()
        else:
            raise TypeError('Unsupported expression grounding: {}.'.format(expr))

    def compose_grounded_assignment(self, assignments: Sequence[E.VariableAssignmentExpression], is_relaxed: bool = False) -> Tuple[List[GSVariableAssignmentExpression], Set[SProposition]]:
        state = self.executor.state
        add_effects = set()
        del_effects = set()
        implicit_propositions = set()
        outputs = list()
        for assign_expr in assignments:
            if isinstance(assign_expr, E.AssignExpression):
                feat = assign_expr.predicate
                if feat.return_type == BOOL:
                    assert E.is_constant_bool_expr(assign_expr.value)
                    if assign_expr.value.constant.item():
                        add_effects.add(self.compose_grounded_predicate(feat, negated=False))
                        if not is_relaxed:
                            del_effects.add(self.compose_grounded_predicate(feat, negated=True))
                    else:
                        add_effects.add(self.compose_grounded_predicate(feat, negated=True))
                        if not is_relaxed:
                            del_effects.add(self.compose_grounded_predicate(feat, negated=False))
                else:
                    # For customized feature types, the "feat(...)" means that "this state variable has been set to an optimistic value."
                    add_effects.add(self.compose_grounded_predicate(feat, optimistic=True))
                    if not is_relaxed:
                        del_effects.add(self.compose_grounded_predicate(feat, optimistic=False))
                    value, ip = self.compose_bool_expression(assign_expr.value)
                    implicit_propositions = ip
                    if isinstance(value, GSBoolOutputExpression):
                        implicit_propositions.update(set(value.iter_propositions()))
            elif isinstance(assign_expr, E.ConditionalAssignExpression):
                assignment, ass_ip = self.compose_grounded_assignment([E.AssignExpression(assign_expr.predicate, assign_expr.value)], is_relaxed=is_relaxed)
                condition_classifier, cond_ip = self.compose_bool_expression(assign_expr.condition)
                outputs.append(GSConditionalAssignExpression(condition_classifier, assignment[0]))
                implicit_propositions = cond_ip | ass_ip
            elif isinstance(assign_expr, E.DeicticAssignExpression):
                for index in range(state.get_nr_objects_by_type(assign_expr.variable.typename)):
                    with self.executor.new_bounded_variables({assign_expr.variable: index}):
                        assignments, ass_ip = self.compose_grounded_assignment([assign_expr.expr], is_relaxed=is_relaxed)
                        for assignment in assignments:
                            if isinstance(assignment, GSSimpleBoolAssignExpression):
                                add_effects.update(assignment.add_effects)
                                del_effects.update(assignment.del_effects)
                            else:
                                outputs.append(assignment)
                        implicit_propositions.update(ass_ip)
        if len(add_effects) > 0 or len(del_effects) > 0:
            outputs.append(GSSimpleBoolAssignExpression(add_effects, del_effects))
        return outputs, implicit_propositions

    def compile_expr(self, expr: Union[str, Expression], state: State) -> Tuple[GSBoolOutputExpression, Set[SProposition]]:
        expr = self.domain.parse(expr)
        expr = flatten_expression(expr)
        with self.executor.with_state(state):
            return self.compose_bool_expression(expr)

    def compile_operator(self, op: OperatorApplier, state: State, is_relaxed=False) -> GStripsOperator:
        # print('compile_operator:: {}'.format(op))
        if getattr(op.operator, 'flatten_precondition', None) is None:
            ctx = ExpressionDefinitionContext(*op.operator.arguments, domain=self.domain)
            precondition = E.AndExpression(*[flatten_expression(e.bool_expr, ctx=ctx) for e in op.operator.preconditions])
            op.operator.flatten_precondition = precondition
        else:
            precondition = op.operator.flatten_precondition

        if getattr(op.operator, 'flatten_effects', None) is None:
            ctx = ExpressionDefinitionContext(*op.operator.arguments, domain=self.domain)
            effects = [flatten_expression(e.assign_expr, ctx=ctx) for e in op.operator.effects]
            op.operator.flatten_effects = effects
        else:
            effects = op.operator.flatten_effects

        # print('  precondition: {}'.format(precondition))
        with self.executor.with_state(state), self.executor.with_bounded_variables(compose_bvdict_args(op.operator.arguments, op.arguments, state=state)):
            precondition, pre_ip = self.compose_bool_expression(precondition)
            effects, eff_ip = self.compose_grounded_assignment(effects, is_relaxed=is_relaxed)
        # print('  compiled precondition: {}'.format(precondition))

        return GStripsOperator(precondition, effects, op, implicit_propositions=pre_ip | eff_ip)

    def compile_state(self, state: State, forward_derived: bool = False) -> SState:
        predicates = set()
        for name, feature in state.features.items():
            if self.domain.functions[name].is_static:
                continue
            if not self.domain.functions[name].is_state_variable:
                continue
            if feature.dtype == BOOL:
                for args, v in _iter_value(feature):
                    if v > 0.5:
                        predicates.add(_format_proposition((self.get_grounded_predicate_indentifier(name),) + args))
                    else:
                        predicates.add(_format_proposition((self.get_grounded_predicate_indentifier(name, 'not'),) + args))
            else:
                for args, _ in _iter_value(feature):
                    predicates.add(_format_proposition((self.get_grounded_predicate_indentifier(name, 'initial'),) + args))
        return SState(predicates)

    def compile_derived_predicate(self, dp: Predicate, state: State, is_relaxed=False) -> List[GStripsDerivedPredicate]:
        raise NotImplementedError('Derived predicates are not supported in Optimistic GStrips translation.')

    def relevance_analysis(self, task: GStripsProblem, relaxed_relevance: bool = False, forward: bool = True, backward: bool = True) -> GStripsProblem:
        return relevance_analysis_v1(task, relaxed_relevance=relaxed_relevance, forward=forward, backward=backward)


"""
class GStripsTranslatorSAS(GStripsTranslatorBase):
    def __init__(
        self,
        session: PDSketchExecutor,
        use_string_name: Optional[bool] = True,
        prob_goal_threshold: float = 0.5,
        cache_bool_predicates: bool = False
    ):
        self.cache_bool_predicates = cache_bool_predicates
        super().__init__(session, use_string_name, prob_goal_threshold, use_derived_predicates=cache_bool_predicates)

    def _init_indices(self):
        for pred in _find_cached_predicates(self.domain, allow_cacheable_bool=self.cache_bool_predicates):
            if pred.return_type == BOOL:
                self.define_grounded_predicate(pred.name)
                self.define_grounded_predicate(pred.name, 'not')
            else:
                for i in range(pred.ao_discretization.size):
                    self.define_grounded_predicate(f'{pred.name}@{i}')
                    self.define_grounded_predicate(f'{pred.name}@{i}', 'not')

    def compose_grounded_predicate_strips(
        self, ctx: ExpressionExecutionContext, feature_app: SE.StripsBoolPredicate,
        negated: bool = False
    ) -> Union[GSSimpleBoolExpression, GSBoolConstantExpression]:
        argument_indices = list()
        for arg_index, arg in enumerate(feature_app.arguments):
            argument_indices.append(ctx.get_bounded_variable(arg))

        feature_name = feature_app.sas_name if isinstance(feature_app, SE.SSASPredicateApplicationExpression) else feature_app.name
        predicate_def = self.domain.functions[feature_name]
        if predicate_def.is_static:
            if isinstance(feature_app, SE.SSASPredicateApplicationExpression):
                value = ctx.state.features[feature_name].tensor_indices[tuple(argument_indices)]
                return GSBoolConstantExpression(value.item() == feature_app.sas_index ^ negated ^ feature_app.negated)
            else:
                value = ctx.state.features[feature_name][tuple(argument_indices)]
                return GSBoolConstantExpression((value.item() > 0.5) ^ negated ^ feature_app.negated)

        predicate_name = self.get_grounded_predicate_indentifier(feature_app.name, 'not' if negated ^ feature_app.negated else None)
        return GSSimpleBoolExpression({_format_proposition((predicate_name,) + tuple(argument_indices))})

    def compose_grounded_predicate(self, ctx: ExpressionExecutionContext, feature_app: E.FunctionApplicationExpression, negated: bool = False) -> Union[GSSimpleBoolExpression, GSBoolConstantExpression]:
        argument_indices = list()
        for arg_index, arg in enumerate(feature_app.arguments):
            if isinstance(arg, E.ObjectConstantExpression):
                arg = ctx.state.get_typed_index(arg.name)
            else:
                assert isinstance(arg, E.VariableExpression)
                arg = ctx.get_bounded_variable(arg.variable)
            argument_indices.append(arg)
        predicate_def = feature_app.function
        feature_name = predicate_def.name

        if predicate_def.is_static:
            value = ctx.state.features[feature_name][tuple(argument_indices)]
            assert value.dtype == BOOL
            return GSBoolConstantExpression((value.item() > 0.5) ^ negated)

        predicate_name = self.get_grounded_predicate_indentifier(predicate_def.name, 'not' if negated else None)
        return GSSimpleBoolExpression({_format_proposition((predicate_name,) + tuple(argument_indices))})

    def _compose_grounded_classifier_strips(self, ctx: ExpressionExecutionContext, expr: SExpression, negated: bool = False) -> Union[GSBoolExpression, SProposition, GSOptimisticStaticObjectType]:
        if isinstance(expr, SE.SBoolConstant):
            return GSBoolConstantExpression(expr.constant ^ negated)
        elif isinstance(expr, SE.SBoolNot):
            return self._compose_grounded_classifier_strips(ctx, expr.expr, not negated)
        elif isinstance(expr, SE.SBoolExpression):
            classifiers = [self._compose_grounded_classifier_strips(ctx, e, negated) for e in expr.arguments]
            if expr.is_conjunction and not negated or expr.is_disjunction and negated:
                return gs_compose_bool_expressions(classifiers, is_disjunction=False, propagate_implicit_propositions=False)
            else:
                return gs_compose_bool_expressions(classifiers, is_disjunction=True, propagate_implicit_propositions=False)
        elif isinstance(expr, SE.SQuantificationExpression):
            classifiers = list()
            for index in range(ctx.state.get_nr_objects_by_type(expr.variable.typename)):
                with ctx.new_bounded_variables({expr.variable: index}):
                    classifiers.append(self._compose_grounded_classifier_strips(ctx, expr.expr, negated))
            if expr.is_forall and not negated or expr.is_exists and negated:
                return gs_compose_bool_expressions(classifiers, is_disjunction=False, propagate_implicit_propositions=False)
            else:
                return gs_compose_bool_expressions(classifiers, is_disjunction=True, propagate_implicit_propositions=False)
        elif isinstance(expr, SE.StripsBoolPredicate):
            return self.compose_grounded_predicate_strips(ctx, expr, negated)
        else:
            raise TypeError('Unknown expression type: {}.'.format(expr))

    def compose_grounded_classifier(
        self,
        ctx: ExpressionExecutionContext,
        expr: E.ValueOutputExpression,
        negated: bool = False
    ) -> Union[GSBoolExpression, GSOptimisticStaticObjectType]:
        if isinstance(expr, E.FunctionApplicationExpression):
            predicate_def = expr.function
            assert predicate_def.is_cacheable and predicate_def.return_type == BOOL
            if predicate_def.expr is None or self.cache_bool_predicates:  # a basic predicate.
                return self.compose_grounded_predicate(ctx, expr, negated)
            else:
                return self._compose_grounded_classifier_strips(ctx, predicate_def.ao_discretization, negated)
        elif E.is_not_expr(expr):
            return self.compose_grounded_classifier(ctx, expr.arguments[0], negated=not negated)
        elif E.is_and_expr(expr) and not negated or E.is_or_expr(expr) and negated:
            classifiers = [self.compose_grounded_classifier(ctx, e, negated=negated) for e in expr.arguments]
            rv = gs_compose_bool_expressions(classifiers, is_disjunction=False, propagate_implicit_propositions=False)
            return rv
        elif E.is_and_expr(expr) and negated or E.is_or_expr(expr) and not negated:
            classifiers = [self.compose_grounded_classifier(ctx, e, negated=negated) for e in expr.arguments]
            return gs_compose_bool_expressions(classifiers, is_disjunction=True, propagate_implicit_propositions=False)
        elif E.is_forall_expr(expr) and not negated or E.is_exists_expr(expr) and negated:
            classifiers = list()
            for index in range(ctx.state.get_nr_objects_by_type(expr.variable.typename)):
                with ctx.new_bounded_variables({expr.variable: index}):
                    classifiers.append(self.compose_grounded_classifier(ctx, expr.expr, negated=negated))
            return gs_compose_bool_expressions(classifiers, is_disjunction=False, propagate_implicit_propositions=False)
        elif E.is_forall_expr(expr) and negated or E.is_exists_expr(expr) and not negated:
            classifiers = list()
            for index in range(ctx.state.get_nr_objects_by_type(expr.variable.typename)):
                with ctx.new_bounded_variables({expr.variable: index}):
                    classifiers.append(self.compose_grounded_classifier(ctx, expr.expr, negated=negated))
            return gs_compose_bool_expressions(classifiers, is_disjunction=True, propagate_implicit_propositions=False)
        else:
            raise TypeError('Unsupported expression grounding: {}.'.format(expr))

    def _compose_grounded_assignment_strips(
        self,
        ctx: ExpressionExecutionContext,
        assignments: Sequence[SE.SVariableAssignmentExpression],
        is_relaxed: bool = False
    ) -> List[GSAssignmentExpression]:
        add_effects = set()
        del_effects = set()
        outputs = list()

        for expr in assignments:
            if isinstance(expr, SE.SDeicticAssignment):
                for index in range(ctx.state.get_nr_objects_by_type(expr.variable.typename)):
                    with ctx.new_bounded_variables({expr.variable: index}):
                        this_outputs = self._compose_grounded_assignment_strips(ctx, [expr.expression], is_relaxed)
                        for ass in this_outputs:
                            if isinstance(ass, GSSimpleAssignment):
                                add_effects.update(ass.add_effects)
                                del_effects.update(ass.del_effects)
                            else:
                                outputs.append(ass)
            elif isinstance(expr, SE.SConditionalAssignment):
                assignments = self._compose_grounded_assignment_strips(ctx, [expr.assign_op], is_relaxed)
                condition = self._compose_grounded_classifier_strips(ctx, expr.condition)
                for ass in assignments:
                    if isinstance(ass, GSSimpleAssignment):
                        outputs.append(GStripsConditionalAssignment(condition, ass))
                    elif isinstance(ass, GStripsConditionalAssignment):
                        outputs.append(GStripsConditionalAssignment(gs_compose_bool_expressions([condition, ass.condition], propagate_implicit_propositions=False), ass.assignment))
                    else:
                        raise TypeError('Invalid assignment type: {}.'.format(ass))
            elif isinstance(expr, SE.SAssignment):
                if isinstance(expr.predicate, SE.SSASPredicateApplicationExpression):
                    if is_relaxed:
                        raise NotImplementedError('Relaxed assignment to SAS predicate not supported during compilation. First compile it without is_relaxed, and re-run recompile_relaxed_operators.')
                    feature = expr.predicate
                    feature_name = feature.sas_name
                    predicate_def = self.domain.functions[feature_name]
                    feature_sas_size = predicate_def.ao_discretization.size
                    assert isinstance(expr.value, SE.SSASExpression)
                    argument_indices = list()
                    for arg_index, arg in enumerate(feature.arguments):
                        argument_indices.append(ctx.get_bounded_variable(arg))
                    expression = {k: self._compose_grounded_classifier_strips(ctx, v) for k, v in expr.value.mappings.items()}
                    sas_assignment = GSSASAssignment(feature_name, feature_sas_size, argument_indices, expression)
                    outputs.extend(sas_assignment.to_conditional_assignments())
                else:
                    feature = expr.predicate
                    value = bool(expr.value)
                    if value:
                        add_effects.add(self.compose_grounded_predicate_strips(ctx, feature))
                        if not is_relaxed:
                            add_effects.add(self.compose_grounded_predicate_strips(ctx, feature, negated=True))
                    else:
                        add_effects.add(self.compose_grounded_predicate_strips(ctx, feature, negated=True))
                        if not is_relaxed:
                            add_effects.add(self.compose_grounded_predicate_strips(ctx, feature))

        if len(add_effects) > 0 or len(del_effects) > 0:
            outputs.append(GSSimpleAssignment(add_effects, del_effects))
        return outputs

    def compile_expr(self, expr: Union[str, Expression], state: State) -> Tuple[GSBoolExpression, Set[SProposition]]:
        expr = self.domain.parse(expr)
        expr = flatten_expression(expr, flatten_cacheable_bool=not self.cache_bool_predicates)
        ctx = ExpressionExecutionContext(self._executor, state, {})
        return self.compose_grounded_classifier(ctx, expr), set()

    def compile_operator(self, op: OperatorApplier, state: State, is_relaxed=False) -> GStripsOperator:
        ctx = ExpressionExecutionContext(self._executor, state, compose_bvdict_args(op.operator.arguments, op.arguments, state=state))
        preconditions = list()
        for pred in op.operator.preconditions:
            preconditions.append(self._compose_grounded_classifier_strips(ctx, pred.ao_discretization))
        precondition = gs_compose_bool_expressions(preconditions, is_disjunction=False, propagate_implicit_propositions=False)
        effects = self._compose_grounded_assignment_strips(ctx, [eff.ao_discretization for eff in op.operator.effects], is_relaxed)
        return GStripsOperator(precondition, effects, op, implicit_propositions=set())

    def compile_derived_predicate(self, dp: Predicate, state: State, is_relaxed=False) -> List[GSDerivedPredicate]:
        arguments = list()
        for arg in dp.arguments:
            arguments.append(range(state.get_nr_objects_by_type(arg.dtype.typename)))

        rvs = list()
        for arg_indices in itertools.product(*arguments):
            bounded_variables = dict()
            for arg, arg_index in zip(dp.arguments, arg_indices):
                bounded_variables.setdefault(arg.typename, dict())[arg.name] = arg_index
            rvs.append(GSDerivedPredicate(
                dp.name, arg_indices,
                self._compose_grounded_classifier_strips(ctx, dp.ao_discretization),
                self._compose_grounded_classifier_strips(ctx, dp.ao_discretization, negated=True),
                is_relaxed=is_relaxed
            ))
        return rvs

    def compile_state(self, state: State, forward_derived: bool = False) -> SState:
        # Note: this function will change the original values of the state.
        # So be sure to make a copy of the state before calling this function.
        # This copying behavior is implemented in the compile_task function. If you are calling this function
        # directly, make sure to copy the state before calling this function.

        if forward_derived and self.cache_bool_predicates:
            self._executor.forward_predicates_and_axioms(state, forward_state_variables=False, forward_derived=True, forward_axioms=False)

        for name, feature in state.features.items():
            predicate_def = self.domain.functions[name]
            if predicate_def.is_state_variable and not (predicate_def.return_type == BOOL):
                state.features[name].tensor_indices = predicate_def.ao_discretization.quantize(feature).tensor

        predicates = set()
        for name, feature in state.features.items():
            predicate_def = self.domain.functions[name]
            if predicate_def.is_state_variable or (self.cache_bool_predicates and predicate_def.return_type == BOOL):
                if feature.dtype == BOOL:
                    for args, v in _iter_value(feature):
                        if v > 0.5:
                            predicates.add(_format_proposition((self.get_grounded_predicate_indentifier(name),) + args))
                        else:
                            predicates.add(_format_proposition((self.get_grounded_predicate_indentifier(name, 'not'),) + args))
                else:
                    codebook = predicate_def.ao_discretization
                    quantized_feature = codebook.quantize(feature)
                    for args, v in _iter_value(quantized_feature):
                        v = int(v)
                        for i in range(codebook.size):
                            if i == v:
                                predicates.add(_format_proposition((self.get_grounded_predicate_indentifier(f'{name}@{i}'),) + args))
                            else:
                                predicates.add(_format_proposition((self.get_grounded_predicate_indentifier(f'{name}@{i}', 'not'),) + args))

        return SState(predicates)

    def compile_task(
        self,
        state: State,
        goal_expr: Union[str, Expression],
        actions: Optional[Sequence[OperatorApplier]] = None,
        is_relaxed = False,
        forward_relevance_analysis: bool = True,
        backward_relevance_analysis: bool = True,
        verbose: bool = False
    ) -> GStripsProblem:
        state = state.clone()
        if self.cache_bool_predicates:
            self._executor.forward_predicates_and_axioms(state, forward_state_variables=True, forward_derived=True, forward_axioms=False)
        return super().compile_task(
            state, goal_expr, actions, is_relaxed,
            forward_relevance_analysis=forward_relevance_analysis, backward_relevance_analysis=backward_relevance_analysis,
            verbose=verbose
        )

    def relevance_analysis(self, task: GStripsProblem, relaxed_relevance: bool = False, forward: bool = True, backward: bool = True) -> GStripsProblem:
        return relevance_analysis_v2(task, relaxed_relevance=relaxed_relevance, forward=forward, backward=backward)
"""


class GStripsTranslatorFromAtomicStrips(GStripsTranslatorBase):
    def compile_expr(self, expr: Union[str, Expression], state: State) -> Tuple[GSBoolOutputExpression, Set[SProposition]]:
        raise NotImplementedError()

    def compile_operator(self, op: OperatorApplier, state: State, is_relaxed=False) -> GStripsOperator:
        raise NotImplementedError()

    def compile_regression_rule(self, op: OperatorApplier, state: State) -> GStripsRegressionRule:
        raise NotImplementedError()

    def compile_state(self, state: State, forward_derived: bool = False) -> SState:
        raise NotImplementedError()

    def compile_derived_predicate(self, dp: Predicate, state: State, is_relaxed=False) -> List[GStripsDerivedPredicate]:
        raise NotImplementedError('Derived predicates are not supported in Optimistic GStrips translation.')

    def relevance_analysis(self, task: GStripsProblem, relaxed_relevance: bool = False, forward: bool = True, backward: bool = True) -> GStripsProblem:
        return relevance_analysis_v1(task, relaxed_relevance=relaxed_relevance, forward=forward, backward=backward)


def _find_cached_predicates(domain: Domain, allow_cacheable_bool: bool = False) -> Iterable[Predicate]:
    """
    Return the set of predicates that are either in the `basic` or the `augmented` group.
    When the flag allow_cacheable_bool is set to True, also return the set of boolean predicates that are cacheable.

    Args:
        domain: the domain to search for predicates
        allow_cacheable_bool: whether to return the set of boolean predicates that are cacheable

    Returns:
        the set of predicates that are either in the `basic` or the `augmented` group and optionally cacheable boolean predicates.
    """
    for f in domain.functions.values():
        if f.is_state_variable:
            yield f
        elif allow_cacheable_bool and f.is_cacheable and f.return_type == BOOL:
            yield f


def _iter_value(value: TensorValue) -> Iterator[Tuple[Tuple[int, ...], Union[bool, int, float]]]:
    indices = [list(range(value.tensor.size(i))) for i in range(value.total_batch_dims)]
    for args, x in zip(itertools.product(*indices), value.tensor.flatten().tolist()):
        yield args, x


def _extract_all_propositions(classifiers: Sequence[Tuple[GSBoolOutputExpression, Set[SProposition]]]) -> Set[SProposition]:
    return set.union(*[c[1] for c in classifiers], *[c[0].iter_propositions() for c in classifiers if isinstance(c[0], GSBoolOutputExpression)])


def _format_proposition(pred_application: tuple[Union[SPredicateName, int], ...]) -> SProposition:
    return make_sproposition(*pred_application)


def relevance_analysis_v1(task: GStripsProblem, relaxed_relevance: bool = False, forward: bool = True, backward: bool = True) -> GStripsProblem:
    """
    Relevance analysis for a task.

    Args:
        task: the StripsTask to be analyzed.
        relaxed_relevance: whether to use relaxed relevance analysis. If this is set to False, we will not drop functions that are "implicitly" used by
            the operators. One example is that if one of the effect of the operator is a function (instead of simply Y/N).
        forward: whether to perform forward relevance analysis.
        backward: whether to perform backward relevance analysis.

    Returns:
        the updated task, removing all irrelevant propositions and operators.
    """

    # forward_analysis. a.k.a. static analysis.
    # useful when most objects are "non-movable". Then we don't need to track
    # their state variables/pose variables.

    # print('relevance_analysis::before')
    # print(task)
    if len(task.derived_predicates) > 0:
        raise RuntimeError('relevance_analysis::task.derived_predicates is not supported in relevance_analysis_v1.')

    if forward:
        task.compile()
        achievable_facts = set(task.state)
        changed = True
        visited = [False for _ in range(len(task.operators))]
        while changed:
            old_lengths = len(achievable_facts)
            for i, op in enumerate(task.operators):
                if not visited[i] and op.applicable(achievable_facts):
                    for e in op.effects:
                        achievable_facts.update(e.add_effects)
                    visited[i] = True
            changed = len(achievable_facts) != old_lengths
        new_operators = [op for i, op in enumerate(task.operators) if visited[i]]

        relevant_facts = set()
        for op in new_operators:
            for e in op.effects:
                relevant_facts.update(e.iter_propositions())

        # Note:: it seems that even when the task is not relaxed, we can also
        # do this drop...
        # Basically, if goal - relevant_facts isn't a subset of the initial
        # state, the problem is just unsolvable.
        # But when there are disjunctions, it's a bit hard to check.
        # relevant_facts.update(task.goal.iter_propositions())

        new_state = task.state.intersection(relevant_facts)
        new_operators = [op.filter_propositions(relevant_facts, state=task.state) for op in new_operators]
        new_goal = task.goal.filter_propositions(relevant_facts, state=task.state)

        task = GStripsProblem(new_state, new_goal, new_operators, is_relaxed=task.is_relaxed, facts=relevant_facts)
        task.compile()

        # print('relevance_analysis::forward')
        # print(task)

    # backward analysis.
    if backward:
        relevant_facts = set()
        relevant_facts.update(task.goal.iter_propositions())
        relevant_facts.update(task.goal_implicit_propositions)

        op_eff_facts = list()
        for op in task.operators:
            effects = set()
            for e in op.effects:
                effects.update(e.iter_propositions())
            op_eff_facts.append(effects)

        changed = True
        while changed:
            old_lengths = len(relevant_facts)
            for op, eff_facts in zip(task.operators, op_eff_facts):
                if set.intersection(eff_facts, relevant_facts):
                    relevant_facts |= set(op.precondition.iter_propositions())
                    if not relaxed_relevance:
                        relevant_facts |= set(op.implicit_propositions)
                    for e in op.effects:
                        if isinstance(e, GSConditionalAssignExpression):
                            relevant_facts |= set(e.condition.iter_propositions())
            changed = len(relevant_facts) != old_lengths

        new_operators = list()
        for op in task.operators:
            new_op = op.filter_propositions(relevant_facts, state=task.state)
            empty = True
            for e in new_op.effects:
                assert isinstance(e, (GSSimpleBoolAssignExpression, GSConditionalAssignExpression))
                if len(e.add_effects.symmetric_difference(e.del_effects)) > 0:
                    empty = False
                    break
                if isinstance(e, GSConditionalAssignExpression) and len(list(e.condition.iter_propositions())) > 0:
                    empty = False
                    break
            if not empty:
                new_operators.append(new_op)

        new_state = task.state.intersection(relevant_facts)
        task = GStripsProblem(new_state, task.goal, new_operators, is_relaxed=task.is_relaxed, facts=relevant_facts)

        # print('relevance_analysis::backward')
        # print(task)

    return task


def relevance_analysis_v2(task: GStripsProblem, relaxed_relevance: bool = False, forward: bool = True, backward: bool = True) -> GStripsProblem:
    """
    Relevance analysis for a task.

    Args:
        task: the StripsTask to be analyzed.
        relaxed_relevance: whether to use relaxed relevance analysis. If this is set to False, we will not drop functions that are "implicitly" used by
        the operators. One example is that if one of the effect of the operator is a function (instead of simply Y/N).
        forward: whether to run the forward pruning (forward reachability checking).
        backward: whether to run the backward pruning (goal regression).

    Returns:
        the updated task, removing all irrelevant propositions and operators.
    """

    # forward_analysis. a.k.a. static analysis.
    # useful when most objects are "non-movable". Then we don't need to track
    # their state variables/pose variables.

    # import ipdb; ipdb.set_trace()
    # print('relevance_analysis::before')
    # print(task)

    if forward:
        # collect all operators and derived predicates applicable.
        used_ops = set()
        used_dps = set()
        task.compile()
        achievable_facts = set(task.state)
        for i, dp in enumerate(task.derived_predicates):
            for j, eff in enumerate(dp.effects):
                if eff.applicable(achievable_facts):
                    achievable_facts.update(eff.add_effects)
                    used_dps.add((i, j))

        changed = True
        while changed:
            old_lengths = len(achievable_facts)
            for i, op in enumerate(task.operators):
                applicable = op.applicable(achievable_facts)
                if applicable:
                    for j, eff in enumerate(op.effects):
                        if (i, j) not in used_ops:
                            if isinstance(eff, GSSimpleBoolAssignExpression) or isinstance(eff, GSConditionalAssignExpression) and eff.applicable(achievable_facts):
                                achievable_facts.update(eff.add_effects)
                                used_ops.add((i, j))
            for i, dp in enumerate(task.derived_predicates):
                for j, eff in enumerate(dp.effects):
                    if (i, j) not in used_dps and eff.applicable(achievable_facts):
                        achievable_facts.update(eff.add_effects)
                        used_dps.add((i, j))
            changed = len(achievable_facts) != old_lengths

        new_operators = list()
        for i, op in enumerate(task.operators):
            used_effects = list()
            for j, eff in enumerate(op.effects):
                if (i, j) in used_ops:
                    used_effects.append(eff)
            if len(used_effects) > 0:
                new_operators.append(GStripsOperator(op.precondition, used_effects, op.raw_operator, op.implicit_propositions))
        new_derived_predicates = list()
        for i, dp in enumerate(task.derived_predicates):
            used_effects = list()
            for j, eff in enumerate(dp.effects):
                if (i, j) in used_dps:
                    used_effects.append(eff)
            if len(used_effects) > 0:
                new_derived_predicates.append(GStripsDerivedPredicate(dp.name, dp.arguments, effects=used_effects))

        relevant_facts = set()
        for op in new_operators:
            for e in op.effects:
                relevant_facts.update(e.iter_propositions())
        for i, dp in enumerate(task.derived_predicates):
            for e in dp.effects:
                relevant_facts.update(e.assignment.iter_propositions())

        # Note:: it seems that even when the task is not relaxed, we can also
        # do this drop...
        # Basically, if goal - relevant_facts isn't a subset of the initial
        # state, the problem is just unsolvable.
        # But when there are disjunctions, it's a bit hard to check.
        # relevant_facts.update(task.goal.iter_propositions())

        new_state = task.state.intersection(relevant_facts)
        new_operators = [op.filter_propositions(relevant_facts, state=task.state) for op in new_operators]
        new_derived_predicates = [dp.filter_propositions(relevant_facts, state=task.state) for dp in task.derived_predicates]
        new_goal = task.goal.filter_propositions(relevant_facts, state=task.state)

        task = GStripsProblem(new_state, new_goal, new_operators, is_relaxed=task.is_relaxed, derived_predicates=new_derived_predicates, facts=relevant_facts)
        task.compile()

        # import ipdb; ipdb.set_trace()
        # print('relevance_analysis::forward')
        # print(task)

    if backward:
        # backward analysis.
        relevant_facts = set()
        relevant_facts.update(task.goal.iter_propositions())
        relevant_facts.update(task.goal_implicit_propositions)

        changed = True
        while changed:
            old_lengths = len(relevant_facts)
            for i, dp in enumerate(task.derived_predicates):
                for j, eff in enumerate(dp.effects):
                    if set.intersection(set(eff.iter_propositions()), relevant_facts):
                        relevant_facts.update(eff.condition.iter_propositions())

            for i, op in enumerate(task.operators):
                for j, eff in enumerate(op.effects):
                    if set.intersection(set(eff.iter_propositions()), relevant_facts):
                        relevant_facts |= set(op.precondition.iter_propositions())
                        if not relaxed_relevance:
                            relevant_facts |= set(op.implicit_propositions)
                        if isinstance(eff, GSConditionalAssignExpression):
                            relevant_facts |= set(eff.condition.iter_propositions())
            changed = len(relevant_facts) != old_lengths

        new_operators = list()
        for op in task.operators:
            new_op = op.filter_propositions(relevant_facts, state=task.state)
            new_effects = list()
            for j, eff in enumerate(new_op.effects):
                assert isinstance(eff, (GSSimpleBoolAssignExpression, GSConditionalAssignExpression))
                if len(eff.add_effects.symmetric_difference(eff.del_effects)) > 0:
                    new_effects.append(eff)
            if len(new_effects) > 0:
                new_op = GStripsOperator(new_op.precondition, new_effects, new_op.raw_operator, new_op.implicit_propositions)
                new_operators.append(new_op)

        new_derived_predicates = list()
        for dp in task.derived_predicates:
            new_dp = dp.filter_propositions(relevant_facts, state=task.state)
            new_effects = list()
            for j, eff in enumerate(new_dp.effects):
                if len(eff.add_effects.symmetric_difference(eff.del_effects)) > 0:
                    new_effects.append(eff)
            if len(new_effects) > 0:
                new_dp = GStripsDerivedPredicate(new_dp.name, new_dp.arguments, effects=new_effects)
                new_derived_predicates.append(new_dp)

        new_state = task.state.intersection(relevant_facts)
        task = GStripsProblem(new_state, task.goal, new_operators, is_relaxed=task.is_relaxed, derived_predicates=new_derived_predicates, facts=relevant_facts)

        # import ipdb; ipdb.set_trace()
        # print('relevance_analysis::backward')
        # print(task)

    return task
