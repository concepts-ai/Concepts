#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : atomic_strips_domain.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/27/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Optional, Union, Sequence, Tuple, Mapping, Dict

import jacinle
from jacinle.utils.cache import cached_property
from concepts.dsl.dsl_types import ObjectType, Variable, ObjectConstant
from concepts.dsl.expression import VariableExpression, FunctionApplicationExpression, NotExpression, AssignExpression, ConstantExpression, is_and_expr
from concepts.pdsketch.predicate import Predicate
from concepts.pdsketch.operator import Operator, RegressionRule, OperatorApplicationExpression, AchieveExpression
from concepts.pdsketch.domain import Domain, Problem
from concepts.pdsketch.parsers.pdsketch_parser import load_domain_file, load_domain_string, load_problem_file, load_problem_string
from concepts.pdsketch.strips.strips_expression import SBoolPredicateApplicationExpression, SProposition, SState, SStateCompatible, make_sproposition_from_function_application


# TODO(Jiayuan Mao @ 2023/04/27): implement ADL operator.

class AtomicStripsOperator(object):
    def __init__(
        self,
        arguments: Sequence[Variable],
        preconditions: Sequence[SBoolPredicateApplicationExpression],
        add_effects: Sequence[SBoolPredicateApplicationExpression],
        del_effects: Sequence[SBoolPredicateApplicationExpression],
        raw_operator: Operator
    ):
        self.name = raw_operator.name
        self.arguments = tuple(arguments)
        self.preconditions = tuple(preconditions)
        self.add_effects = tuple(add_effects)
        self.del_effects = tuple(del_effects)
        self.raw_operator = raw_operator

    arguments: Tuple[Variable]
    """The arguments of the operator, as a tuple of :class:`~concepts.dsl.dsl_types.Variable`."""

    preconditions: Tuple[SBoolPredicateApplicationExpression, ...]
    """The precondition of the operator, as a tuple of :class:`~concepts.pdsketch.strips.strips_expression.SBoolPredicateApplicationExpression`."""

    add_effects: Tuple[SBoolPredicateApplicationExpression, ...]
    """The add effects of the operator, as a tuple of :class:`~concepts.pdsketch.strips.strips_expression.SBoolPredicateApplicationExpression`."""

    del_effects: Tuple[SBoolPredicateApplicationExpression, ...]
    """The delete effects of the operator, as a tuple of :class:`~concepts.pdsketch.strips.strips_expression.SBoolPredicateApplicationExpression`."""

    raw_operator: Operator
    """The raw operator that this strips operator is derived from."""

    def relax(self) -> 'AtomicStripsOperator':
        return type(self)(
            arguments=self.arguments,
            preconditions=self.preconditions,
            add_effects=self.add_effects,
            del_effects=[],
            raw_operator=self.raw_operator
        )

    @classmethod
    def from_operator(cls, operator: Operator) -> 'AtomicStripsOperator':
        """Create a strips operator from a PDSketch operator.

        Args:
            operator: the PDSketch operator.

        Returns:
            the strips operator.
        """
        arguments = operator.arguments

        preconditions = list()
        for precondition in operator.preconditions:
            precondition = precondition.bool_expr
            preconditions.append(SBoolPredicateApplicationExpression.from_function_application_expression(precondition))

        add_effects = list()
        del_effects = list()
        for effect in operator.effects:
            effect = effect.assign_expr
            assert isinstance(effect, AssignExpression), f'Not supported assign expression: {effect}.'

            sassign = SBoolPredicateApplicationExpression.from_function_application_expression(effect.predicate)
            assert isinstance(effect.value, ConstantExpression), f'Not supported assign expression: {effect}.'
            value = effect.value.constant.item()
            assert isinstance(value, int)
            if value > 0.5:
                add_effects.append(sassign)
            else:
                del_effects.append(sassign)

        return cls(arguments, preconditions, add_effects, del_effects, operator)

    def __str__(self) -> str:
        return f'{self.name}({", ".join(arg.name for arg in self.arguments)})'

    def __repr__(self) -> str:
        return f'StripsOperator({self.name}, arguments={self.arguments}, preconditions={self.preconditions}, {self.add_effects}, {self.del_effects})'

    def to_applier_str(self, bound_arguments: Dict[str, str]):
        return self.name + '(' + ', '.join(f'{arg.name}={bound_arguments[arg.name]}' for arg in self.arguments) + ')'

    def to_applier_pddl_str(self, bound_arguments: Dict[str, str]):
        return '(' + self.name + ' ' + ' '.join(bound_arguments[arg.name] for arg in self.arguments) + ')'

    def ground(self, variable_dict: Dict[str, str], state: Optional[SStateCompatible] = None):
        # TODO(Jiayuan Mao @ 2023/04/27): include the state in the grounding.
        return AtomicStripsOperatorApplier(self, variable_dict)

    def __call__(self, *args: str):
        assert len(args) == len(self.arguments), f'Expected {len(self.arguments)} arguments, got {len(args)}.'
        bound_arguments = {arg.name: argv for arg, argv in zip(self.arguments, args)}
        return self.ground(bound_arguments)


class AtomicStripsOperatorApplier(object):
    def __init__(self, operator: AtomicStripsOperator, bound_arguments: Dict[str, str]):
        self.operator = operator
        self.arguments = tuple(bound_arguments[arg.name] for arg in operator.arguments)
        self.bound_arguments = bound_arguments

        self.preconditions = tuple(precondition.ground(bound_arguments, return_proposition=True) for precondition in operator.preconditions)
        self.add_effects = tuple(add_effect.ground(bound_arguments, return_proposition=True) for add_effect in operator.add_effects)
        self.del_effects = tuple(del_effect.ground(bound_arguments, return_proposition=True) for del_effect in operator.del_effects)

    operator: AtomicStripsOperator
    """The operator that this applier is derived from."""

    arguments: Tuple[str, ...]
    """The arguments of the operator, as a tuple of strings."""

    bound_arguments: Dict[str, str]
    """The bound arguments of the operator."""

    preconditions: Tuple[SProposition, ...]
    """The preconditions of the operator, as a tuple of :class:`~concepts.pdsketch.strips.strips_expression.SProposition`."""

    add_effects: Tuple[SProposition, ...]
    """The add effects of the operator, as a tuple of :class:`~concepts.pdsketch.strips.strips_expression.SProposition`."""

    del_effects: Tuple[SProposition, ...]
    """The delete effects of the operator, as a tuple of :class:`~concepts.pdsketch.strips.strips_expression.SProposition`."""

    def __str__(self) -> str:
        return f'{self.operator.name}({", ".join(f"{arg.name}={self.bound_arguments[arg.name]}" for arg in self.operator.arguments)})'

    __repr__ = jacinle.repr_from_str


class AtomicStripsOperatorApplicationExpression(object):
    def __init__(self, operator: AtomicStripsOperator, arguments: Sequence[Variable]):
        self.operator = operator
        self.arguments = tuple(arguments)

    operator: AtomicStripsOperator
    """The operator to be applied."""

    arguments: Tuple[Variable, ...]
    """The arguments of the operator."""

    def __str__(self) -> str:
        return f'{self.operator.name}({", ".join(arg.name if isinstance(arg, Variable) else arg for arg in self.arguments)})'

    __repr__ = jacinle.repr_from_str

    @classmethod
    def from_operator_application_expression(cls, expression: OperatorApplicationExpression, atomic_strips_operators: Mapping[str, AtomicStripsOperator]) -> 'AtomicStripsOperatorApplicationExpression':
        """Create a strips operator application expression from a PDSketch operator application expression.

        Args:
            expression: the PDSketch operator application expression.

        Returns:
            the strips operator application expression.
        """
        operator = atomic_strips_operators[expression.operator.name]
        arguments = list()
        for arg in expression.arguments:
            assert isinstance(arg, VariableExpression), f'Not supported argument expression: {arg}.'
            arguments.append(arg.variable)
        return cls(operator, arguments)

    def ground(self, variable_dict: Dict[str, str], state: Optional[SStateCompatible] = None):
        return AtomicStripsOperatorApplier(self.operator, {arg.name: variable_dict[arg.name] for arg in self.arguments})

    def __call__(self, *args):
        assert len(args) == len(self.arguments), f'Expected {len(self.arguments)} arguments, got {len(args)}.'
        bound_arguments = {arg.name: argv for arg, argv in zip(self.arguments, args)}
        return self.ground(bound_arguments)


class AtomicStripsAchieveExpression(object):
    def __init__(self, goal: SBoolPredicateApplicationExpression, maintains: Sequence[SBoolPredicateApplicationExpression]):
        self.goal = goal
        self.maintains = tuple(maintains)

    goal: SBoolPredicateApplicationExpression
    """The goal of the achieve expression."""

    maintains: Tuple[SBoolPredicateApplicationExpression, ...]
    """A list of expression to be maintained."""

    def __str__(self) -> str:
        return f'achieve({self.goal}, maintains={{{self.maintains}}})'

    __repr__ = jacinle.repr_from_str

    @classmethod
    def from_achieve_expression(cls, expression: AchieveExpression) -> 'AtomicStripsAchieveExpression':
        """Create a strips achieve expression from a PDSketch achieve expression.

        Args:
            expression: the PDSketch achieve expression.

        Returns:
            the strips achieve expression.
        """
        goal = SBoolPredicateApplicationExpression.from_function_application_expression(expression.goal)
        maintains = tuple(SBoolPredicateApplicationExpression.from_function_application_expression(maintain) for maintain in expression.maintains)
        return cls(goal, maintains)

    def ground(self, variable_dict: Dict[str, str], state: Optional[SStateCompatible] = None):
        return AtomicStripsGroundedAchieveExpression(
            self.goal.ground(variable_dict, return_proposition=True),
            tuple(maintain.ground(variable_dict, return_proposition=True) for maintain in self.maintains)
        )

    def __call__(self, *args):
        assert len(args) == len(self.arguments), f'Expected {len(self.arguments)} arguments, got {len(args)}.'
        bound_arguments = {arg.name: argv for arg, argv in zip(self.arguments, args)}
        return self.ground(bound_arguments)


class AtomicStripsGroundedAchieveExpression(object):
    def __init__(self, goal: SProposition, maintains: Sequence[SProposition]):
        self.goal = goal
        self.maintains = tuple(maintains)

    goal: SProposition
    """The goal of the achieve expression."""

    maintains: Tuple[SProposition, ...]
    """A list of expression to be maintained."""

    def __str__(self) -> str:
        return f'achieve({self.goal}, maintains={{{", ".join(map(str, self.maintains))}}})'

    __repr__ = jacinle.repr_from_str


class AtomicStripsRegressionRule(object):
    def __init__(
        self,
        arguments: Sequence[Variable],
        preconditions: Sequence[SBoolPredicateApplicationExpression],
        preconstraints: Sequence[SBoolPredicateApplicationExpression],
        goal: SBoolPredicateApplicationExpression,
        body: Sequence[Union[AtomicStripsOperatorApplicationExpression, AtomicStripsAchieveExpression]],
        raw_regression_rule: RegressionRule,
    ):
        self.name = raw_regression_rule.name
        self.arguments = tuple(arguments)
        self.preconditions = tuple(preconditions)
        self.preconstraints = tuple(preconstraints)
        self.goal = goal
        self.body = tuple(body)
        self.raw_regression_rule = raw_regression_rule

    arguments: Tuple[Variable, ...]
    """The arguments of the regression rule."""

    preconditions: Tuple[SBoolPredicateApplicationExpression, ...]
    """The preconditions of the regression rule."""

    preconstraints: Tuple[SBoolPredicateApplicationExpression, ...]
    """The preconstraints of the regression rule."""

    goal: SBoolPredicateApplicationExpression
    """The goal of the regression rule."""

    body: Tuple[Union[AtomicStripsOperatorApplicationExpression, AtomicStripsAchieveExpression], ...]
    """The body of the regression rule."""

    raw_regression_rule: RegressionRule
    """The raw regression rule."""

    def __str__(self) -> str:
        return f'{self.goal} <- {", ".join(str(item) for item in self.body)}'

    __repr__ = jacinle.repr_from_str

    @classmethod
    def from_regression_rule(cls, regression_rule: RegressionRule, atomic_strips_operators: Mapping[str, AtomicStripsOperator]):
        arguments = regression_rule.arguments
        preconditions = tuple(SBoolPredicateApplicationExpression.from_function_application_expression(precondition.bool_expr) for precondition in regression_rule.preconditions)
        preconstraints = tuple(SBoolPredicateApplicationExpression.from_function_application_expression(preconstraint.bool_expr) for preconstraint in regression_rule.preconstraints)

        assert len(regression_rule.goal) == 1, f'Expected exactly one goal, got {len(regression_rule.goal)}.'
        goal_expression = regression_rule.goal[0].assign_expr
        assert isinstance(goal_expression, AssignExpression), f'Expected an assign expression, got {type(goal_expression)}.'
        assert isinstance(goal_expression.value, ConstantExpression), f'Expected a constant expression for the goal expression assignment, got {type(goal_expression.value)}.'
        value = goal_expression.value.constant.item()
        assert isinstance(value, int), f'Expected an integer for the goal expression assignment, got {type(value)}.'
        sassign = SBoolPredicateApplicationExpression.from_function_application_expression(goal_expression.predicate, negated=value < 0.5)

        body = []
        for body_expression in regression_rule.body:
            if isinstance(body_expression, OperatorApplicationExpression):
                body.append(AtomicStripsOperatorApplicationExpression.from_operator_application_expression(body_expression, atomic_strips_operators))
            elif isinstance(body_expression, AchieveExpression):
                body.append(AtomicStripsAchieveExpression.from_achieve_expression(body_expression))
            else:
                raise NotImplementedError(f'Not supported body expression: {body_expression}.')

        return cls(arguments, preconditions, preconstraints, sassign, body, regression_rule)

    def ground(self, variable_dict: Dict[str, str], state: Optional[SStateCompatible] = None):
        # TODO(Jiayuan Mao @ 2023/07/12): include the state in the grounding.
        return AtomicStripsRegressionRuleApplier(self, variable_dict)

    def __call__(self, *args: str):
        assert len(args) == len(self.arguments), f'Expected {len(self.arguments)} arguments, got {len(args)}.'
        bound_arguments = {arg.name: argv for arg, argv in zip(self.arguments, args)}
        return self.ground(bound_arguments)


class AtomicStripsRegressionRuleApplier(object):
    def __init__(self, regression_rule: AtomicStripsRegressionRule, bound_arguments: Dict[str, str]):
        self.regression_rule = regression_rule
        self.arguments = tuple(bound_arguments[arg.name] for arg in regression_rule.arguments)
        self.bound_arguments = bound_arguments

        self.preconditions = tuple(precondition.ground(bound_arguments, return_proposition=True) for precondition in regression_rule.preconditions)
        self.preconstraints = tuple(preconstraint.ground(bound_arguments, return_proposition=True) for preconstraint in regression_rule.preconstraints)
        self.goal = regression_rule.goal.ground(bound_arguments, return_proposition=True)
        self.body = tuple(item.ground(bound_arguments) for item in regression_rule.body)

    regression_rule: AtomicStripsRegressionRule
    """The regression rule."""

    arguments: Tuple[str, ...]
    """The arguments of the operator, as a tuple of strings."""

    bound_arguments: Dict[str, str]
    """The bound arguments of the operator."""

    def __str__(self) -> str:
        return f'{self.regression_rule.name}({", ".join(f"{arg}={argv}" for arg, argv in self.bound_arguments.items())})'

    __repr__ = jacinle.repr_from_str


class AtomicStripsDomain(object):
    """The domain of the atomic STRIPS planning problem."""

    def __init__(self, types: Dict[str, ObjectType], predicates: Dict[str, Predicate], operators: Dict[str, AtomicStripsOperator], regression_rules: Dict[str, AtomicStripsRegressionRule], constants: Dict[str, ObjectConstant]):
        self.types = types
        self.predicates = predicates
        self.operators = operators
        self.regression_rules = regression_rules
        self.constants = constants

    types: Dict[str, ObjectType]
    """The types of the domain, as a dictionary from type name to :class:`~concepts.pdsketch.strips.strips_expression.ObjectType`."""

    predicates: Dict[str, Predicate]
    """The predicates of the domain, as a dictionary from predicate name to :class:`~concepts.pdsketch.predicate.Predicate`."""

    operators: Dict[str, AtomicStripsOperator]
    """The operators of the domain, as a dictionary from operator name to :class:`~concepts.pdsketch.strips.atomic_strips_domain.AtomicStripsOperator."""

    regression_rules: Dict[str, AtomicStripsRegressionRule]
    """The regression rules of the domain, as a dictionary from regression rule name to :class:`~concepts.pdsketch.strips.atomic_strips_domain.AtomicStripsRegressionRule`."""

    constants: Dict[str, ObjectConstant]
    """The constants of the domain, as a dictionary from constant name to :class:`~concepts.dsl.dsl_types.ObjectConstant`."""

    @classmethod
    def from_domain(cls, domain: Domain) -> 'AtomicStripsDomain':
        operators = dict()
        for operator in domain.operators.values():
            if isinstance(operator, Operator):
                if operator.is_macro:
                    continue
                # TODO(Jiayuan Mao @ 2023/03/19): support macro operator here.
                strips_operator = AtomicStripsOperator.from_operator(operator)
                operators[operator.name] = strips_operator
        regression_rules = dict()
        for regression_rule in domain.regression_rules.values():
            strips_regression_rule = AtomicStripsRegressionRule.from_regression_rule(regression_rule, operators)
            regression_rules[regression_rule.name] = strips_regression_rule

        return cls(
            types=domain.types.copy(),
            predicates=domain.functions.copy(),
            operators=operators,
            regression_rules=regression_rules,
            constants=domain.constants.copy(),
        )


class AtomicStripsProblem(object):
    def __init__(self, domain: AtomicStripsDomain, objects: Dict[str, str], initial_state: SStateCompatible, conjunctive_goal: Sequence[SProposition]):
        self.domain = domain
        self.objects = objects
        self.initial_state = SState(initial_state)
        self.conjunctive_goal = tuple(conjunctive_goal)

    domain: AtomicStripsDomain
    """The domain of the problem, as a :class:`~concepts.pdsketch.strips.strips_domain.AtomicStripsDomain`."""

    objects: Dict[str, str]
    """The objects of the problem, as a dictionary from object name to object type."""

    initial_state: SState
    """The initial state of the problem, as a :class:`~concepts.pdsketch.strips.strips_expression.SState`."""

    conjunctive_goal: Tuple[SProposition, ...]
    """The conjunctive goal of the problem, as a tuple of :class:`~concepts.pdsketch.strips.strips_expression.SProposition`."""

    @cached_property
    def objects_type2names(self) -> Dict[str, Tuple[str, ...]]:
        objects_type2names = dict()
        for name, type_ in self.objects.items():
            objects_type2names.setdefault(type_, []).append(name)
        return {type_: tuple(names) for type_, names in objects_type2names.items()}

    @classmethod
    def from_domain_and_problem(cls, domain: AtomicStripsDomain, problem: Problem) -> 'AtomicStripsProblem':
        objects = problem.objects.copy()
        for _, typename in objects.items():
            if typename not in domain.types:
                raise ValueError(f'Unknown type {typename}.')

        initial_state = list()
        for proposition in problem.predicates:
            if proposition.function.name not in domain.predicates:
                raise ValueError(f'Unknown predicate {proposition.function.name}.')
            initial_state.append(make_sproposition_from_function_application(proposition, objects))

        conjunctive_goal = list()
        if not is_and_expr(problem.goal):
            raise ValueError(f'Expected conjunctive goal, got {problem.goal}.')
        for proposition in problem.goal.arguments:
            if not isinstance(proposition, FunctionApplicationExpression):
                raise ValueError(f'Expected proposition, got {proposition}.')
            if proposition.function.name not in domain.predicates:
                raise ValueError(f'Unknown predicate {proposition.function.name}.')
            conjunctive_goal.append(make_sproposition_from_function_application(proposition, objects))

        return cls(
            domain,
            objects=objects,
            initial_state=initial_state,
            conjunctive_goal=conjunctive_goal,
        )


def load_astrips_domain_file(filename: str, return_raw_domain: bool = False) -> Union[AtomicStripsDomain, Tuple[Domain, AtomicStripsDomain]]:
    """Load a strips domain from a domain file.

    Args:
        filename: the domain file.
        return_raw_domain: whether to return the raw domain.

    Returns:
        the strips domain, or a tuple of the raw domain and the strips domain.
    """
    domain = load_domain_file(filename)
    strips_domain = AtomicStripsDomain.from_domain(domain)
    if return_raw_domain:
        return domain, strips_domain
    return strips_domain


def load_astrips_domain_string(domain_string: str, return_raw_domain: bool = False) -> Union[AtomicStripsDomain, Tuple[Domain, AtomicStripsDomain]]:
    """Load a strips domain from a domain string.

    Args:
        domain_string: the domain string.
        return_raw_domain: whether to return the raw domain.

    Returns:
        the strips domain, or a tuple of the raw domain and the strips domain.
    """
    domain = load_domain_string(domain_string)
    strips_domain = AtomicStripsDomain.from_domain(domain)
    if return_raw_domain:
        return domain, strips_domain
    return strips_domain


def load_astrips_problem_file(domain: Domain, astrips_domain: AtomicStripsDomain, filename: str) -> AtomicStripsProblem:
    """Load a strips problem from a problem file.

    Args:
        domain: the domain of the problem.
        filename: the problem file.

    Returns:
        the strips problem.
    """
    problem = load_problem_file(filename, domain=domain, return_tensor_state=False)
    return AtomicStripsProblem.from_domain_and_problem(astrips_domain, problem)


def load_astrips_problem_string(domain: Domain, astrips_domain: AtomicStripsDomain, problem_string: str) -> AtomicStripsProblem:
    """Load a strips problem from a problem string.

    Args:
        domain: the domain of the problem.
        problem_string: the problem string.

    Returns:
        the strips problem.
    """
    problem = load_problem_string(problem_string, domain=domain, return_tensor_state=False)
    return AtomicStripsProblem.from_domain_and_problem(astrips_domain, problem)

