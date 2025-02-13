#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : pdsketch_parser.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/30/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os
import os.path as osp
import itertools
import collections
from typing import Any, Optional, Union, Sequence, Tuple, Set, List, Dict
from lark import Lark, Tree, Transformer, v_args

import jacinle

import concepts.dsl.expression as E
from concepts.dsl.dsl_types import ObjectType, TensorValueTypeBase, VectorValueType, ListType, AutoType, BOOL, Variable, ObjectConstant, UnnamedPlaceholder
from concepts.dsl.expression import ExpressionDefinitionContext, get_expression_definition_context, FunctionApplicationExpression
from concepts.dsl.expression_utils import iter_exprs
from concepts.dsl.tensor_state import TensorState
from concepts.dm.pdsketch.predicate import Predicate
from concepts.dm.pdsketch.operator import Precondition, Effect, Implementation, Operator, OperatorApplicationExpression
from concepts.dm.pdsketch.regression_rule import RegressionRuleApplicationExpression, BindExpression, AchieveExpression, RegressionCommitFlag, RegressionRule
from concepts.dm.pdsketch.generator import Generator
from concepts.dm.pdsketch.domain import Domain, Problem
from concepts.dm.pdsketch.executor import PDSketchExecutor

__all__ = [
    'PDSketchParser', 'PDSketchTransformer',
    'load_domain_file', 'load_domain_string', 'parse_expression', 'load_problem_file', 'load_problem_string', 'parse_expression', 'load_domain_string_incremental'
]

logger = jacinle.get_logger(__file__)

# lark.v_args
inline_args = v_args(inline=True)
DEBUG_LOG_COMPOSE = False

EllipsisType = type(Ellipsis)


def _log_function(func):
    if DEBUG_LOG_COMPOSE:
        return jacinle.log_function(func)
    return func


class PDSketchParser(object):
    """Parser for PDSketch domain and problem files. Users should not use this class directly.
    Instead, use the following functions:

    - :func:`load_domain_file`
    - :func:`load_domain_string`
    - :func:`load_problem_file`
    - :func:`load_problem_string`
    - :func:`parse_expression`
    """

    grammar_file = osp.join(osp.dirname(__file__), 'pdsketch-v2.grammar')
    """The grammar definition for PDSketch."""

    def __init__(self):
        with open(type(self).grammar_file) as f:
            self.lark = Lark(f)

    def load(self, file) -> Tree:
        """Load a domain or problem file and return the corresponding tree."""
        with open(file) as f:
            return self.lark.parse(f.read())

    def loads(self, string) -> Tree:
        """Load a domain or problem string and return the corresponding tree."""
        return self.lark.parse(string)

    def make_domain(self, tree: Tree, domain_file_paths: Sequence[str] = tuple()) -> Domain:
        """Construct a PDSketch domain from a tree."""
        assert tree.children[0].data == 'definition'
        transformer = PDSketchTransformer(Domain(), domain_file_paths=domain_file_paths)
        transformer.transform(tree)
        domain = transformer.domain
        domain.post_init()
        return domain

    def make_problem(self, tree: Tree, domain: Domain, ignore_unknown_predicates: bool = False) -> Problem:
        """Construct a PDSketch problem from a tree."""
        assert tree.children[0].data == 'definition'
        transformer = PDSketchTransformer(domain, ignore_unknown_predicates=ignore_unknown_predicates)
        transformer.transform(tree)
        problem = transformer.problem
        return problem

    def make_expression(self, domain: Domain, tree: Tree, variables: Optional[Sequence[Variable]] = None) -> E.Expression:
        """Construct a PDSketch expression from a tree."""
        if variables is None:
            variables = list()
        transformer = PDSketchTransformer(domain, allow_object_constants=True)
        node = transformer.transform(tree).children[0]
        assert isinstance(node, (_FunctionApplicationImm, _QuantifierApplicationImm))
        with ExpressionDefinitionContext(*variables, domain=domain).as_default():
            return node.compose()

    def incremental_define_domain(self, domain: Domain, tree: Tree) -> None:
        """Incrementally define a PDSketch domain from a tree."""
        transformer = PDSketchTransformer(domain)
        transformer.transform(tree)
        domain.post_init()
        return domain


_parser = PDSketchParser()


def load_domain_file(filename: str, domain_file_paths: Sequence[str] = tuple()) -> Domain:
    """Load a domain from a file.

    Args:
        filename: the filename of the domain.

    Returns:
        the domain.
    """
    tree = _parser.load(filename)
    domain_file_paths = list(domain_file_paths) + [osp.realpath(osp.dirname(filename)), os.getcwd()]
    domain = _parser.make_domain(tree, domain_file_paths=domain_file_paths)
    return domain


def load_domain_string(domain_string: str, domain_file_paths: Sequence[str] = tuple()) -> Domain:
    """Load a domain from a string.

    Args:
        domain_string: the string of the domain.

    Returns:
        the domain.
    """
    tree = _parser.loads(domain_string)
    domain = _parser.make_domain(tree, domain_file_paths=domain_file_paths)
    return domain


def load_problem_file(filename: str, domain: Domain, ignore_unknown_predicates: bool = False, return_tensor_state: bool = True, executor: Optional[PDSketchExecutor] = None) -> Union[Tuple[TensorState, E.ValueOutputExpression], Problem]:
    """Load a problem from a file.

    Args:
        filename: the filename of the problem.
        domain: the domain.
        ignore_unknown_predicates: whether to ignore unknown predicates. If set to True, unknown predicates will be ignored with a warning.
            Otherwise, an error will be raised while parsing the problem.
        return_tensor_state: whether to return the initial state as a tensor state. If set to True, the return value wille be a tuple of (initial state, goal).
            Otherwise, the return value will be the :class:`~concepts.dm.pdsketch.domain.Problem` object.
        executor: the executor used to create initial states.

    Returns:
        the problem, as a tuple of (initial state, goal).
    """
    tree = _parser.load(filename)
    with ExpressionDefinitionContext(domain=domain).as_default():
        problem = _parser.make_problem(tree, domain, ignore_unknown_predicates=ignore_unknown_predicates)

    if return_tensor_state:
        assert executor is not None
        return problem.to_state(executor), problem.goal
    return problem


def load_problem_string(problem_string: str, domain: Domain, ignore_unknown_predicates: bool = False, return_tensor_state: bool = True, executor: Optional[PDSketchExecutor] = None) -> Union[Tuple[TensorState, E.ValueOutputExpression], Problem]:
    """Load a problem from a string.

    Args:
        problem_string: the string of the problem.
        domain: the domain.
        ignore_unknown_predicates: whether to ignore unknown predicates. If set to True, unknown predicates will be ignored with a warning.
            Otherwise, an error will be raised while parsing the problem.
        return_tensor_state: whether to return the initial state as a tensor state. If set to True, the return value wille be a tuple of (initial state, goal).
            Otherwise, the return value will be the :class:`~concepts.dm.pdsketch.domain.Problem` object.
        executor: the executor used to create initial states.

    Returns:
        the problem, as a tuple of (initial state, goal) if `return_tensor_state` is True, otherwise the :class:`~concepts.dm.pdsketch.domain.Problem` object.
    """
    tree = _parser.loads(problem_string)
    with ExpressionDefinitionContext(domain=domain).as_default():
        problem = _parser.make_problem(tree, domain, ignore_unknown_predicates=ignore_unknown_predicates)

    if return_tensor_state:
        if executor is None:
            raise RuntimeError('Executor must be provided when return_tensor_state is True.')
        return problem.to_state(executor), problem.goal
    return problem


def parse_expression(domain: Domain, string: str, variables: Optional[Sequence[Variable]] = None) -> E.Expression:
    """Parse an expression from a string.

    Args:
        domain: the domain.
        string: the string of the expression.
        variables: a list of variables that are used in the expression.

    Returns:
        the expression.
    """
    tree = _parser.loads(string)
    expr = _parser.make_expression(domain, tree, variables)
    return expr


def load_domain_string_incremental(domain: Domain, string: str) -> Domain:
    """Incrementally load a domain from a string.

    Args:
        domain: the domain.
        string: the string of the domain.

    Returns:
        the domain.
    """
    tree = _parser.loads(string)
    _parser.incremental_define_domain(domain, tree)
    return domain


class PDSketchTransformer(Transformer):
    """The tree-to-object transformer for PDSketch domain and problem files. Users should not use this class directly."""

    def __init__(
        self, init_domain: Domain = None, allow_object_constants: bool = True, ignore_unknown_predicates: bool = False,
        domain_file_paths: Sequence[str] = tuple()
    ):
        super().__init__()

        self.domain = init_domain
        self.problem = Problem()
        self.allow_object_constants = allow_object_constants
        self.ignore_unknown_predicates = ignore_unknown_predicates
        self.domain_file_paths = domain_file_paths
        self.ignored_predicates: Set[str] = set()
        self.defaults = {
            'default_regression_rule_always': False,
            'default_regression_rule_serializability': 'strong',
            'default_regression_rule_csp_serializability': 'none'
        }

    @inline_args
    def definition_decl(self, definition_type, definition_name):
        if definition_type.value == 'domain':
            self.domain.name = definition_name.value

    @inline_args
    def extends_definition(self, filename):
        filename = self.find_domain_file(filename)
        tree = _parser.load(filename)
        self.transform(tree)

    @inline_args
    def pragma_definition(self, kwargs):
        self.defaults.update(kwargs)

    def find_domain_file(self, filename: str) -> str:
        for path in self.domain_file_paths:
            full_path = osp.join(path, filename)
            if osp.exists(full_path):
                return full_path
        raise FileNotFoundError(f'Cannot find domain file {filename}.')

    def type_definition(self, args):
        """Parse a type definition.

        Very ugly hack to handle multi-line definition in PDDL. In PDDL, type definition can be separated by newline.
        This kinds of breaks the parsing strategy that ignores all whitespaces. More specifically, consider the following two definitions:

        .. code-block:: pddl

            (:types
              a
              b - a
            )

        and

        .. code-block:: pddl

            (:types
              a b - a
            )
        """
        if isinstance(args[-1], Tree) and args[-1].data == "parent_type_name":
            parent_line, parent_name = args[-1].children[0]
            args = args[:-1]
        else:
            parent_line, parent_name = -1, 'object'

        for lineno, typedef in args:
            assert typedef is not AutoType, 'AutoType is not allowed in type definition.'

        for arg in args:
            arg_line, arg_name = arg
            if arg_line == parent_line:
                self.domain.define_type(arg_name, parent_name)
            else:
                self.domain.define_type(arg_name, parent_name)

    @inline_args
    def constants_definition(self, *args):
        for arg in args:
            self.domain.constants[arg.name] = arg

    @inline_args
    def predicate_definition(self, name, *args):
        name, kwargs = name

        return_type = kwargs.pop('return_type', BOOL)
        self._predicate_definition_inner(name, args, return_type, kwargs)

    @inline_args
    def predicate_definition2(self, name, *args):
        name, kwargs = name
        assert 'return_type' not in kwargs
        args, return_type = args[:-1], args[-1]

        return_type = return_type
        self._predicate_definition_inner(name, args, return_type, kwargs)

    def _predicate_definition_inner(self, name, args, return_type, kwargs):
        generators = kwargs.pop('generators', None)
        predicate_def = self.domain.define_predicate(name, args, return_type, **kwargs)
        self._define_inplace_generators(predicate_def, generators)

    def _define_inplace_generators(self, predicate: Predicate, generators) -> List[Generator]:
        defined_generators = list()
        if generators is not None:
            generators: List[str]
            for target_variable_name in generators:
                assert target_variable_name.startswith('?')
                parameters, certifies, context, generates = _canonicalize_inline_generator_def_predicate(self.domain, target_variable_name, predicate)
                generator_name = f'gen-{predicate.name}-{target_variable_name[1:]}' if len(generators) > 1 else f'gen-{predicate.name}'
                generator = self.domain.define_generator(generator_name, parameters, certifies, context, generates)
                defined_generators.append(generator)
        return defined_generators

    @inline_args
    def predicate_name(self, name, kwargs=None):
        if kwargs is None:
            kwargs = dict()
        return name.value, kwargs

    @inline_args
    def type_name(self, name):
        if name.value == 'auto':
            return name.line, AutoType
        # propagate the "lineno" of the type definition up.
        return name.line, name.value

    @inline_args
    def object_type_name(self, typedef):
        return typedef

    @inline_args
    def value_type_name(self, typedef):
        lineno, typedef = typedef
        if typedef is AutoType:
            return lineno, AutoType
        if isinstance(typedef, VectorValueType):
            return lineno, typedef
        assert isinstance(typedef, str)
        return lineno, self.domain.get_type(typedef)

    @inline_args
    def vector_type_name(self, dtype, dim, choices, kwargs=None):
        choices = choices.children[0] if len(choices.children) > 0 else 0
        if kwargs is None:
            kwargs = dict()
        lineno, dtype = dtype
        return lineno, VectorValueType(dtype, dim, choices, **kwargs)

    @inline_args
    def list_object_type_name(self, dtype):
        lineno, dtype = dtype
        return lineno, ListType(self.domain.get_type(dtype))

    @inline_args
    def list_value_type_name(self, dtype):
        lineno, dtype = dtype
        return lineno, ListType(dtype)

    @inline_args
    def object_type_name_unwrapped(self, typedef):
        return typedef[1]

    @inline_args
    def value_type_name_unwrapped(self, typedef):
        return typedef[1]

    @inline_args
    def predicate_group_definition(self, *args):
        raise NotImplementedError()

    @inline_args
    def action_definition(self, name, *defs):
        name, kwargs = name

        parameters = tuple()
        precondition = None
        effect = None
        controller = None

        for def_ in defs:
            if isinstance(def_, _ParameterListWrapper):
                parameters = def_.parameters
            elif isinstance(def_, _PreconditionWrapper):
                precondition = def_.precondition
            elif isinstance(def_, _EffectWrapper):
                effect = def_.effect
            elif isinstance(def_, _ControllerWrapper):
                controller = def_.controller
            else:
                raise TypeError('Unknown definition type: {}.'.format(type(def_)))

        if precondition is not None:
            with ExpressionDefinitionContext(*parameters, domain=self.domain, scope=f"action::{name}").as_default():
                precondition = _canonicalize_precondition(precondition)
        else:
            precondition = list()

        if effect is not None:
            with ExpressionDefinitionContext(*parameters, domain=self.domain, scope=f"action::{name}", is_effect_definition=True).as_default():
                effect = _canonicalize_effect(effect)
        else:
            effect = list()

        if controller is not None:
            with ExpressionDefinitionContext(*parameters, domain=self.domain, scope=f"action::{name}").as_default():
                controller = _canonicalize_controller(controller)

        self.domain.define_operator(name, parameters, precondition, effect, controller, **kwargs)

    @inline_args
    def action_definition2(self, name, extends, *defs):
        name, kwargs = name

        assert 'extends' not in kwargs, 'Instantiation cannot be set using decorators. Use :extends instead.'
        kwargs['extends'] = extends

        template_op = self.domain.operators[extends]
        parameters = template_op.arguments

        precondition = None
        effect = None
        controller = None

        for def_ in defs:
            if isinstance(def_, _ParameterListWrapper):
                parameters += def_.parameters
            elif isinstance(def_, _PreconditionWrapper):
                precondition = def_.precondition
            elif isinstance(def_, _EffectWrapper):
                effect = def_.effect
            elif isinstance(def_, _ControllerWrapper):
                controller = def_.controller
            else:
                raise TypeError('Unknown definition type: {}.'.format(type(def_)))

        if precondition is not None:
            with ExpressionDefinitionContext(*parameters, domain=self.domain, scope=f"action::{name}").as_default():
                precondition = tuple(_canonicalize_precondition(precondition))
        else:
            precondition = tuple()
        if effect is not None:
            with ExpressionDefinitionContext(*parameters, domain=self.domain, scope=f"action::{name}", is_effect_definition=True).as_default():
                effect = tuple(_canonicalize_effect(effect))
        else:
            effect = tuple()
        if controller is not None:
            with ExpressionDefinitionContext(*parameters, domain=self.domain, scope=f"action::{name}").as_default():
                controller = _canonicalize_controller(controller)
        else:
            controller = template_op.controller

        self.domain.define_operator(name, parameters, template_op.preconditions + precondition, template_op.effects + effect, controller, **kwargs)

    def action_parameters(self, args):
        return _ParameterListWrapper(tuple(args))

    @inline_args
    def action_precondition(self, function_call):
        return _PreconditionWrapper(function_call)

    @inline_args
    def action_effect(self, function_call):
        return _EffectWrapper(function_call)

    @inline_args
    def action_controller(self, function_call):
        return _ControllerWrapper(function_call)

    @inline_args
    def action_name(self, name, kwargs=None):
        if kwargs is None:
            kwargs = dict()
        return name.value, kwargs

    @inline_args
    def action_instantiates(self, name):
        return name.value

    @inline_args
    def regression_definition(self, name, *defs):
        name, kwargs = name
        kwargs.setdefault('always', self.defaults['default_regression_rule_always'])

        parameters = tuple()
        precondition = None
        goal = None
        side_effect = None
        body = None

        for def_ in defs:
            if isinstance(def_, _ParameterListWrapper):
                parameters = def_.parameters
            elif isinstance(def_, _PreconditionWrapper):
                precondition = def_.precondition
            elif isinstance(def_, _EffectWrapper) and not def_.is_side_effect:
                goal = def_.effect
            elif isinstance(def_, _EffectWrapper) and def_.is_side_effect:
                side_effect = def_.effect
            else:
                assert body is None, 'Multiple bodies are not allowed.'
                body = def_

        assert goal is not None and body is not None, 'Both goal and body must be defined.'

        if precondition is not None:
            with ExpressionDefinitionContext(*parameters, domain=self.domain, slot_functions_are_sgc=True, scope=f"regression::{name}").as_default():
                precondition = tuple(_canonicalize_precondition(precondition))
        else:
            precondition = tuple()

        if side_effect is not None:
            with ExpressionDefinitionContext(*parameters, domain=self.domain, slot_functions_are_sgc=True, scope=f"regression::{name}").as_default():
                side_effect = tuple(_canonicalize_effect(side_effect))
        else:
            side_effect = tuple()

        with ExpressionDefinitionContext(*parameters, domain=self.domain, scope=f"regression::{name}", is_effect_definition=False).as_default():
            goal_expression = goal.compose()

        # if isinstance(goal, _FunctionApplicationImm):  # Simple goal
        #     with ExpressionDefinitionContext(*parameters, domain=self.domain, scope=f"regression::{name}", is_effect_definition=True).as_default():
        #         goal = tuple(_canonicalize_effect(goal))
        # else:
        #     goal = tuple()

        ctx = ExpressionDefinitionContext(*parameters, domain=self.domain, slot_functions_are_sgc=True, scope=f"regression::{name}")
        with ctx.as_default():
            last_achieve = tuple()
            body_statements = list()
            for i, statement in enumerate(body):
                if isinstance(statement, _FunctionApplicationImm):
                    if statement.name == 'achieve':
                        if len(statement.arguments) == 1:
                            new_subgoal = statement.arguments[0].compose(BOOL)
                            this_subgoal = (new_subgoal, last_achieve)
                        elif len(statement.arguments) == 2:
                            new_subgoal = statement.arguments[0].compose(BOOL)
                            this_subgoal = (new_subgoal, statement.arguments[1].compose(BOOL))
                        else:
                            raise ValueError('Invalid achieve statement.')
                        item_kwargs = statement.kwargs
                        item_kwargs.setdefault('serializability', self.defaults['default_regression_rule_serializability'])
                        item_kwargs.setdefault('csp_serializability', self.defaults['default_regression_rule_csp_serializability'])
                        body_statements.append(AchieveExpression(*this_subgoal, **item_kwargs))
                        last_achieve = last_achieve + (new_subgoal, )
                    elif statement.name == 'regress':
                        if len(statement.arguments) == 1:
                            regression_rule_call = statement.arguments[0]
                            maintains = tuple()
                        elif len(statement.arguments) == 2:
                            regression_rule_call = statement.arguments[0]
                            maintains = tuple(statement.arguments[1].children)
                        else:
                            raise ValueError('Invalid regress statement.')

                        regression_rule = self.domain.regression_rules[statement.name]
                        item_kwargs = statement.kwargs
                        item_kwargs.setdefault('serializability', self.defaults['default_regression_rule_serializability'])
                        item_kwargs.setdefault('csp_serializability', self.defaults['default_regression_rule_csp_serializability'])
                        expression = _canonicalize_regression_rule_expression_arguments(regression_rule, statement.arguments, item_kwargs)
                        body_statements.append(expression)
                    else:
                        if statement.name in self.domain.operators:
                            operator = self.domain.operators[statement.name]
                            if statement.annotation.get('specialize', None) is not None:
                                specialize = statement.annotation['specialize']
                                specialize_kwargs = {'sampler': False, 'controller': False}
                                for key in specialize.split(','):
                                    assert key in specialize_kwargs, f'Unknown specialization key: {key}'
                                    specialize_kwargs[key.strip()] = True
                                operator = self._make_macro_sub_operator(name, operator, i, **specialize_kwargs)

                            expression = _canonicalize_operator_expression_arguments(operator, statement.arguments)
                            body_statements.append(expression)
                        elif statement.name in self.domain.regression_rules:
                            regression_rule = self.domain.regression_rules[statement.name]
                            item_kwargs = statement.kwargs
                            item_kwargs.setdefault('serializability', self.defaults['default_regression_rule_serializability'])
                            item_kwargs.setdefault('csp_serializability', self.defaults['default_regression_rule_csp_serializability'])
                            expression = _canonicalize_regression_rule_expression_arguments(regression_rule, statement.arguments, item_kwargs)
                            body_statements.append(expression)
                        elif statement.name == 'list-expand':
                            expression = statement.arguments[0].compose(self.domain.get_type('__totally_ordered_plan__'))
                            body_statements.append(E.ListExpansionExpression(expression))
                        else:
                            raise ValueError('Unknown operator or regression rule: {}.'.format(statement.name))
                elif isinstance(statement, _QuantifierApplicationImm):
                    if statement.quantifier == 'find':
                        variables = statement.darg
                        ctx.add_variables(*variables)
                        expression = statement.expr.compose(BOOL)
                        body_statements.append(BindExpression(variables, expression, **statement.kwargs))
                    else:
                        raise ValueError('Unknown quantifier: {}.'.format(statement.quantifier))
                elif isinstance(statement, RegressionCommitFlag):
                    body_statements.append(statement)
                else:
                    raise ValueError('Unknown statement: {}.'.format(statement))
            body = tuple(body_statements)

        self.domain.define_regression_rule(name, parameters, precondition, goal_expression, side_effect, body, **kwargs)

    def regression_parameters(self, args):
        return _ParameterListWrapper(tuple(args))

    @inline_args
    def regression_precondition(self, function_call):
        return _PreconditionWrapper(function_call)

    @inline_args
    def regression_goal(self, function_call):
        return _EffectWrapper(function_call)

    @inline_args
    def regression_side_effect(self, function_call):
        return _EffectWrapper(function_call, is_side_effect=True)

    @inline_args
    def regression_body(self, *body_statements):
        return body_statements

    @inline_args
    def regression_name(self, name, kwargs=None):
        if kwargs is None:
            kwargs = dict()
        return name.value, kwargs

    @inline_args
    def cspcommitflag(self, *args):
        if len(args) > 0:
            kwargs = args[0]
            return RegressionCommitFlag(**kwargs)
        return RegressionCommitFlag()

    @inline_args
    def axiom_definition(self, decorator, vars, context, implies):
        kwargs = dict() if len(decorator.children) == 0 else decorator.children[0]
        vars = tuple(vars.children)
        precondition = context.children[0]
        effect = implies.children[0]

        name = kwargs.pop('name', None)
        scope = None if name is None else f"axiom::{name}"

        with ExpressionDefinitionContext(*vars, domain=self.domain, scope=scope).as_default():
            precondition = _canonicalize_precondition(precondition)
        with ExpressionDefinitionContext(*vars, domain=self.domain, scope=scope, is_effect_definition=True).as_default():
            effect = _canonicalize_effect(effect)
        self.domain.define_axiom(name, vars, precondition, effect, **kwargs)

    @inline_args
    def macro_definition(self, signature, body):
        ((name, kwargs), parameters) = signature

        sub_operators = list()
        parameters_map = {p.name: p for p in parameters}
        for i, sub_call in enumerate(body):
            op_name = sub_call.name
            assert op_name in self.domain.operators, f'Unknown operator {op_name} in macro definition.'
            sub_op = self.domain.operators[op_name]
            sub_op = self._make_macro_sub_operator(name, sub_op, i)
            sub_parameters = _canonicalize_simple_operator_expression_arguments(sub_op, sub_call.arguments, parameters_map)
            sub_operators.append(sub_op(*sub_parameters))

        self.domain.define_macro(name, parameters, sub_operators, **kwargs)

    def _make_macro_sub_operator(self, macro_name: str, sub_op: Operator, index: int, sampler: bool = True, controller: bool = False):
        assert sampler is True and controller is False, 'Currently we have only implemented specialization for samplers and no-specialization for controllers'
        # TODO(Jiayuan Mao @ 2024/01/18): implement specialization for samplers and controllers.

        sub_op = sub_op.rename(f'{macro_name}-{index}-{sub_op.name}', is_primitive=False)

        new_preconditions = list()
        for i, precondition in enumerate(sub_op.preconditions):
            expr = precondition.bool_expr
            if isinstance(expr, FunctionApplicationExpression):
                if expr.function.is_generator_placeholder:
                    new_function = expr.function.rename(f'{macro_name}-{index}-{expr.function.name}')
                    self.domain.define_predicate_inner(new_function.name, new_function)
                    for generator in self._define_inplace_generators(new_function, new_function.inplace_generators):
                        self.domain.declare_external_function_crossref(
                            f'generator::{generator.name}',
                            f'generator::{generator.name.replace(new_function.name, expr.function.name)}'
                        )
                    new_preconditions.append(Precondition(FunctionApplicationExpression(
                        new_function,
                        expr.arguments
                    )))
                else:
                    new_preconditions.append(precondition)
            else:
                new_preconditions.append(precondition)

        sub_op.preconditions = tuple(new_preconditions)
        self.domain.define_operator_inner(sub_op.name, sub_op)

        return sub_op

    @inline_args
    def macro_signature(self, name, *args):
        return name, args

    @inline_args
    def macro_name(self, name, kwargs=None):
        if kwargs is None:
            kwargs = dict()
        return name.value, kwargs

    @inline_args
    def macro_content(self, *body_statements):
        return body_statements

    @inline_args
    def extended_macro_definition(self, name, *defs):
        name, kwargs = name

        parameters = tuple()
        precondition = None
        effect = None
        body = None

        for def_ in defs:
            if isinstance(def_, _ParameterListWrapper):
                parameters = def_.parameters
            elif isinstance(def_, _PreconditionWrapper):
                precondition = def_.precondition
            elif isinstance(def_, _EffectWrapper):
                effect = def_.effect
            elif isinstance(def_, _ControllerWrapper):
                body = def_.controller
            else:
                raise TypeError('Unknown definition type: {}.'.format(type(def_)))

        if precondition is not None:
            with ExpressionDefinitionContext(*parameters, domain=self.domain, scope=f"action::{name}").as_default():
                precondition = _canonicalize_precondition(precondition)
        else:
            precondition = list()

        if effect is not None:
            with ExpressionDefinitionContext(*parameters, domain=self.domain, scope=f"action::{name}", is_effect_definition=True).as_default():
                effect = _canonicalize_effect(effect)
        else:
            effect = list()

        if body is None:
            raise ValueError('Body must be defined.')

        parameters_map = {p.name: p for p in parameters}
        sub_operators = list()
        for i, sub_call in enumerate(body):
            op_name = sub_call.name
            assert op_name in self.domain.operators, f'Unknown operator {op_name} in macro definition.'
            sub_op = self.domain.operators[op_name]
            sub_op = self._make_macro_sub_operator(name, sub_op, i)
            sub_parameters = _canonicalize_simple_operator_expression_arguments(sub_op, sub_call.arguments, parameters_map)
            sub_operators.append(sub_op(*sub_parameters))

        self.domain.define_macro(name, parameters, sub_operators, preconditions=precondition, effects=effect, **kwargs)

    @inline_args
    def extended_macro_name(self, name, kwargs=None):
        if kwargs is None:
            kwargs = dict()
        return name.value, kwargs

    def extended_macro_parameters(self, args):
        return _ParameterListWrapper(tuple(args))

    @inline_args
    def extended_macro_precondition(self, function_call):
        return _PreconditionWrapper(function_call)

    @inline_args
    def extended_macro_effect(self, function_call):
        return _EffectWrapper(function_call)

    @inline_args
    def extended_macro_body(self, *body_statements):
        return _ControllerWrapper(body_statements)

    @inline_args
    def derived_definition(self, signature, expr):
        name, args, kwargs = signature
        expr = expr

        return_type = kwargs.pop('return_type', BOOL)
        with ExpressionDefinitionContext(*args, domain=self.domain, scope=name).as_default():
            if return_type is AutoType:
                expr = expr.compose()
                assert isinstance(expr, (E.VariableExpression, E.ValueOutputExpression))
                return_type = expr.return_type
            else:
                expr = expr.compose(return_type)
        self.domain.define_derived(name, args, return_type, expr=expr, **kwargs)

    @inline_args
    def derived_signature1(self, name, *args):
        name, kwargs = name
        return name, args, kwargs

    @inline_args
    def derived_signature2(self, name, *args):
        name, kwargs = name
        assert 'return_type' not in kwargs, 'Return type cannot be set using decorators.'
        kwargs['return_type'] = args[-1]
        return name, args[:-1], kwargs

    @inline_args
    def derived_name(self, name, kwargs=None):
        if kwargs is None:
            kwargs = dict()
        return name.value, kwargs

    @inline_args
    def generator_definition(self, name, parameters, certifies, context, generates):
        name, kwargs = name
        parameters = tuple(parameters.children)
        certifies = certifies.children[0]
        context = context.children[0]
        generates = generates.children[0]

        ctx = ExpressionDefinitionContext(*parameters, domain=self.domain, scope=f"generator::{name}")
        with ctx.as_default():
            certifies = certifies.compose(BOOL)
            assert context.name == 'and'
            context = [_compose(ctx, c) for c in context.arguments]
            assert generates.name == 'and'
            generates = [_compose(ctx, c) for c in generates.arguments]

        self.domain.define_generator(name, parameters, certifies, context, generates, **kwargs)

    @inline_args
    def fancy_generator_definition(self, name, parameters, certifies):
        name, kwargs = name
        certifies = certifies.children[0]

        ctx = ExpressionDefinitionContext(domain=self.domain, scope=f"generator::{name}")
        with ctx.as_default():
            certifies = certifies.compose(BOOL)

        self.domain.define_fancy_generator(name, certifies, **kwargs)

    @inline_args
    def generator_name(self, name, kwargs=None):
        if kwargs is None:
            kwargs = dict()
        return name.value, kwargs

    @inline_args
    def objects_definition(self, *constants):
        for constant in constants:
            assert constant.dtype != AutoType, f'AutoType is not allowed in object definition. Got AutoType for {constant.name}.'
            self.problem.add_object(constant.name, constant.typename)

    @inline_args
    def init_definition_item(self, function_call):
        if function_call.name not in self.domain.functions:
            if self.ignore_unknown_predicates:
                if function_call.name not in self.ignored_predicates:
                    logger.warning(f"Unknown predicate: {function_call.name}.")
                    self.ignored_predicates.add(function_call.name)
            else:
                raise ValueError(f"Unknown predicate: {function_call.name}.")
            return
        self.problem.add_proposition(function_call.compose())

    @inline_args
    def goal_definition(self, function_call):
        self.problem.set_goal(function_call.compose())

    @inline_args
    def variable(self, name) -> Variable:
        return Variable(name.value, AutoType)

    @inline_args
    def typedvariable(self, name, typename):
        # name is of type `Variable`.
        if typename is AutoType:
            return Variable(name.name, AutoType)
        return Variable(name.name, self.domain.get_type(typename) if isinstance(typename, str) else typename)

    @inline_args
    def quantifiedvariable(self, quantifier, variable):
        variable.set_quantifier_flag(quantifier.value)
        return variable

    @inline_args
    def constant(self, name) -> ObjectConstant:
        assert self.allow_object_constants
        return ObjectConstant(name.value, AutoType)

    @inline_args
    def typedconstant(self, name, typename):
        return ObjectConstant(name.name, self.domain.get_type(typename))

    @inline_args
    def bool(self, v):
        return v.value == 'true'

    @inline_args
    def int(self, v):
        return int(v.value)

    @inline_args
    def float(self, v):
        return float(v.value)

    @inline_args
    def string(self, v):
        return v.value[1:-1]

    @inline_args
    def list(self, *args):
        return list(args)

    @inline_args
    def decorator_k(self, k):
        return k.value

    @inline_args
    def decorator_v(self, v):
        return v

    @inline_args
    def decorator_kwarg(self, k, v=True):
        return (k, v)

    def decorator_kwargs(self, args):
        return {k: v for k, v in args}

    @inline_args
    def slot(self, _, name, kwargs=None):
        return _Slot(name.children[0].value, kwargs)

    @inline_args
    def function_name(self, name, kwargs=None):
        return name, kwargs

    @inline_args
    def method_name(self, predicate_name, _, method_name):
        return _MethodName(predicate_name, method_name), None

    @inline_args
    def function_call(self, name, *args):
        name, kwargs = name
        if isinstance(name, (_MethodName, _Slot)):
            return _FunctionApplicationImm(name, args, kwargs=kwargs)
        else:
            return _FunctionApplicationImm(name.value, args, kwargs=kwargs)

    @inline_args
    def annotated_function_call(self, annotation, function_call):
        return function_call.set_annotation(annotation)

    @inline_args
    def function_annotation(self, kwargs):
        return kwargs

    @inline_args
    def list_construction(self, *args):
        return _FunctionApplicationImm('list', args)

    @inline_args
    def list_expansion(self, e, args):
        return _FunctionApplicationImm('list-expand', (args, ))

    @inline_args
    def simple_function_call(self, name, *args):
        name, kwargs = name
        return _FunctionApplicationImm(name.value, args, kwargs=kwargs)

    @inline_args
    def pm_function_call(self, pm_sign, function_call):
        if pm_sign.value == '+':
            return function_call
        else:
            return _FunctionApplicationImm('not', [function_call])

    @inline_args
    def quantifier_name(self, name, kwargs=None):
        return name, kwargs

    @inline_args
    def quantified_function_call(self, quantifier, *args):
        *variables, expr = args
        quantifier, kwargs = quantifier
        return _QuantifierApplicationImm(quantifier, variables, expr, kwargs=kwargs)

    @inline_args
    def ellipsis(self):
        return Ellipsis


class _ParameterListWrapper(collections.namedtuple('_ParameterListWrapper', 'parameters')):
    pass


class _PreconditionWrapper(collections.namedtuple('_PreconditionWrapper', 'precondition')):
    pass


class _EffectWrapper(collections.namedtuple('_EffectWrapper', 'effect,is_side_effect', defaults=(False,))):
    pass


class _ControllerWrapper(collections.namedtuple('_ControllerWrapper', 'controller')):
    pass


class _FunctionApplicationImm(object):
    def __init__(self, name, arguments, kwargs=None):
        self.name = name
        self.arguments = arguments
        self.kwargs = kwargs if kwargs is not None else dict()
        self.annotation = dict()

    def __str__(self):
        arguments_str = ', '.join([str(arg) for arg in self.arguments])
        return f'IMM::{self.name}({arguments_str})'

    __repr__ = jacinle.repr_from_str

    def set_annotation(self, annotation):
        self.annotation = annotation
        return self

    @_log_function
    def compose(self, expect_value_type: Optional[Union[ObjectType, TensorValueTypeBase]] = None):
        ctx = get_expression_definition_context()
        domain: Domain = ctx.domain

        if isinstance(self.name, _Slot):
            assert ctx.scope is not None, 'Cannot define slots inside anonymous actino/axioms.'

            name = ctx.scope + '::' + self.name.name
            arguments = self._compose_arguments(ctx, self.arguments)
            argument_types = [arg.return_type for arg in arguments]
            return_type = self.name.kwargs.pop('return_type', None)
            if return_type is None:
                assert expect_value_type is not None, f'Cannot infer return type for function {name}; please specify by [return_type=Type]'
                return_type = expect_value_type
            else:
                if expect_value_type is not None:
                    assert return_type == expect_value_type, f'Return type mismatch for function {name}: expect {expect_value_type}, got {return_type}.'
            kwargs = self.name.kwargs
            kwargs.setdefault('is_sgc_function', ctx.slot_functions_are_sgc)

            function_def = domain.declare_external_function(name, argument_types, return_type, kwargs=kwargs)

            # if _has_list_arguments(arguments):
            #     return E.ListFunctionApplicationExpression(function_def, arguments)
            # TODO(Jiayuan Mao @ 2023/11/18): handle List arguments correctly.
            return E.FunctionApplicationExpression(function_def, arguments)
        elif isinstance(self.name, _MethodName):
            assert self.name.predicate_name in domain.functions, 'Unkwown predicate: {}.'.format(self.name.predicate_name)
            predicate_def = domain.functions[self.name.predicate_name]

            if self.name.method_name == 'equal':
                nr_index_arguments = len(self.arguments) - 1
            elif self.name.method_name == 'assign':
                nr_index_arguments = len(self.arguments) - 1
            elif self.name.method_name == 'cond-select':
                nr_index_arguments = len(self.arguments) - 1
            elif self.name.method_name == 'cond-assign':
                nr_index_arguments = len(self.arguments) - 2
            else:
                raise NameError('Unknown method name: {}.'.format(self.name.method_name))

            arguments = self._compose_arguments(ctx, self.arguments[:nr_index_arguments], predicate_def.arguments, is_variable_list=True)
            with ctx.mark_is_effect_definition(False):
                value = self._compose_arguments(ctx, [self.arguments[-1]], predicate_def.return_type.assignment_type())[0]

            feature = E.FunctionApplicationExpression(predicate_def, arguments)

            if self.name.method_name == 'equal':
                return E.PredicateEqualExpression(feature, value)
            elif self.name.method_name == 'assign':
                return E.AssignExpression(feature, value)
            elif self.name.method_name == 'cond-select':
                with ctx.mark_is_effect_definition(False):
                    condition = self._compose_arguments(ctx, [self.arguments[-1]], BOOL)[0]
                return E.ConditionalSelectExpression(feature, condition)
            elif self.name.method_name == 'cond-assign':
                with ctx.mark_is_effect_definition(False):
                    condition = self._compose_arguments(ctx, [self.arguments[-2]], BOOL)[0]
                return E.ConditionalAssignExpression(feature, value, condition)
            else:
                raise NameError('Unknown method name: {}.'.format(self.name.method_name))
        elif self.name == 'list':
            arguments = self._compose_arguments(ctx, self.arguments)
            return E.ListCreationExpression(arguments)
        elif self.name == 'and':
            arguments = [arg.compose(expect_value_type) for arg in self.arguments]
            return E.AndExpression(*arguments)
        elif self.name == 'or':
            arguments = [arg.compose(expect_value_type) for arg in self.arguments]
            return E.OrExpression(*arguments)
        elif self.name == 'not':
            arguments = [arg.compose(expect_value_type) for arg in self.arguments]
            return E.NotExpression(*arguments)
        elif self.name == 'equal':
            assert len(self.arguments) == 2, 'PredicateEqualOp takes two arguments, got: {}.'.format(len(self.arguments))
            feature = self.arguments[0]
            feature = _compose(ctx, feature, None)
            value = self.arguments[1]
            value = _compose(ctx, value, feature.return_type.assignment_type())
            return E.PredicateEqualExpression(feature, value)
        elif self.name == 'assign':
            assert len(self.arguments) == 2, 'AssignOp takes two arguments, got: {}.'.format(len(self.arguments))
            assert isinstance(self.arguments[0], _FunctionApplicationImm)
            feature = self.arguments[0].compose(None)
            assert isinstance(feature, E.FunctionApplicationExpression)
            with ctx.mark_is_effect_definition(False):
                value = _compose(ctx, self.arguments[1], feature.return_type.assignment_type())
            return E.AssignExpression(feature, value)
        else:  # the name is a predicate name.
            if self.name in domain.functions or ctx.allow_auto_predicate_def:
                if self.name not in domain.functions:
                    arguments: List[Union[E.ValueOutputExpression, E.VariableExpression]] = self._compose_arguments(ctx, self.arguments, None)
                    argument_types = [arg.return_type for arg in arguments]
                    argument_defs = list()
                    for i, (arg, arg_type) in enumerate(zip(arguments, argument_types)):
                        if isinstance(arg, E.ValueOutputExpression):
                            argument_defs.append(Variable(f'?arg{i}', arg_type))
                        elif isinstance(arg, E.VariableExpression):
                            argument_defs.append(arg.variable)
                        else:
                            raise TypeError(f'Unexpected argument type: {type(arg)}. Can only be ValueOutputExpression or VariableExpression.')
                    self.kwargs.setdefault('state', False)
                    self.kwargs.setdefault('observation', False)
                    generators = self.kwargs.pop('generators', None)
                    self.kwargs['inplace_generators'] = generators
                    predicate_def = domain.define_predicate(self.name, argument_defs, BOOL, **self.kwargs)

                    if _has_list_arguments(arguments):
                        rv = E.ListPredicateExpression(predicate_def, arguments)
                    else:
                        rv = E.FunctionApplicationExpression(predicate_def, arguments)
                    logger.info(f'Auto-defined predicate {self.name} with arguments {argument_defs} and return type {BOOL}.')

                    # create generators inline
                    if generators is not None:
                        generators: List[str]
                        for i, target_variable_name in enumerate(generators):
                            assert target_variable_name.startswith('?')
                            parameters, context, generates = _canonicalize_inline_generator_def(ctx, target_variable_name, arguments)
                            generator_name = f'gen-{self.name}-{target_variable_name[1:]}' if len(generators) > 1 else f'gen-{self.name}'
                            domain.define_generator(generator_name, parameters=parameters, certifies=rv, context=context, generates=generates)
                    return rv
                else:
                    assert len(self.kwargs) == 0, 'Cannot specify decorators for non-auto predicate definition.'
                    predicate_def = domain.functions[self.name]
                    arguments = self._compose_arguments(ctx, self.arguments, predicate_def.arguments, is_variable_list=True)

                    if _has_list_arguments(arguments):
                        return E.ListFunctionApplicationExpression(predicate_def, arguments)
                    return E.FunctionApplicationExpression(predicate_def, arguments)
            else:
                raise ValueError('Unknown function: {}.'.format(self.name))

    def _compose_arguments(self, ctx, arguments, expect_value_type=None, is_variable_list: bool = False) -> List[E.Expression]:
        if isinstance(expect_value_type, (tuple, list)):
            assert len(expect_value_type) == len(arguments), 'Mismatched number of arguments: expect {}, got {}. Expression: {}.'.format(len(expect_value_type), len(arguments), self)

            if is_variable_list:
                output_list = list()
                for arg, var in zip(arguments, expect_value_type):
                    rv = _compose(ctx, arg, var.dtype if var.dtype is not AutoType else None)
                    if var.dtype is AutoType:
                        var.dtype = rv.return_type
                    output_list.append(rv)
                return output_list

            return [_compose(ctx, arg, evt) for arg, evt in zip(arguments, expect_value_type)]
        return [_compose(ctx, arg, expect_value_type) for arg in arguments]


def _compose(ctx, arg, evt=None):
    if isinstance(arg, Variable):
        return ctx[arg.name]
    elif isinstance(arg, ObjectConstant):
        return E.ObjectConstantExpression(arg)
    else:
        return arg.compose(evt)


def _has_list_arguments(arguments):
    for arg in arguments:
        if arg.return_type.is_list_type:
            return True
    return False


class _Slot(object):
    def __init__(self, name, kwargs=None):
        self.scope = None
        self.name = name
        self.kwargs = kwargs

        if self.kwargs is None:
            self.kwargs = dict()

    def set_scope(self, scope):
        self.scope = scope

    def __str__(self):
        kwargs = ', '.join([f'{k}={v}' for k, v in self.kwargs.items()])
        return f'??{self.name}[{kwargs}]'


class _MethodName(object):
    def __init__(self, predicate_name, method_name):
        self.predicate_name = predicate_name
        self.method_name = method_name

    def __str__(self):
        return f'{self.predicate_name}::{self.method_name}'


class _QuantifierApplicationImm(object):
    def __init__(self, quantifier, darg: Tuple[Variable, ...], expr: _FunctionApplicationImm, kwargs: Optional[Dict[str, Any]] = None):
        self.quantifier = quantifier
        self.darg = darg
        self.expr = expr
        self.kwargs = kwargs if kwargs is not None else dict()
        self.annotation = dict()

    def __str__(self):
        return f'QIMM::{self.quantifier}({self.darg}: {self.expr})'

    __repr__ = jacinle.repr_from_str

    def set_annotation(self, annotation):
        self.annotation = annotation
        return self

    @_log_function
    def compose(self, expect_value_type: Optional[TensorValueTypeBase] = None):
        ctx = get_expression_definition_context()

        assert len(self.darg) == 1, 'Quantifier can only take one argument, got {}.'.format(len(self.darg))
        with ctx.new_variables(self.darg[0]):
            if ctx.is_effect_definition:
                expr = _canonicalize_effect(self.expr)
            else:
                expr = self.expr.compose(expect_value_type)

        if self.quantifier in ('foreach', 'forall') and ctx.is_effect_definition:
            outputs = list()
            for e in expr:
                assert isinstance(e, Effect)
                outputs.append(E.DeicticAssignExpression(self.darg[0], e.assign_expr))
            return outputs
        if self.quantifier == 'foreach':
            assert E.is_value_output_expression(expr)
            return E.DeicticSelectExpression(self.darg[0], expr)

        assert E.is_value_output_expression(expr), f'Expect value output expression, got {expr}.'
        return E.QuantificationExpression(E.QuantificationOpType.from_string(self.quantifier), self.darg[0], expr)


@_log_function
def _canonicalize_precondition(precondition: Union[_FunctionApplicationImm, _QuantifierApplicationImm]):
    if isinstance(precondition, _FunctionApplicationImm) and precondition.name == 'and':
        return list(itertools.chain(*[_canonicalize_precondition(pre) for pre in precondition.arguments]))
    return [Precondition(precondition.compose(BOOL), **precondition.annotation)]


@_log_function
def _canonicalize_effect(effect: Union[_FunctionApplicationImm, _QuantifierApplicationImm]):
    if isinstance(effect, _QuantifierApplicationImm):
        output_effect = effect.compose()
        if isinstance(output_effect, list):
            output_effect = [Effect(e, **effect.annotation) for e in output_effect]
    else:
        assert isinstance(effect, _FunctionApplicationImm)

        if effect.name == 'and':
            return list(itertools.chain(*[_canonicalize_effect(eff) for eff in effect.arguments]))

        if isinstance(effect.name, _MethodName):
            output_effect = effect.compose()
        elif effect.name == 'assign':
            output_effect = effect.compose()
        elif effect.name == 'not':
            assert len(effect.arguments) == 1, 'NotOp only takes 1 argument, got {}.'.format(len(effect.arguments))
            feat = effect.arguments[0].compose()
            assert feat.return_type == BOOL
            output_effect = E.AssignExpression(feat, E.ConstantExpression.FALSE)
        elif effect.name == 'when':
            assert len(effect.arguments) == 2, 'WhenOp takes two arguments, got: {}.'.format(len(effect.arguments))
            condition = effect.arguments[0].compose(BOOL)
            if effect.arguments[1].name == 'and':
                inner_effects = effect.arguments[1].arguments
            else:
                inner_effects = [effect.arguments[1]]
            inner_effects = list(itertools.chain(*[_canonicalize_effect(arg) for arg in inner_effects]))
            output_effect = list()
            for e in inner_effects:
                assert isinstance(e.assign_expr, E.AssignExpression)
                output_effect.append(Effect(E.ConditionalAssignExpression(e.assign_expr.predicate, e.assign_expr.value, condition), **e.annotation))
        else:
            feat = effect.compose()
            assert isinstance(feat, E.FunctionApplicationExpression) and feat.return_type == BOOL
            output_effect = E.AssignExpression(feat, E.ConstantExpression.TRUE)

    if isinstance(output_effect, list):
        return output_effect

    assert isinstance(output_effect, E.VariableAssignmentExpression)
    return [Effect(output_effect, **effect.annotation)]


def _canonicalize_controller(controller: _FunctionApplicationImm):
    controller_name = controller.name
    assert isinstance(controller_name, str)

    ctx = get_expression_definition_context()
    arguments = [_compose(ctx, arg, None) for arg in controller.arguments]
    return Implementation(controller_name, arguments)


def _canonicalize_operator_expression_arguments(operator: Operator, arguments: List[Union[_FunctionApplicationImm, Variable, EllipsisType]]) -> OperatorApplicationExpression:
    ctx = get_expression_definition_context()
    if arguments[-1] is Ellipsis:
        arguments = [_compose(ctx, arg, operator.arguments[i]) for i, arg in enumerate(arguments[:-1])]
        arguments = arguments + [UnnamedPlaceholder(var.dtype) for var in operator.arguments[len(arguments):]]
    else:
        assert len(operator.arguments) == len(arguments), 'Operator {} takes {} arguments, got {}.'.format(operator.name, len(operator.arguments), len(arguments))
        arguments = [_compose(ctx, arg, operator.arguments[i]) for i, arg in enumerate(arguments)]
    return OperatorApplicationExpression(operator, arguments)


def _canonicalize_simple_operator_expression_arguments(operator: Operator, arguments: List[Union[Variable, EllipsisType]], parameters_map: Dict[str, Variable]) -> List[Union[Variable, UnnamedPlaceholder]]:
    """The major difference between this function and `_canonicalize_operator_expression_arguments` is that this function returns
    a list of combined [Variable, UnnamedPlaceholder] instances instead of a list of [Expression, UnnamedPlaceholder] instances.
    In the old-style macro definitions, the body of the macro is stored as a list of :class:`~concepts.dm.pdsketch.operator.OperatorApplier` instances,
    instead of an actual :class:`~concepts.dm.pdsketch.operator.OperatorApplicationExpression` instances."""

    for p in arguments:
        if isinstance(p, Variable) and p.name not in parameters_map:
            raise ValueError(f'Unknown parameter {p.name} in macro body definition for operator {operator.name}.')

    sub_parameters = [
        parameters_map[p.name] if isinstance(p, Variable) else p
        for p in arguments
    ]
    if len(sub_parameters) > 0 and sub_parameters[-1] is Ellipsis:
        sub_parameters = sub_parameters[:-1] + [
            UnnamedPlaceholder(x.dtype) for x in operator.arguments[len(sub_parameters) - 1:]
        ]
    return sub_parameters


def _canonicalize_regression_rule_expression_arguments(regression_rule: RegressionRule, arguments: List[_FunctionApplicationImm], kwargs: Dict[str, Any]):
    ctx = get_expression_definition_context()
    assert len(regression_rule.arguments) == len(arguments), 'Regression rule {} takes {} arguments, got {}.'.format(regression_rule.name, len(regression_rule.arguments), len(arguments))
    arguments = [_compose(ctx, arg, regression_rule.arguments[i]) for i, arg in enumerate(arguments)]
    return RegressionRuleApplicationExpression(regression_rule, arguments, **kwargs)


def _canonicalize_inline_generator_def(ctx: ExpressionDefinitionContext, variable_name: str, arguments: List[Union[E.VariableExpression, E.ValueOutputExpression]]):
    parameters, context, generates = list(), list(), list()
    used_parameters = set()

    for arg in arguments:
        if isinstance(arg, E.VariableExpression):
            if arg.name not in used_parameters:
                used_parameters.add(arg.name)
                parameters.append(arg.variable)
            if arg.name == variable_name:
                generates.append(arg)
            else:
                context.append(arg)
        else:
            context.append(arg)
            assert isinstance(arg, E.ValueOutputExpression)
            for sub_expr in iter_exprs(arg):
                if isinstance(sub_expr, E.VariableExpression):
                    if sub_expr.name not in used_parameters:
                        used_parameters.add(sub_expr.name)
                        parameters.append(sub_expr.variable)

    assert len(generates) == 1, f'Generator must generate exactly one variable, got {len(generates)}.'
    return parameters, context, generates


def _canonicalize_inline_generator_def_predicate(domain: Domain, variable_name: str, predicate_def: Predicate):
    parameters = predicate_def.arguments
    context, generates = list(), list()

    if predicate_def.return_type != BOOL:
        for arg in parameters:
            assert arg.name != '?rv', 'Arguments cannot be named ?rv.'
    parameters = tuple(parameters) + (Variable('?rv', predicate_def.return_type),)

    ctx = ExpressionDefinitionContext(*parameters, domain=domain)
    with ctx.as_default():
        for arg in predicate_def.arguments:
            assert isinstance(arg, Variable)
            if arg.name == variable_name:
                generates.append(ctx[arg.name])
            else:
                context.append(ctx[arg.name])
        certifies = E.FunctionApplicationExpression(predicate_def, [ctx[arg.name] for arg in predicate_def.arguments])
        if predicate_def.return_type != BOOL:
            context.append(ctx['?rv'])
            certifies = E.PredicateEqualExpression(certifies, ctx['?rv'])

    assert len(generates) == 1, f'Generator must generate exactly one variable, got {len(generates)}.'
    return parameters, certifies, context, generates

