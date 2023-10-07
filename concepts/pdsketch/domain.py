#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : domain.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/30/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import itertools
from typing import TYPE_CHECKING, Any, Optional, Union, Sequence, Tuple, List, Dict

import jacinle
import torch
from jacinle.utils.printing import indent_text, stprint

from concepts.dsl.dsl_types import BOOL, FLOAT32, INT64, ObjectType, TensorValueTypeBase, PyObjValueType, ScalarValueType, VectorValueType, NamedTensorValueType, Variable, ObjectConstant
from concepts.dsl.dsl_functions import FunctionArgumentType, FunctionReturnType, FunctionType, Function
from concepts.dsl.dsl_domain import DSLDomainBase
from concepts.dsl.expression import Expression, ExpressionDefinitionContext, VariableExpression, ValueOutputExpression, cvt_expression_list
from concepts.dsl.expression import FunctionApplicationExpression, AssignExpression, ConditionalAssignExpression, DeicticAssignExpression
from concepts.dsl.constraint import is_optimistic_value
from concepts.dsl.tensor_state import NamedObjectTensorState

from concepts.pdsketch.predicate import Predicate, flatten_expression, get_used_state_variables
from concepts.pdsketch.operator import Precondition, Effect, Controller, Operator, MacroOperator, OperatorApplier, OperatorApplicationExpression, AchieveExpression, RegressionRule
from concepts.pdsketch.generator import Generator, FancyGenerator

if TYPE_CHECKING:
    from concepts.pdsketch.executor import PDSketchExecutor

logger = jacinle.get_logger(__file__)

__all__ = ['Domain', 'Problem', 'State']


class _TypedVariableView(object):
    """Use `domain.typed_variable['type_name']('variable_name')`"""

    def __init__(self, domain):
        self.domain = domain

    def __getitem__(self, typename):
        def function(string):
            return Variable(string, self.domain.types[typename])
        return function


class Domain(DSLDomainBase):
    """The planning domain definition."""

    def __init__(self, name: Optional[str] = None):
        """Initialize a planning domain.

        Args:
            name: The name of the domain.
        """
        super().__init__(name)

        self.operators = dict()
        self.operator_templates = dict()
        self.regression_rules = dict()
        self.axioms = dict()
        self.generators = dict()
        self.fancy_generators = dict()

        self.external_functions = dict()
        self.external_function_crossrefs = dict()

        self.tv = self.typed_variable = _TypedVariableView(self)

    name: str

    types: Dict[str, Union[ObjectType, PyObjValueType, TensorValueTypeBase]]
    """The types defined in the domain, as a dictionary from type names to types."""

    functions: Dict[str, Predicate]
    """A mapping from function name to the corresponding :class:`~concepts.pdsketch.predicate.Predicate` class. Note that,
    unlike the basic :class:`~concepts.dsl.dsl_domain.DSLDomainBase`, in planning domain, all functions should be of type :class:`~concepts.pdsketch.predicate.Predicate`."""

    constants: Dict[str, ObjectConstant]
    """The constants defined in the domain, as a dictionary from constant names to values."""

    operators: Dict[str, Union[Operator, MacroOperator]]
    """A mapping of operators: from operator name to the corresponding :class:`~concepts.pdsketch.operator.Operator` class."""

    operator_templates: Dict[str, Operator]
    """A mapping of operator templates: from operator name to the corresponding :class:`~concepts.pdsketch.operator.Operator` class."""

    regression_rules: Dict[str, RegressionRule]
    """A mapping of regression rules: from regression rule name to the corresponding :class:`~concepts.pdsketch.operator.RegressionRule` class."""

    axioms: Dict[str, Operator]
    """A mapping of axioms: from axiom name to the corresponding :class:`~concepts.pdsketch.operator.Operator` class."""

    generators: Dict[str, Generator]
    """A mapping of generators: from generator name to the corresponding :class:`~concepts.pdsketch.generator.Generator` class."""

    fancy_generators: Dict[str, FancyGenerator]
    """A mapping of fancy generators: from fancy generator name to the corresponding :class:`~concepts.pdsketch.generator.FancyGenerator` class."""

    external_functions: Dict[str, Function]
    """A mapping of external functions: from function name to the corresponding :class:`~concepts.dsl.dsl_functions.Function` class."""

    external_function_crossrefs: Dict[str, str]
    """A mapping from function name to another function name. This is useful when defining one function as an derived function of another function."""

    tv: _TypedVariableView
    """A helper function that returns a variable with the given type.
    For example, `domain.tv['object']('x')` returns a variable of type `object` with name `x`."""

    def __getattr__(self, item):
        if item.startswith('__') and item.endswith('__'):
            raise AttributeError

        # NB(Jiayuan Mao @ 09/03): PDDL definition convention.
        item = item.replace('_', '-')

        if item.startswith('t-'):
            return self.types[item[2:]]
        elif item.startswith('p-') or item.startswith('f-'):
            return self.functions[item[2:]]
        elif item.startswith('op-'):
            return self.operators[item[3:]]
        elif item.startswith('gen-'):
            return self.generators[item[4:]]
        raise NameError('Unknown attribute: {}.'.format(item))

    def set_name(self, name: str):
        """Set the name of the domain.

        Args:
            name: the new name of the domain.
        """
        self.name = name

    BUILTIN_TYPES = ['object', 'pyobject', 'bool', 'int64', 'float32']
    BUILTIN_NUMERIC_TYPES = {
        'bool': BOOL,
        'int64': INT64,
        'float32': FLOAT32
    }

    def define_type(self, typename, parent_name: Optional[Union[VectorValueType, ScalarValueType, str]] = 'object') -> Union[ObjectType, PyObjValueType, VectorValueType, ScalarValueType]:
        """Define a new type.

        Args:
            typename: the name of the new type.
            parent_name: the parent type of the new type, default to 'object'.

        Returns:
            the newly defined type.
        """

        if typename == 'object':
            logger.warning_once('Shadowing built-in type name "object".')
        elif typename in type(self).BUILTIN_TYPES:
            raise ValueError('Typename {} is a built-in type.'.format(typename))

        assert isinstance(parent_name, (str, VectorValueType)), f'Currently only support inheritance from builtin types: {type(self).BUILTIN_TYPES}.'

        if isinstance(parent_name, str):
            if parent_name == 'object':
                self.types[typename] = ObjectType(typename)
            elif parent_name == 'pyobject':
                dtype = PyObjValueType(typename)
                self.types[typename] = dtype
                self.declare_external_function(f'type::{typename}::equal', [dtype, dtype], BOOL)
            elif parent_name == 'int64':
                dtype = NamedTensorValueType(typename, INT64)
                self.types[typename] = dtype
                self.declare_external_function(f'type::{typename}::equal', [dtype, dtype], BOOL)
            elif parent_name == 'float32':
                dtype = NamedTensorValueType(typename, FLOAT32)
                self.types[typename] = dtype
                self.declare_external_function(f'type::{typename}::equal', [dtype, dtype], BOOL)
            else:
                raise ValueError(f'Unknown parent type: {parent_name}.')
        elif isinstance(parent_name, VectorValueType):
            dtype = NamedTensorValueType(typename, parent_name)
            self.types[typename] = dtype
            self.declare_external_function(f'type::{typename}::equal', [dtype, dtype], BOOL)
        else:
            raise ValueError(f'Unknown parent type: {parent_name}.')

        return self.types[typename]

    def get_type(self, typename: str) -> Union[ObjectType, PyObjValueType, VectorValueType, ScalarValueType, NamedTensorValueType]:
        """Get a type by name.

        Args:
            typename: the name of the type.

        Returns:
            the type with the given name.
        """
        if typename in type(self).BUILTIN_NUMERIC_TYPES:
            return type(self).BUILTIN_NUMERIC_TYPES[typename]
        if typename not in self.types:
            raise ValueError(f'Unknown type: {typename}.')
        return self.types[typename]

    def define_predicate(
        self, name: str, arguments: Sequence[FunctionArgumentType], return_type: FunctionReturnType = BOOL,
        observation: Optional[bool] = None, state: Optional[bool] = None,
        generator_placeholder: bool = False, inplace_generators: Optional[Sequence[str]] = None,
    ):
        """Define a new predicate.

        Args:
            name: the name of the new predicate.
            arguments: the arguments of the new predicate.
            return_type: the return type of the new predicate.
            observation: whether the new predicate is an observation variable.
            state: whether the new predicate is a state variable.
            generator_placeholder: whether the new predicate is a generator placeholder.

        Returns:
            the newly defined predicate.
        """
        predicate = Predicate(name, FunctionType(arguments, return_type), observation=observation, state=state, generator_placeholder=generator_placeholder, inplace_generators=inplace_generators)
        self.define_predicate_inner(name, predicate)
        return predicate

    def define_derived(self, name: str, arguments: Sequence[FunctionArgumentType], return_type: Optional[FunctionReturnType] = None, expr: ValueOutputExpression = None, state: bool = False, generator_placeholder: bool = False):
        """Define a new derived predicate. Note that a derived predicate can not be an observation variable.

        Args:
            name: the name of the new derived predicate.
            arguments: the arguments of the new derived predicate.
            return_type: the return type of the new derived predicate.
            expr: the expression of the new derived predicate.
            state: whether the new derived predicate is a state variable.
            generator_placeholder: whether the new derived predicate is a generator placeholder.

        Returns:
            the newly defined derived predicate.
        """
        predicate_def = Predicate(name, FunctionType(arguments, return_type), observation=False, state=state, generator_placeholder=generator_placeholder, derived_expression=expr)
        return self.define_predicate_inner(name, predicate_def)

    def define_predicate_inner(self, name, predicate_def: Predicate):
        self.functions[name] = predicate_def

        # NB(Jiayuan Mao @ 07/21): a non-cacheable function is basically an external function.
        if not predicate_def.is_cacheable and predicate_def.derived_expression is None:
            identifier = f'predicate::{name}'
            self.external_functions[identifier] = predicate_def

        return predicate_def

    def define_operator(self, name: str, parameters: Sequence[Variable], preconditions: Sequence[Precondition], effects: Sequence[Effect], controller: Controller, template=False, extends=None) -> Operator:
        """Define a new operator.

        Args:
            name: the name of the new operator.
            parameters: the parameters of the new operator.
            preconditions: the preconditions of the new operator.
            effects: the effects of the new operator.
            controller: the controller of the new operator.
            template: whether the new operator is a template.
            extends: the parent operator of the new operator.

        Returns:
            the newly defined operator.
        """
        self.operators[name] = op = Operator(name, parameters, preconditions, effects, controller, extends=extends, is_template=template)
        return op

    def define_operator_inner(self, name: str, operator: Operator) -> Operator:
        assert name not in self.operators
        self.operators[name] = operator
        return operator

    def define_regression_rule(
        self, name: str, parameters: Sequence[Variable],
        preconditions: Sequence[Precondition], preconstraints: Sequence[Precondition],
        goal: Sequence[Effect], goal_expression: ValueOutputExpression, body: Sequence[Union[OperatorApplicationExpression, AchieveExpression]],
        always: bool = False
    ):
        """Define a new regression rule.

        Args:
            name: the name of the new regression rule.
            parameters: the parameters of the new regression rule.
            preconditions: the preconditions of the new regression rule.
            preconstraints: the preconstraints of the new regression rule.
            goal: the goal of the new regression rule.
            body: the body of the new regression rule.
            always: whether the new regression rule is always applicable.

        Returns:
            the newly defined regression rule.
        """
        self.regression_rules[name] = rule = RegressionRule(name, parameters, preconditions, preconstraints, goal, goal_expression, body, always=always)
        return rule

    def define_axiom(self, name: Optional[str], parameters: Sequence[Variable], preconditions: Sequence[Precondition], effects: Sequence[Effect]) -> Operator:
        """Define a new axiom.

        Args:
            name: the name of the new axiom. If None, a unique name will be generated.
            parameters: the parameters of the new axiom.
            preconditions: the preconditions of the new axiom.
            effects: the effects of the new axiom.

        Returns:
            the newly defined axiom.
        """

        if name is None:
            name = f'axiom_{len(self.axioms)}'
        self.axioms[name] = op = Operator(name, parameters, preconditions, effects, is_axiom=True)
        return op

    def define_macro(self, name: str, parameters: Sequence[Variable], sub_operators: Sequence[OperatorApplier]):
        """Define a new macro.

        Args:
            name: the name of the new macro.
            parameters: the parameters of the new macro.
            sub_operators: the sub operators of the new macro.

        Returns:
            the newly defined macro.
        """
        self.operators[name] = op = MacroOperator(name, parameters, sub_operators)
        return op

    def define_generator(
        self, name: str, parameters: Sequence[Variable], certifies: ValueOutputExpression,
        context: Sequence[Union[VariableExpression, ValueOutputExpression]], generates: Sequence[Union[VariableExpression, ValueOutputExpression]],
        priority: int = 0, unsolvable: bool = False
    ) -> Generator:
        """Define a new generator.

        Args:
            name: the name of the new generator.
            parameters: the parameters of the new generator.
            certifies: the certified condition of the new generator.
            context: the context of the new generator.
            generates: the generates of the new generator.
            priority: the priority of the new generator.
            unsolvable: whether the new generator is unsolvable.

        Returns:
            the newly defined generator.
        """

        if unsolvable:
            priority = int(1e9)

        identifier = f'generator::{name}'
        context: List[Union[VariableExpression, ValueOutputExpression]] = cvt_expression_list(context)
        generates: List[Union[VariableExpression, ValueOutputExpression]] = cvt_expression_list(generates)

        arguments = [Variable(f'?c{i}', c.return_type) for i, c in enumerate(context)]
        return_type = [target.return_type for target in generates]
        output_vars = [Variable(f'?g{i}', g.return_type) for i, g in enumerate(generates)]
        function = Function(identifier, FunctionType(arguments, return_type))

        all_variables = {c: cv for c, cv in zip(context, arguments)}
        all_variables.update({g: gv for g, gv in zip(generates, output_vars)})
        ctx = ExpressionDefinitionContext(*arguments, *output_vars, domain=self)
        flatten_certifies = flatten_expression(certifies, all_variables, ctx, flatten_cacheable_expression=True)

        if not unsolvable:
            self.external_functions[identifier] = function

        if name in self.generators:
            raise ValueError(f'Duplicate generator: {name}.')
        self.generators[name] = generator = Generator(
            name, parameters, certifies,
            context=context, generates=generates,
            function=function, output_vars=output_vars, flatten_certifies=flatten_certifies,
            priority=priority, unsolvable=unsolvable
        )
        return generator

    def define_fancy_generator(
        self, name: str, certifies: ValueOutputExpression,
        priority: int = 10, unsolvable: bool = False
    ) -> Generator:
        """Declare a new fancy generator. The difference between a fancy generator and a normal generator is that
        a fancy generator is not directional. That is, it can generate a set of variables satisfies the constraints,
        without requring specific `contexts` to `genereates` directions. Therefore, we don't need to specify the
        `context` and `generates` of a fancy generator.

        Args:
            name: the name of the new fancy generator.
            certifies: the certified condition of the new fancy generator.
            priority: the priority of the new fancy generator.
            unsolvable: whether the new fancy generator is unsolvable.

        Returns:
            the newly declared fancy generator.
        """
        if unsolvable:
            priority = int(1e9)

        identifier = f'generator::{name}'
        # TODO(Jiayuan Mao @ 2023/04/04): fix the typing for this.
        function = Function(identifier, FunctionType([], []))
        flatten_certifies = certifies

        if not unsolvable:
            self.external_functions[identifier] = function

        if name in self.generators:
            raise ValueError(f'Duplicate generator: {name}.')
        self.fancy_generators[name] = generator = FancyGenerator(name, certifies, function=function, flatten_certifies=flatten_certifies, priority=priority, unsolvable=unsolvable)
        return generator

    def declare_external_function(self, function_name: str, argument_types: Sequence[FunctionArgumentType], return_type: FunctionArgumentType, kwargs: Optional[Dict[str, Any]] = None) -> Function:
        """Declare an external function.

        Args:
            function_name: the name of the external function.
            argument_types: the argument types of the external function.
            return_type: the return type of the external function.
            kwargs: the keyword arguments of the external function. Supported keyword arguments are:
                - ``observation``: whether the external function is an observation variable.
                - ``state``: whether the external function is a state variable.
        """
        if kwargs is None:
            kwargs = dict()

        self.external_functions[function_name] = Predicate(function_name, FunctionType(argument_types, return_type), **kwargs)
        return self.external_functions[function_name]

    def declare_external_function_crossref(self, function_name: str, cross_ref_name: str):
        """Declare a cross reference to an external function.
        This is useful when one function is an derived function of another function.

        Args:
            function_name: the name of the external function.
            cross_ref_name: the name of the cross reference.
        """
        self.external_function_crossrefs[function_name] = cross_ref_name

    def parse(self, string: Union[str, Expression], variables: Optional[Sequence[Variable]] = None) -> Expression:
        """Parse a string into an expression.

        Args:
            string: the string to be parsed.
            variables: the variables to be used in the expression.

        Returns:
            the parsed expression.
        """
        if isinstance(string, Expression):
            return string

        from concepts.pdsketch.parsers.pdsketch_parser import parse_expression
        return parse_expression(self, string, variables)

    def incremental_define(self, string: str):
        """Incrementally define new parts of the domain.

        Args:
            string: the string to be parsed and defined.
        """
        from concepts.pdsketch.parsers.pdsketch_parser import load_domain_string_incremental
        load_domain_string_incremental(self, string)

    def print_summary(self):
        """Print a summary of the domain."""
        print(f'Domain {self.name}')
        stprint(key='Types: ', data=self.types, indent=1)
        stprint(key='Functions: ', data=self.functions, indent=1)
        stprint(key='External Functions: ', data=self.external_functions, indent=1)
        stprint(key='Generators: ', data=self.generators, indent=1)
        stprint(key='Fancy Generators: ', data=self.fancy_generators, indent=1)
        print('  Operators:')
        if len(self.operators) > 0:
            for op in self.operators.values():
                print(indent_text(op.pddl_str(), level=2))
        else:
            print('    <Empty>')
        print('  Axioms:')
        if len(self.axioms) > 0:
            for op in self.axioms.values():
                print(indent_text(op.pddl_str(), level=2))
        else:
            print('    <Empty>')
        print('  Regression Rules:')
        if len(self.regression_rules) > 0:
            for op in self.regression_rules.values():
                print(indent_text(op.pddl_str(), level=2))
        else:
            print('    <Empty>')

    def post_init(self):
        """Post-initialization of the domain.
        This function should be called by the domain generator after all the domain definitions (predicates and operators) are done.
        Currently, the following post-initialization steps are performed:

        1. Analyze the static predicates.
        """
        self._analyze_static_predicates()

    def _analyze_static_predicates(self):
        """Run static analysis on the predicates to determine which predicates are static."""
        dynamic = set()
        for op in itertools.chain(self.operators.values(), self.axioms.values()):
            if isinstance(op, MacroOperator):
                continue
            for eff in op.effects:
                if isinstance(eff.assign_expr, (AssignExpression, ConditionalAssignExpression)):
                    dynamic.add(eff.assign_expr.predicate.function.name)
                elif isinstance(eff.assign_expr, DeicticAssignExpression):
                    expr = eff.unwrapped_assign_expr
                    assert isinstance(expr, (AssignExpression, ConditionalAssignExpression))
                    dynamic.add(expr.predicate.function.name)
                else:
                    raise TypeError(f'Unknown effect type: {eff.assign_expr}.')

        # propagate the static predicates.
        for p in self.functions.values():
            if p.is_state_variable:
                p.mark_static(p.name not in dynamic)
            else:
                if p.is_cacheable and p.derived_expression is not None:
                    used_predicates = get_used_state_variables(p.derived_expression)
                    static = True
                    for predicate_def in used_predicates:
                        if not predicate_def.is_static:
                            static = False
                            break
                    p.mark_static(static)


class Problem(object):
    """The representation for a planning problem. It contains the set of objects, a inital state (a set of propositions), and a goal expression."""

    def __init__(self):
        """Initialize the problem."""
        self.objects = dict()
        self.predicates = list()
        self.goal = None

    objects: Dict[str, str]
    """The set of objects, which are mappings from object names to object type names."""

    predicates: List[FunctionApplicationExpression]
    """The initial state, which is a set of propositions."""

    goal: Optional[ValueOutputExpression]
    """The goal expression."""

    def add_object(self, name: str, typename: str):
        """Add an object to the problem.

        Args:
            name: the name of the object.
            typename: the type of the object.
        """
        self.objects[name] = typename

    def add_proposition(self, proposition: FunctionApplicationExpression):
        """Add a proposition to the initial problem.

        Args:
            proposition: the proposition to add.
        """
        self.predicates.append(proposition)

    def set_goal(self, goal: ValueOutputExpression):
        """Set the goal of the problem.

        Args:
            goal: the goal expression.
        """
        self.goal = goal

    def to_state(self, executor: 'PDSketchExecutor') -> 'State':
        """Convert the problem to a :class:`State` object.

        Args:
            executor: the executor to use to instantiate the state.

        Returns:
            the state object.
        """

        domain = executor.domain
        object_names = list(self.objects.keys())
        object_types = [executor.domain.types[self.objects[name]] for name in object_names]

        for constant in domain.constants.values():
            object_names.append(constant.name)
            object_types.append(constant.dtype)

        state = State(None, object_names, object_types)

        from concepts.pdsketch.executor import StateDefinitionHelper
        ctx = StateDefinitionHelper(executor, state)
        predicates = list()
        for p in self.predicates:
            predicates.append(ctx.get_predicate(p.function.name)(*[arg.constant.name for arg in p.arguments]))
        ctx.define_predicates(predicates)

        return state


class State(NamedObjectTensorState):
    """Planning domain state."""

    def make_quantized(self, domain: Domain, features=None) -> 'State':
        """Make a quantized version of the state.

        Args:
            domain: the planning domain.
            features: the features to use for quantization. If None, use all state variables.

        Returns:
            the quantized state.
        """
        if features is None:
            features = [name for name in self.features.all_feature_names if domain.functions[name].is_state_variable]

        new_tensor_dict = dict()
        for feature_name in features:
            new_tensor_dict[feature_name] = self.features[feature_name].simple_quantize()
        return type(self)(self.object_types, new_tensor_dict, self.object_names)

    def generate_tuple_description(self, domain: Domain) -> Tuple[int, ...]:
        """Generate a tuple description of the state.

        Args:
            domain: the planning domain.

        Returns:
            the tuple description of the state.
        """
        rv = list()
        for feature_name in sorted(self.features.all_feature_names):
            if domain.functions[feature_name].is_state_variable:
                feature = self.features[feature_name]
                if isinstance(feature.dtype, TensorValueTypeBase) and feature.dtype.is_intrinsically_quantized():
                    rv.extend(_maybe_apply_optimistic_mask(feature.tensor, feature.tensor_optimistic_values).flatten().tolist())
                elif feature.tensor_quantized_values is not None:
                    rv.extend(_maybe_apply_optimistic_mask(feature.tensor_quantized_values, feature.tensor_optimistic_values).flatten().tolist())
                else:
                    raise RuntimeError(f'Cannot generate tuple description for feature {feature_name}.')
        return tuple(rv)


def _maybe_apply_optimistic_mask(tensor, optimistic_values):
    if optimistic_values is None:
        return tensor
    assert tensor.shape == optimistic_values.shape
    optimistic_mask = is_optimistic_value(optimistic_values)
    return torch.where(optimistic_mask, optimistic_values, tensor.to(torch.int64))

