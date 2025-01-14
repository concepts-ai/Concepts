#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : crow_domain.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/15/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Any, Optional, Union, Sequence, List, Mapping, Dict, TYPE_CHECKING

import torch
from jacinle.logging import get_logger

from concepts.dsl.dsl_types import TypeBase, AliasType
from concepts.dsl.dsl_types import ObjectType, PyObjValueType, TensorValueTypeBase, Variable, ObjectConstant
from concepts.dsl.dsl_types import VectorValueType, ScalarValueType, NamedTensorValueType
from concepts.dsl.dsl_types import BOOL, INT64, FLOAT32, STRING
from concepts.dsl.dsl_functions import Function, FunctionType, FunctionArgumentListType, FunctionReturnType
from concepts.dsl.dsl_domain import DSLDomainBase
from concepts.dsl.constraint import OPTIM_MAGIC_NUMBER_MAGIC
from concepts.dsl.expression import Expression, ValueOutputExpression, NullExpression
from concepts.dsl.tensor_value import TensorValue
from concepts.dsl.tensor_state import NamedObjectTensorState, ObjectNameArgument, ObjectTypeArgument

from concepts.dm.crow.controller import CrowController
from concepts.dm.crow.crow_function import CrowFeature, CrowFunction
from concepts.dm.crow.crow_generator import CrowGeneratorBase, CrowDirectedGenerator, CrowUndirectedGenerator
from concepts.dm.crow.behavior import CrowBehaviorOrderingSuite, CrowBehavior

if TYPE_CHECKING:
    from concepts.dm.crow.executors.crow_executor import CrowExecutor

logger = get_logger(__file__)

__all__ = ['CrowDomain', 'CrowProblem', 'CrowState']


class CrowDomain(DSLDomainBase):
    """The planning domain definition."""

    def __init__(self, name: Optional[str] = None):
        """Initialize a planning domain.

        Args:
            name: The name of the domain.
        """
        super().__init__(name)

        self.features = dict()
        self.functions = dict()
        self.controllers = dict()
        self.behaviors = dict()
        self.generators = dict()

        self.external_functions = dict()
        self.external_function_crossrefs = dict()
        self.external_function_implementation_files = list()

    name: str
    """The name of the domain."""

    types: Dict[str, Union[ObjectType, PyObjValueType, TensorValueTypeBase, AliasType]]
    """The types defined in the domain, as a dictionary from type names to types."""

    functions: Dict[str, CrowFunction]
    """A mapping of functions: from function name to the corresponding :class:`~concepts.dm.crow.function.CrowFunction` class."""

    features: Dict[str, CrowFeature]
    """A mapping of features: from feature name to the corresponding :class:`~concepts.dm.crow.feature.CrowFeature` class."""

    controllers: Dict[str, CrowController]
    """A mapping of controllers: from controller name to the corresponding :class:`~concepts.dm.crow.controller.Controller` class."""

    constants: Dict[str, ObjectConstant]
    """The constants defined in the domain, as a dictionary from constant names to values."""

    behaviors: Dict[str, CrowBehavior]
    """A mapping of behaviors: from behavior name to the corresponding :class:`~concepts.dm.crow.behavior.CrowBehavior` class."""

    generators: Dict[str, CrowGeneratorBase]
    """A mapping of generators: from generator name to the corresponding :class:`~concepts.dm.crow.generator.CrowGeneratorBase` class."""

    external_functions: Dict[str, Function]
    """A mapping of external functions: from function name to the corresponding :class:`~concepts.dsl.dsl_functions.Function` class."""

    external_function_crossrefs: Dict[str, str]
    """A mapping from function name to another function name. This is useful when defining one function as an derived function of another function."""

    external_function_implementation_files: List[str]
    """A list of external function implementation files (e.g., Python libs that implements the external functions)."""

    def set_name(self, name: str):
        """Set the name of the domain.

        Args:
            name: the new name of the domain.
        """
        self.name = name

    BUILTIN_TYPES = ['object', 'pyobject', 'bool', 'int64', 'float32', 'string', '__totally_ordered_plan__']
    BUILTIN_NUMERIC_TYPES = {
        'bool': BOOL,
        'int64': INT64,
        'float32': FLOAT32,
        'string': STRING
    }
    BUILTIN_PYOBJ_TYPES = {
        '__behavior_body__': PyObjValueType('__behavior_body__', alias='__behavior_body__'),
    }

    def clone(self, deep: bool = True):
        if deep:
            raise NotImplementedError('Deep cloning is not supported yet.')

        domain = CrowDomain(self.name)
        domain.types = self.types.copy()
        domain.functions = self.functions.copy()
        domain.features = self.features.copy()
        domain.controllers = self.controllers.copy()
        domain.constants = self.constants.copy()
        domain.behaviors = self.behaviors.copy()
        domain.generators = self.generators.copy()
        domain.external_functions = self.external_functions.copy()
        domain.external_function_crossrefs = self.external_function_crossrefs.copy()
        domain.external_function_implementation_files = self.external_function_implementation_files.copy()
        return domain

    def define_type(self, typename, parent_name: Optional[Union[VectorValueType, ScalarValueType, str]] = 'object') -> Union[ObjectType, PyObjValueType, VectorValueType, ScalarValueType]:
        """Define a new type.

        Args:
            typename: the name of the new type.
            parent_name: the parent type of the new type, default to 'object'.

        Returns:
            the newly defined type.
        """

        if typename == 'object':
            raise ValueError('Typename "object" is reserved.')
        elif typename in type(self).BUILTIN_TYPES:
            raise ValueError('Typename {} is a built-in type.'.format(typename))

        assert isinstance(parent_name, (str, (VectorValueType, TypeBase))), f'Currently only support inheritance from builtin types: {type(self).BUILTIN_TYPES}.'

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
            elif parent_name == 'string':
                dtype = PyObjValueType(typename, parent_type=STRING)
                self.types[typename] = dtype
                self.declare_external_function(f'type::{typename}::equal', [dtype, dtype], BOOL)
            else:
                raise ValueError(f'Unknown parent type: {parent_name}.')
        elif isinstance(parent_name, VectorValueType):
            dtype = NamedTensorValueType(typename, parent_name)
            self.types[typename] = dtype
            self.declare_external_function(f'type::{typename}::equal', [dtype, dtype], BOOL)
        elif isinstance(parent_name, TypeBase):
            dtype = AliasType(typename, parent_name)
            self.types[typename] = dtype
        else:
            raise ValueError(f'Unknown parent type: {parent_name}.')

        return self.types[typename]

    def has_type(self, typename: str):
        """Check whether a type exists.

        Args:
            typename: the name of the type.

        Returns:
            whether the type exists.
        """
        if typename in type(self).BUILTIN_TYPES:
            return True
        if typename in type(self).BUILTIN_PYOBJ_TYPES:
            return True
        return typename in self.types

    def get_type(self, typename: str) -> Union[ObjectType, PyObjValueType, VectorValueType, ScalarValueType, NamedTensorValueType]:
        """Get a type by name.

        Args:
            typename: the name of the type.

        Returns:
            the type with the given name.
        """
        if typename == 'object':
            return ObjectType('object')
        elif typename == 'pyobject':
            return PyObjValueType('pyobject')
        if typename in type(self).BUILTIN_NUMERIC_TYPES:
            return type(self).BUILTIN_NUMERIC_TYPES[typename]
        elif typename in type(self).BUILTIN_PYOBJ_TYPES:
            return type(self).BUILTIN_PYOBJ_TYPES[typename]
        if typename not in self.types:
            raise ValueError(f'Unknown type: {typename}, known types are: {list(self.types.keys())}.')
        return self.types[typename].unwrap_alias()

    def define_object_constant(self, name: str, typename: str) -> ObjectConstant:
        """Define a new object constant.

        Args:
            name: the name of the new constant.
            typename: the type of the new constant.

        Returns:
            the newly defined constant.
        """
        if name in self.constants:
            raise ValueError(f'Constant {name} already exists.')
        self.constants[name] = ObjectConstant(name, self.get_type(typename))
        return self.constants[name]

    def define_feature(
        self, name: str, arguments: FunctionArgumentListType, return_type: FunctionReturnType = BOOL, *,
        derived_expression: Optional[ValueOutputExpression] = None,
        observation: Optional[bool] = None, state: Optional[bool] = None,
        default: Optional[Any] = None,
    ) -> CrowFeature:
        """Define a new feature.

        Args:
            name: the name of the new feature.
            arguments: the arguments of the new feature.
            return_type: the return type of the new feature.
            derived_expression: the derived expression of the new feature.
            observation: whether the new feature is an observation variable.
            state: whether the new feature is a state variable.
            default: the default value of the new feature.

        Returns:
            the newly defined feature.
        """
        if name in self.features:
            raise ValueError(f'Feature {name} already exists.')
        feature = CrowFeature(
            name, FunctionType(arguments, return_type), derived_expression=derived_expression, observation=observation, state=state, default=default
        )
        self.features[name] = feature
        return feature

    def has_feature(self, name: str) -> bool:
        """Check whether a feature exists.

        Args:
            name: the name of the feature.

        Returns:
            whether the feature exists.
        """
        return name in self.features

    def get_feature(self, name: str, allow_function: bool = False) -> CrowFeature:
        """Get a feature by name.

        Args:
            name: the name of the feature.

        Returns:
            the feature with the given name.
        """
        if name not in self.features:
            if allow_function:
                if name in self.functions:
                    return self.functions[name]

            raise ValueError(f'Unknown feature: {name}.')
        return self.features[name]

    def define_crow_function(
        self, name: str, arguments: FunctionArgumentListType, return_type: FunctionReturnType = BOOL, *,
        derived_expression: Optional[ValueOutputExpression] = None,
        generator_placeholder: bool = False, inplace_generators: Optional[Sequence[str]] = None,
        simulation: bool = False, execution: bool = False,
        is_generator_function: bool = False,
    ):
        """Define a new function.

        Args:
            name: the name of the new function.
            arguments: the arguments of the new function.
            return_type: the return type of the new function.
            derived_expression: the derived expression of the new function.
            generator_placeholder: whether the new function is a generator placeholder.
            inplace_generators: a list of generators that will be defined in-place for this function.
            simulation: whether the new function requires the up-to-date simulation state to evaluate.
            execution: whether the new function requires the up-to-date execution state to evaluate.
            is_generator_function: whether the new function is a generator function.

        Returns:
            the newly defined function.
        """
        if name in self.functions:
            raise ValueError(f'Function {name} already exists.')
        function = CrowFunction(
            name, FunctionType(arguments, return_type, is_generator_function=is_generator_function),
            derived_expression=derived_expression,
            generator_placeholder=generator_placeholder, inplace_generators=inplace_generators,
            simulation=simulation, execution=execution
        )
        self.functions[name] = function

        if function.derived_expression is not None:
            self.external_functions[name] = function

        return function

    def has_function(self, name: str) -> bool:
        """Check whether a function exists.

        Args:
            name: the name of the function.

        Returns:
            whether the function exists.
        """
        return name in self.functions

    def get_function(self, name: str) -> CrowFunction:
        """Get a function by name.

        Args:
            name: the name of the function.

        Returns:
            the function with the given name.
        """
        if name not in self.functions:
            raise ValueError(f'Unknown function: {name}.')
        return self.functions[name]

    def define_controller(self, name: str, arguments: Sequence[Variable], effect_body: Optional[CrowBehaviorOrderingSuite] = None, python_effect: bool = False) -> CrowController:
        """Define a new controller.

        Args:
            name: the name of the new controller.
            arguments: the arguments of the new controller.
            effect_body: the effect body of the new controller.

        Returns:
            the newly defined controller.
        """
        if name in self.controllers:
            raise ValueError(f'Controller {name} already exists.')
        controller = CrowController(name, arguments, effect_body=effect_body, python_effect=python_effect)
        self.controllers[name] = controller
        return controller

    def has_controller(self, name: str) -> bool:
        """Check whether a controller exists.

        Args:
            name: the name of the controller.

        Returns:
            whether the controller exists.
        """
        return name in self.controllers

    def get_controller(self, name: str) -> CrowController:
        """Get a controller by name.

        Args:
            name: the name of the controller.

        Returns:
            the controller with the given name.
        """
        if name not in self.controllers:
            raise ValueError(f'Unknown controller: {name}.')
        return self.controllers[name]

    def define_behavior(
        self, name: str, arguments: Sequence[Variable],
        goal: Union[ValueOutputExpression, NullExpression],
        body: CrowBehaviorOrderingSuite,
        effect_body: CrowBehaviorOrderingSuite,
        heuristic: Optional[CrowBehaviorOrderingSuite] = None,
        minimize: Optional[ValueOutputExpression] = None,
        always: bool = False,
        python_effect: bool = False
    ):
        """Define a new behavior.

        Args:
            name: the name of the new behavior.
            arguments: the arguments of the new behavior.
            goal: the goal of the new behavior.
            body: the body of the new behavior.
            effect_body: the effect body of the new behavior.
            heuristic: the heuristic of the new behavior.
            minimize: the minimize condition of the new behavior.
            always: whether the new behavior is always "feasible".
            python_effect: whether the effect is implemented as a Python function.

        Returns:
            the newly defined behavior.
        """
        if name in self.behaviors:
            raise ValueError(f'Behavior {name} already exists.')
        self.behaviors[name] = behavior = CrowBehavior(
            name, arguments, goal, body,
            effect_body=effect_body,
            heuristic=heuristic,
            minimize=minimize,
            always=always,
            python_effect=python_effect
        )
        return behavior

    def has_behavior(self, name: str) -> bool:
        return name in self.behaviors

    def get_behavior(self, name: str) -> CrowBehavior:
        if name not in self.behaviors:
            raise ValueError(f'Behavior {name} not found.')
        return self.behaviors[name]

    def define_generator(
        self, name: str, arguments: Sequence[Variable],
        certifies: Union[Sequence[Union[ValueOutputExpression, NullExpression]], Union[ValueOutputExpression, NullExpression]],
        inputs: Optional[Sequence[Variable]] = None, outputs: Optional[Sequence[Variable]] = None,
        priority: int = 0, simulation: bool = False, execution: bool = False
    ) -> CrowGeneratorBase:
        """Define a new generator.

        Args:
            name: the name of the new generator.
            arguments: the parameters of the new generator.
            certifies: the certified condition of the new generator.
            inputs: the input variables of the new generator.
            outputs: the output variables of the new generator.
            priority: the priority of the new generator.
            simulation: whether the new generator requires the up-to-date simulation state to evaluate.
            execution: whether the new generator requires the up-to-date execution state to evaluate.

        Returns:
            the newly defined generator.
        """
        if name in self.generators:
            raise ValueError(f'Generator {name} already exists.')

        if not isinstance(certifies, (list, tuple)):
            certifies = [certifies]

        if inputs is None and outputs is None:
            generator = CrowUndirectedGenerator(name, arguments, certifies, priority=priority, simulation=simulation, execution=execution)
        else:
            assert inputs is not None and outputs is not None, 'Both inputs and outputs should be specified.'
            generator = CrowDirectedGenerator(name, arguments, certifies, inputs, outputs, priority=priority, simulation=simulation, execution=execution)

        self.generators[name] = generator
        return generator

    def has_generator(self, name: str) -> bool:
        return name in self.generators

    def get_generator(self, name: str) -> CrowGeneratorBase:
        if name in self.generators:
            return self.generators[name]
        raise ValueError(f'Generator {name} not found.')

    def declare_external_function(self, function_name: str, argument_types: FunctionArgumentListType, return_type: FunctionReturnType, kwargs: Optional[Dict[str, Any]] = None) -> Function:
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

        self.external_functions[function_name] = CrowFunction(function_name, FunctionType(argument_types, return_type), **kwargs)
        return self.external_functions[function_name]

    def declare_external_function_crossref(self, function_name: str, cross_ref_name: str):
        """Declare a cross-reference to an external function.
        This is useful when one function is an derived function of another function.

        Args:
            function_name: the name of the external function.
            cross_ref_name: the name of the cross-reference.
        """
        self.external_function_crossrefs[function_name] = cross_ref_name

    def add_external_function_implementation_file(self, file_path: str):
        """Add an external function implementation file. This is useful if we want to declare external functions directly in the domain file.

        Args:
            file_path: the path of the external function implementation file.
        """
        self.external_function_implementation_files.append(file_path)

    def parse(self, string: Union[str, Expression], state: Optional['State'] = None, variables: Optional[Sequence[Variable]] = None) -> Expression:
        """Parse a string into an expression.

        Args:
            string: the string to be parsed.
            variables: the variables to be used in the expression.

        Returns:
            the parsed expression.
        """
        if isinstance(string, Expression):
            return string

        from concepts.dm.crow.parsers.cdl_parser import parse_expression
        return parse_expression(self, string, state=state, variables=variables)

    def make_executor(self) -> 'CrowExecutor':
        """Make an executor for this domain."""
        from concepts.dm.crow.executors.crow_executor import CrowExecutor
        return CrowExecutor(self)

    def incremental_define(self, string: str):
        """Incrementally define new parts of the domain.

        Args:
            string: the string to be parsed and defined.
        """
        from concepts.dm.crow.parsers.cdl_parser import load_domain_string_incremental
        return load_domain_string_incremental(self, string)

    def set_goal_program(self, string: str):
        if '__goal__' in self.behaviors:
            del self.behaviors['__goal__']

        assert '__goal__' in string, 'Goal program should contain a behavior named "__goal__".'
        self.incremental_define(string)

    def incremental_define_file(self, filename: str):
        """Incrementally define new parts of the domain from a file.

        Args:
            filename: the path of the file to be loaded.
        """
        from concepts.dm.crow.parsers.cdl_parser import load_domain_file_incremental
        return load_domain_file_incremental(self, filename)

    def print_summary(self, external_functions: bool = False, full_generators: bool = False):
        """Print a summary of the domain."""
        # TODO(Jiayuan Mao @ 2024/03/15): implement this.

    def post_init(self):
        """Post-initialization of the domain.
        This function should be called by the domain generator after all the domain definitions (predicates and operators) are done.
        Currently, the following post-initialization steps are performed:

        1. Analyze the static predicates.
        """
        self._analyze_static_predicates()

    def _analyze_static_predicates(self):
        """Run static analysis on the predicates to determine which predicates are static."""
        # TODO(Jiayuan Mao @ 2024/03/15): implement this.
        pass


class CrowProblem(object):
    def __init__(self, domain: CrowDomain):
        self.domain = domain
        self.name = None
        self.objects = dict()
        self.state = None
        self.goal = None
        self.planner_options = dict()

    def set_planner_option(self, key: str, value: Any):
        self.planner_options[key] = value

    def get_state_or_init(self) -> 'CrowState':
        self.init_state()
        return self.state

    @classmethod
    def from_state_and_goal(cls, domain: CrowDomain, state: 'CrowState', goal: Optional[Union[ValueOutputExpression, str]] = None):
        problem = cls(domain)
        problem.state = state
        problem.goal = goal
        return problem

    def add_object(self, name: str, typename: str):
        self.objects[name] = typename

    def init_state(self):
        if self.state is not None:
            return

        domain = self.domain
        self.state = CrowState.make_empty_state(domain, self.objects)

    def set_goal(self, goal: Union[ValueOutputExpression, str]):
        self.goal = goal


class CrowState(NamedObjectTensorState):
    simulation_state: Optional[int] = None
    simulation_state_index: int = 0

    def set_simulation_state(self, state: Optional[int], state_index: int):
        self.simulation_state = state
        self.simulation_state_index = state_index

    def init_dirty_feature(self, function: Function):
        """Initialize a dirty feature. A dirty feature is a cacheable feature but not in the original state representation.
        The convention for dirty features is that they are initialized with optimistic values being OPTIM_MAGIC_NUMBER_MAGIC.

        Args:
            function: the feature to initialize.
        """
        feature_name = function.name
        return_type = function.return_type

        if feature_name not in self.features:
            sizes = list()
            for arg_def in function.arguments:
                sizes.append(len(self.object_type2name[arg_def.typename]) if arg_def.typename in self.object_type2name else 0)
            sizes = tuple(sizes)
            self.features[feature_name] = tensor = TensorValue.make_empty(return_type, [var.name for var in function.arguments], sizes)
            tensor.init_tensor_optimistic_values()
            tensor.tensor_optimistic_values.fill_(OPTIM_MAGIC_NUMBER_MAGIC)
            self.internals.setdefault('dirty_features', set()).add(feature_name)

    def batch_set_value(self, feature_name: str, value: Union[torch.Tensor, tuple, list]) -> None:
        if feature_name not in self.features:
            raise ValueError(f'Unknown feature: {feature_name}.')
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=self.features[feature_name].tensor.dtype)
        self.features[feature_name].tensor = value

    def fast_set_value(self, feature_name: str, indices: Sequence[str], value: Any):
        if feature_name not in self.features:
            raise ValueError(f'Unknown feature: {feature_name}.')
        indices = tuple(self.get_typed_index(arg) for arg in indices)
        self.features[feature_name].fast_set_index(indices, value)

    def clone(self) -> 'CrowState':
        rv: CrowState = super().clone()
        rv.set_simulation_state(self.simulation_state, self.simulation_state_index)
        return rv

    @classmethod
    def make_empty_state(cls, domain: CrowDomain, objects: Dict[str, str]):
        object_names = list(objects.keys())
        object_types = [domain.types[objects[name]] for name in object_names]
        state = cls(None, object_names, object_types)

        for feature_name, feature in domain.features.items():
            if not feature.is_state_variable:
                continue
            return_type = feature.return_type

            if feature_name not in state.features:
                sizes = list()
                for arg_def in feature.arguments:
                    sizes.append(len(state.object_type2name[arg_def.typename]) if arg_def.typename in state.object_type2name else 0)
                sizes = tuple(sizes)
                state.features[feature_name] = TensorValue.make_empty(return_type, [var.name for var in feature.arguments], sizes)

                if feature.default is not None:
                    if isinstance(feature.default, (int, float)):
                        state.features[feature_name].tensor.fill_(feature.default)

        return state

    def clone_with_new_objects(self, domain: CrowDomain, object_names: ObjectNameArgument, object_types: ObjectTypeArgument = None):
        """Append objects to the state."""

        if isinstance(object_names, Mapping):
            object_names = list(object_names.keys())
            object_types = list(object_names.values())
        else:
            assert object_types is not None, 'object_types should not be None if object_names is not a mapping.'
            object_names = list(object_names)
            object_types = list(object_types)

        type2names = {t.typename: [] for t in object_types}
        for name, type_ in zip(object_names, object_types):
            type2names[type_.typename].append(name)

        new_features = dict()
        for name, feat in self.features.items():
            feat_def = domain.get_feature(name)
            current_shape = list(feat.variable_shape)
            assert len(current_shape) == len(feat_def.arguments), f'Feature {name} has wrong number of arguments: def={len(feat_def.arguments)} vs actual={len(current_shape)}.'
            found = False
            for i, arg in enumerate(feat_def.arguments):
                if arg.typename in type2names:
                    current_shape[i] += len(type2names[arg.typename])
                    found = True

            if not found:
                new_features[name] = feat.clone()
            else:
                new_features[name] = feat.pad(tuple(current_shape))

        rv = type(self)(features=new_features, object_types=self.object_types + tuple(object_types), object_names=self.object_names + tuple(object_names), batch_dims=self._batch_dims, internals=self.clone_internals())
        rv.set_simulation_state(self.simulation_state, self.simulation_state_index)
        return rv


