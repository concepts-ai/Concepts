#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : all.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 11/02/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Import all the DSL-related modules.

.. rubric:: Types

.. autosummary::

    ~concepts.dsl.dsl_types.AliasType
    ~concepts.dsl.dsl_types.AutoType
    ~concepts.dsl.dsl_types.AnyType
    ~concepts.dsl.dsl_types.ObjectType
    ~concepts.dsl.dsl_types.ValueType
    ~concepts.dsl.dsl_types.ConstantType
    ~concepts.dsl.dsl_types.PyObjValueType
    ~concepts.dsl.dsl_types.TensorValueTypeBase
    ~concepts.dsl.dsl_types.ScalarValueType
    ~concepts.dsl.dsl_types.STRING
    ~concepts.dsl.dsl_types.BOOL
    ~concepts.dsl.dsl_types.INT64
    ~concepts.dsl.dsl_types.FLOAT32
    ~concepts.dsl.dsl_types.VectorValueType
    ~concepts.dsl.dsl_types.NamedTensorValueType
    ~concepts.dsl.dsl_types.TupleType
    ~concepts.dsl.dsl_types.ListType
    ~concepts.dsl.dsl_types.BatchedListType

.. rubric:: Variable, constant, and slices

.. autosummary::

    ~concepts.dsl.dsl_types.QINDEX
    ~concepts.dsl.dsl_types.Variable
    ~concepts.dsl.dsl_types.ObjectConstant
    ~concepts.dsl.dsl_types.UnnamedPlaceholder

.. rubric:: Function types

.. autosummary::

    ~concepts.dsl.dsl_types.FunctionType
    ~concepts.dsl.dsl_types.OverloadedFunctionType
    ~concepts.dsl.dsl_types.FunctionTyping
    ~concepts.dsl.dsl_types.Function

.. rubric:: Domain

.. autosummary::

    ~concepts.dsl.dsl_domain.DSLDomainBase
    ~concepts.dsl.function_domain.FunctionDomain
    ~concepts.dsl.function_domain.resolve_lambda_function_type

.. rubric:: Values

.. autosummary::

    ~concepts.dsl.tensor_value.TensorValue
    ~concepts.dsl.tensor_value.TensorizedPyObjValues
    ~concepts.dsl.tensor_value.concat_tvalues
    ~concepts.dsl.tensor_value.expand_as_tvalue
    ~concepts.dsl.tensor_value.expand_tvalue
    ~concepts.dsl.value.Value
    ~concepts.dsl.value.ListValue

.. rubric:: State

.. autosummary::

    ~concepts.dsl.tensor_state.StateObjectReference
    ~concepts.dsl.tensor_state.StateObjectList
    ~concepts.dsl.tensor_state.StateObjectDistribution
    ~concepts.dsl.tensor_state.TensorState
    ~concepts.dsl.tensor_state.NamedObjectTensorState
    ~concepts.dsl.tensor_state.concat_states

.. rubric:: Constraints

.. autosummary::

    ~concepts.dsl.constraint.OptimisticValue
    ~concepts.dsl.constraint.Constraint
    ~concepts.dsl.constraint.EqualityConstraint
    ~concepts.dsl.constraint.GroupConstraint
    ~concepts.dsl.constraint.SimulationFluentConstraintFunction
    ~concepts.dsl.constraint.ConstraintSatisfactionProblem
    ~concepts.dsl.constraint.NamedConstraintSatisfactionProblem
    ~concepts.dsl.constraint.AssignmentType
    ~concepts.dsl.constraint.Assignment
    ~concepts.dsl.constraint.AssignmentDict
    ~concepts.dsl.constraint.print_assignment_dict
    ~concepts.dsl.constraint.ground_assignment_value

.. rubric:: Executors

.. autosummary::

    ~concepts.dsl.executors.executor_base.DSLExecutorBase
    ~concepts.dsl.executors.function_domain_executor.FunctionDomainExecutor
    ~concepts.dsl.executors.tensor_value_executor.TensorValueExecutorBase
    ~concepts.dsl.executors.tensor_value_executor.FunctionDomainTensorValueExecutor

.. rubric:: Parsers

.. autosummary::

    ~concepts.dsl.parsers.parser_base.ParserBase
    ~concepts.dsl.parsers.function_expression_parser.FunctionExpressionParser
    ~concepts.dsl.parsers.fol_python_parser.FOLPythonParser

"""

from concepts.dsl.dsl_types import AliasType, AutoType, AnyType, ObjectType, ValueType, ConstantType, PyObjValueType
from concepts.dsl.dsl_types import TensorValueTypeBase, ScalarValueType, STRING, BOOL, INT64, FLOAT32, VectorValueType, NamedTensorValueType
from concepts.dsl.dsl_types import TupleType, ListType, BatchedListType
from concepts.dsl.dsl_types import QINDEX, Variable, ObjectConstant, UnnamedPlaceholder
from concepts.dsl.dsl_functions import FunctionType, OverloadedFunctionType, FunctionTyping, Function
from concepts.dsl.dsl_domain import DSLDomainBase

from concepts.dsl.function_domain import FunctionDomain, resolve_lambda_function_type

from concepts.dsl.tensor_value import TensorValue, TensorizedPyObjValues, concat_tvalues, expand_as_tvalue, expand_tvalue
from concepts.dsl.tensor_state import StateObjectReference, StateObjectList, StateObjectDistribution, TensorState, NamedObjectTensorState, concat_states
from concepts.dsl.value import Value, ListValue

from concepts.dsl.constraint import OptimisticValue, Constraint, EqualityConstraint, GroupConstraint, SimulationFluentConstraintFunction
from concepts.dsl.constraint import ConstraintSatisfactionProblem, NamedConstraintSatisfactionProblem
from concepts.dsl.constraint import AssignmentType, Assignment, AssignmentDict, print_assignment_dict, ground_assignment_value

from concepts.dsl.executors.executor_base import DSLExecutorBase
from concepts.dsl.executors.function_domain_executor import FunctionDomainExecutor
from concepts.dsl.executors.tensor_value_executor import TensorValueExecutorBase, FunctionDomainTensorValueExecutor

from concepts.dsl.parsers.parser_base import ParserBase
from concepts.dsl.parsers.function_expression_parser import FunctionExpressionParser
from concepts.dsl.parsers.fol_python_parser import FOLPythonParser

__all__ = [
    'AliasType', 'AutoType', 'AnyType', 'ObjectType', 'ValueType', 'ConstantType', 'PyObjValueType',
    'TensorValueTypeBase', 'ScalarValueType', 'STRING', 'BOOL', 'INT64', 'FLOAT32', 'VectorValueType', 'NamedTensorValueType',
    'TupleType', 'ListType', 'BatchedListType',
    'QINDEX', 'Variable', 'ObjectConstant', 'UnnamedPlaceholder',
    'FunctionType', 'OverloadedFunctionType', 'FunctionTyping', 'Function',
    'DSLDomainBase',
    'FunctionDomain', 'resolve_lambda_function_type',
    'TensorValue', 'TensorizedPyObjValues', 'concat_tvalues', 'expand_as_tvalue', 'expand_tvalue',
    'StateObjectReference', 'StateObjectList', 'StateObjectDistribution', 'TensorState', 'NamedObjectTensorState', 'concat_states',
    'Value', 'ListValue',
    'OptimisticValue', 'Constraint', 'EqualityConstraint', 'GroupConstraint', 'SimulationFluentConstraintFunction',
    'ConstraintSatisfactionProblem', 'NamedConstraintSatisfactionProblem',
    'AssignmentType', 'Assignment', 'AssignmentDict', 'print_assignment_dict', 'ground_assignment_value',
    'ParserBase',
    'FunctionDomainExecutor',
    'TensorValueExecutorBase', 'FunctionDomainTensorValueExecutor',
    'DSLDomainBase',
    'FunctionExpressionParser',
    'FOLPythonParser',
]
