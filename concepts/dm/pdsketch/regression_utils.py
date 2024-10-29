#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : regression_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/30/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Utility functions for regression search."""

import itertools
from typing import Any, Optional, Union, Sequence, Tuple, NamedTuple, List, Dict

import jacinle
import torch

from concepts.dsl.dsl_types import ObjectType, ObjectConstant, UnnamedPlaceholder, Variable, QINDEX
from concepts.dsl.constraint import ConstraintSatisfactionProblem, EqualityConstraint, GroupConstraint
from concepts.dsl.constraint import OptimisticValue, AssignmentDict
from concepts.dsl.executors.tensor_value_executor import BoundedVariablesDictCompatible
from concepts.dsl.expression import (
    BoolExpression, FunctionApplicationExpression, ListCreationExpression, ListFunctionApplicationExpression, ListExpansionExpression,
    ConstantExpression, ObjectConstantExpression, QuantificationExpression, ValueOutputExpression, VariableExpression,
    is_and_expr
)
from concepts.dsl.expression_utils import iter_exprs
from concepts.dsl.expression_visitor import IdentityExpressionVisitor
from concepts.dsl.tensor_value import TensorValue
from concepts.dsl.value import ValueBase, ListValue
from concepts.dm.pdsketch.domain import State
from concepts.dm.pdsketch.executor import PDSketchSGC, PDSketchExecutor
from concepts.dm.pdsketch.operator import OperatorApplier, OperatorApplicationExpression
from concepts.dm.pdsketch.regression_rule import RegressionRule, RegressionRuleApplier, RegressionRuleApplicationExpression, AchieveExpression, BindExpression, RuntimeAssignExpression, RegressionCommitFlag
from concepts.dm.pdsketch.crow.crow_state import TotallyOrderedPlan, PartiallyOrderedPlan

__all__ = [
    'surface_fol_downcast', 'ground_fol_expression', 'ground_fol_expression_v2',
    'ground_operator_application_expression', 'ground_regression_application_expression',
    'evaluate_bool_scalar_expression',
    'ApplicableRegressionRuleItem', 'ApplicableRegressionRuleGroup', 'gen_applicable_regression_rules', 'len_candidate_regression_rules',
    'create_find_expression_csp_variable', 'create_find_expression_variable_placeholder',
    'mark_constraint_group_solver',
    'has_optimistic_value_or_list', 'is_single_optimistic_value_or_list', 'cvt_single_optimistic_value_or_list',
    'has_optimistic_constant_expression',
    'map_csp_placeholder_goal', 'map_csp_placeholder_action', 'map_csp_placeholder_regression_rule_applier', 'map_csp_variable_mapping', 'map_csp_variable_state',
    'gen_grounded_subgoals_with_placeholders'
]


def surface_fol_downcast(expression_1: ValueOutputExpression, expression_2: ValueOutputExpression) -> Optional[Dict[str, Union[Variable, ObjectConstant]]]:
    """Trying to downcast the `expression_1` to the same form as `expression_2`. Downcasting means that
    we try to replace variables in `expression_1` with constants in `expression_2` to make them the same.

    Args:
        expression_1: the first expression.
        expression_2: the second expression.

    Returns:
        the downcasted mapping if the downcasting is successful, otherwise None.
    """
    current_mapping = dict()

    # @jacinle.log_function(verbose=True)
    def dfs(expr1, expr2):
        nonlocal current_mapping

        if isinstance(expr1, VariableExpression):
            if expr1.name in current_mapping:
                expr2_name = _get_variable_or_constant_name(expr2)
                return expr2_name == current_mapping[expr1.name].name
            else:
                if isinstance(expr2, VariableExpression):
                    current_mapping[expr1.name] = expr2.variable
                    return True
                elif isinstance(expr2, ObjectConstantExpression):
                    current_mapping[expr1.name] = expr2.constant
                    return True
                elif isinstance(expr2, ConstantExpression):
                    current_mapping[expr1.name] = expr2.constant
                    return True
                elif isinstance(expr2, ListCreationExpression):
                    current_mapping[expr1.name] = ListValue(expr1.return_type, [_get_variable_or_constant_object(x) for x in expr2.arguments])
                    return True
                else:
                    return False
        elif isinstance(expr1, ObjectConstantExpression):
            if isinstance(expr2, ObjectConstantExpression):
                return expr1.name == expr2.name
            else:
                return False
        elif isinstance(expr1, (FunctionApplicationExpression, ListFunctionApplicationExpression)):
            if not isinstance(expr2, (FunctionApplicationExpression, ListFunctionApplicationExpression)):
                return False
            if expr1.function.name != expr2.function.name:
                return False
            if len(expr1.arguments) != len(expr2.arguments):
                return False
            for arg1, arg2 in zip(expr1.arguments, expr2.arguments):
                if not dfs(arg1, arg2):
                    return False
            return True
        elif isinstance(expr1, BoolExpression):
            if not isinstance(expr2, BoolExpression):
                return False
            if expr1.bool_op != expr2.bool_op:
                return False
            if len(expr1.arguments) != len(expr2.arguments):
                return False
            for arg1, arg2 in zip(expr1.arguments, expr2.arguments):
                if not dfs(arg1, arg2):
                    return False
            return True
        elif isinstance(expr1, QuantificationExpression):
            if not isinstance(expr2, QuantificationExpression):
                return False
            if expr1.quantification_op != expr2.quantification_op:
                return False
            assert expr1.variable.name not in current_mapping
            current_mapping[expr1.variable.name] = expr2.variable
            try:
                return dfs(expr1.expression, expr2.expression)
            finally:
                del current_mapping[expr1.variable.name]
        else:
            raise TypeError(f'Unsupported expression type: {type(expr1)}')

    rv = dfs(expression_1, expression_2)
    if rv:
        return current_mapping
    return None


def ground_fol_expression(expression: ValueOutputExpression, variable_mapping: Dict[Variable, str]) -> ValueOutputExpression:
    """Ground the given FOL expression with the given variable mapping.

    Args:
        expression: the expression to ground.
        variable_mapping: the variable mapping, which is a mapping from the Variable object to the constant name.

    Returns:
        the grounded expression.
    """
    name2symbol = dict()
    for var, content in variable_mapping.items():
        if isinstance(content, ListValue):
            if isinstance(content.element_type, ObjectType):
                name2symbol[var.name] = ObjectConstantExpression(content)
            else:
                name2symbol[var.name] = ConstantExpression(content)
        elif isinstance(content, ObjectConstant):
            name2symbol[var.name] = ObjectConstantExpression(content)
        elif isinstance(content, ValueBase):
            name2symbol[var.name] = ConstantExpression(content)
        elif isinstance(content, Variable):
            name2symbol[var.name] = VariableExpression(content)
        elif isinstance(content, str):
            name2symbol[var.name] = ObjectConstantExpression(ObjectConstant(content, var.dtype))
        else:
            raise TypeError(f'Unsupported type: {type(content)}')

    bounded_variables = set()

    def dfs(e):
        if isinstance(e, VariableExpression):
            if e.name not in bounded_variables:
                return name2symbol[e.name]
            return e
        elif isinstance(e, ObjectConstantExpression):
            return e
        elif isinstance(e, ConstantExpression):
            return e
        elif isinstance(e, ListFunctionApplicationExpression):
            return ListFunctionApplicationExpression(e.function, [dfs(arg) for arg in e.arguments])
        elif isinstance(e, FunctionApplicationExpression):
            return FunctionApplicationExpression(e.function, [dfs(arg) for arg in e.arguments])
        elif isinstance(e, BoolExpression):
            return BoolExpression(e.bool_op, [dfs(arg) for arg in e.arguments])
        elif isinstance(e, QuantificationExpression):
            bounded_variables.add(e.variable.name)
            rv = QuantificationExpression(e.quantification_op, e.variable, dfs(e.expression))
            bounded_variables.remove(e.variable.name)
            return rv
        else:
            raise TypeError(f'Unsupported expression type: {type(e)}')

    return dfs(expression)


def ground_fol_expression_v2(expression: ValueOutputExpression, variable_mapping: Dict[str, Union[ListValue, ObjectConstant, ValueBase, Variable]]) -> ValueOutputExpression:
    """Ground the given FOL expression with the given variable mapping.

    Args:
        expression: the expression to ground.
        variable_mapping: the variable mapping, which is a mapping from the Variable object to the constant name.

    Returns:
        the grounded expression.
    """

    name2symbol = dict()
    for var, content in variable_mapping.items():
        if isinstance(content, ListValue):
            if isinstance(content.element_type, ObjectType):
                name2symbol[var] = ObjectConstantExpression(content)
            else:
                name2symbol[var] = ConstantExpression(content)
        elif isinstance(content, ObjectConstant):
            name2symbol[var] = ObjectConstantExpression(content)
        elif isinstance(content, ValueBase):
            name2symbol[var] = ConstantExpression(content)
        elif isinstance(content, Variable):
            name2symbol[var] = VariableExpression(content)
        else:
            raise TypeError(f'Unsupported type: {type(content)}')
    bounded_variables = set()

    def dfs(e):
        if isinstance(e, VariableExpression):
            if e.name not in bounded_variables:
                return name2symbol[e.name]
            return e
        elif isinstance(e, ObjectConstantExpression):
            return e
        elif isinstance(e, ListFunctionApplicationExpression):
            return ListFunctionApplicationExpression(e.function, [dfs(arg) for arg in e.arguments])
        elif isinstance(e, FunctionApplicationExpression):
            arguments = [dfs(arg) for arg in e.arguments]
            return FunctionApplicationExpression(e.function, arguments)
        elif isinstance(e, BoolExpression):
            return BoolExpression(e.bool_op, [dfs(arg) for arg in e.arguments])
        elif isinstance(e, QuantificationExpression):
            bounded_variables.add(e.variable.name)
            rv = QuantificationExpression(e.quantification_op, e.variable, dfs(e.expression))
            bounded_variables.remove(e.variable.name)
            return rv
        else:
            raise TypeError(f'Unsupported expression type: {type(e)}')

    return dfs(expression)


def _get_variable_or_constant_name(expr: Union[VariableExpression, ObjectConstantExpression]) -> str:
    """Get the name of the given variable or constant expression."""
    if isinstance(expr, VariableExpression):
        return expr.name
    elif isinstance(expr, ObjectConstantExpression):
        return expr.name
    else:
        raise TypeError(f'Unsupported type: {type(expr)} for _get_variable_or_constant_name.')


def _get_variable_or_constant_object(expr: Union[VariableExpression, ObjectConstantExpression]) -> Union[Variable, ObjectConstant]:
    """Get the object of the given variable or constant expression."""
    if isinstance(expr, VariableExpression):
        return expr.variable
    elif isinstance(expr, ObjectConstantExpression):
        return expr.constant
    else:
        raise TypeError(f'Unsupported type: {type(expr)} for _get_variable_or_constant_object.')


def ground_operator_application_expression(expression: OperatorApplicationExpression, variable_mapping: Dict[Variable, str], csp: Optional[ConstraintSatisfactionProblem] = None, rule_applier: Optional[RegressionRuleApplier] = None) -> OperatorApplier:
    """Ground the given operator application expression with the given variable mapping.

    Args:
        expression: the expression to ground.
        variable_mapping: the variable mapping, which is a mapping from the Variable object to the constant name.
        csp: the constraint satisfaction problem to add the constraints to.
        rule_applier: the rule applier to use.

    Returns:
        the grounded expression.
    """
    name2symbol = {var.name: name for var, name in variable_mapping.items()}
    arguments = list()
    for arg in expression.arguments:
        if isinstance(arg, VariableExpression):
            symbol = name2symbol[arg.variable.name]
            arguments.append(symbol.name if isinstance(symbol, ObjectConstant) else symbol)
        elif isinstance(arg, UnnamedPlaceholder):
            if csp is not None:
                arguments.append(TensorValue.from_optimistic_value(csp.new_var(arg.dtype, wrap=True)))
            else:
                arguments.append(arg)
        elif isinstance(arg, str):
            arguments.append(arg)
        else:
            raise TypeError(f'Unknown argument type: {type(arg)}')
    return OperatorApplier(expression.operator, arguments, regression_rule=rule_applier)


def ground_regression_application_expression(expression: RegressionRuleApplicationExpression, variable_mapping: Dict[Variable, str], csp: Optional[ConstraintSatisfactionProblem] = None) -> RegressionRuleApplier:
    """Ground the given regression application expression with the given variable mapping.

    Args:
        expression: the expression to ground.
        variable_mapping: the variable mapping, which is a mapping from the Variable object to the constant name.
        csp: the constraint satisfaction problem to add the constraints to.

    Returns:
        the grounded expression.
    """
    name2symbol = {var.name: name for var, name in variable_mapping.items()}
    arguments = list()
    for arg in expression.arguments:
        if isinstance(arg, VariableExpression):
            symbol = name2symbol[arg.variable.name]
            arguments.append(symbol.name if isinstance(symbol, ObjectConstant) else symbol)
        elif isinstance(arg, UnnamedPlaceholder):
            if csp is not None:
                arguments.append(TensorValue.from_optimistic_value(csp.new_var(arg.dtype, wrap=True)))
            else:
                arguments.append(arg)
        elif isinstance(arg, str):
            arguments.append(arg)
        else:
            raise TypeError(f'Unknown argument type: {type(arg)}')
    return RegressionRuleApplier(
        expression.regression_rule, arguments, maintains=_ground_maintains_expressions(expression.maintains, variable_mapping),
        serializability=expression.serializability, csp_serializability=expression.csp_serializability
    )


def evaluate_bool_scalar_expression(
    executor: PDSketchExecutor, expr: Union[ValueOutputExpression, Sequence[ValueOutputExpression]], state: State,
    bounded_variables: BoundedVariablesDictCompatible, csp: ConstraintSatisfactionProblem, csp_note: str = ''
) -> Tuple[bool, bool, Optional[ConstraintSatisfactionProblem]]:
    if csp is not None:
        csp = csp.clone()
    is_optimistic = False
    if not isinstance(expr, (list, tuple)):
        expr = [expr]
    for e in expr:
        rv = executor.execute(e, state=state, bounded_variables=bounded_variables, csp=csp).item()
        if isinstance(rv, OptimisticValue):
            csp.add_constraint(EqualityConstraint.from_bool(rv, True), note=csp_note)
            is_optimistic = True
        elif float(rv) < 0.5:
            return False, is_optimistic, csp
    return True, is_optimistic, csp


class ApplicableRegressionRuleItem(NamedTuple):
    regression_rule: RegressionRule
    bounded_variables: BoundedVariablesDictCompatible


class ApplicableRegressionRuleGroup(NamedTuple):
    chain_index: int
    subgoal_index: int
    regression_rules: List[ApplicableRegressionRuleItem]


def gen_applicable_regression_rules(
    executor: PDSketchExecutor, state: State, goals: PartiallyOrderedPlan,
    maintains: Sequence[ValueOutputExpression],
    return_all_candidates: bool = True,
    verbose: bool = False
) -> List[ApplicableRegressionRuleGroup]:
    from concepts.dm.pdsketch.planners.optimistic_search_bilevel_utils import extract_bounded_variables_from_nonzero, extract_bounded_variables_from_nonzero_dc

    # TODO(Jiayuan Mao @ 2023/09/10): implement maintains.
    candidate_regression_rules = list()
    for chain_index, chain in goals.iter_feasible_chains():
        if chain.is_ordered:
            subgoal_indices = [len(chain) - 1]
        else:
            subgoal_indices = list(range(len(chain)))

        for subgoal_index in subgoal_indices:
            subgoal = chain.sequence[subgoal_index]
            this_chain_candidate_regression_rules = list()

            if isinstance(subgoal, RegressionRuleApplier):
                this_chain_candidate_regression_rules.append(ApplicableRegressionRuleItem(subgoal.regression_rule, {arg: argv for arg, argv in zip(subgoal.regression_rule.arguments, subgoal.arguments)}))
            else:
                # For each subgoal in the goal_set, try to find a list of applicable regression rules.
                # If one of the regression rules is always applicable, then we can stop searching.
                for regression_rule in executor.domain.regression_rules.values():
                    goal_expr = regression_rule.goal_expression
                    variable_binding = surface_fol_downcast(goal_expr, subgoal)
                    if verbose:
                        jacinle.log_function.print(f'Matching goal {subgoal} with template {goal_expr} -> {variable_binding}')
                    if variable_binding is None:
                        continue

                    bounded_variables = dict()
                    for v in regression_rule.goal_arguments:
                        value = variable_binding[v.name]
                        bounded_variables[v] = value

                    if len(regression_rule.binding_arguments) > 0:
                        for v in regression_rule.binding_arguments:
                            bounded_variables[v] = QINDEX

                    if len(regression_rule.preconditions_conjunction.arguments) > 0:
                        if len(regression_rule.binding_arguments) > 4 and len(regression_rule.preconditions_conjunction.arguments) >= 2:
                            rv = extract_bounded_variables_from_nonzero_dc(executor, state, regression_rule, bounded_variables, use_optimistic=False)
                        else:
                            sgc = PDSketchSGC(state, regression_rule.goal_expression, maintains)
                            rv = executor.execute(regression_rule.preconditions_conjunction, state=state, bounded_variables=bounded_variables, sgc=sgc)
                            if verbose:
                                jacinle.log_function.print(f'Precondition evaluation: {rv}, bounded_variables: {bounded_variables}')
                            rv = extract_bounded_variables_from_nonzero(state, rv, regression_rule, default_bounded_variables=bounded_variables, use_optimistic=False)
                    else:
                        rv = None

                    if rv is None:
                        type_binding_arguments = [state.object_type2name[v.dtype.typename] for v in regression_rule.binding_arguments]
                        all_forall = True
                        for i, arg in enumerate(regression_rule.binding_arguments):
                            if arg.quantifier_flag == 'forall':
                                type_binding_arguments[i] = type_binding_arguments[i][:1] if len(type_binding_arguments[i]) > 0 else []
                            else:
                                all_forall = False
                        for binding_arguments in itertools.product(*type_binding_arguments):
                            cbv = bounded_variables.copy()
                            for variable, value in zip(regression_rule.binding_arguments, binding_arguments):
                                cbv[variable] = value
                            if all_forall and regression_rule.always:
                                this_chain_candidate_regression_rules = [ApplicableRegressionRuleItem(regression_rule, cbv)]
                                break
                            this_chain_candidate_regression_rules.append(ApplicableRegressionRuleItem(regression_rule, cbv))
                    else:
                        all_forall, candidate_bounded_variables = rv
                        if all_forall and regression_rule.always:
                            candidate_bounded_variables = _expand_type_binding_arguments(state, regression_rule, candidate_bounded_variables, return_first=True)
                            this_chain_candidate_regression_rules = [ApplicableRegressionRuleItem(regression_rule, candidate_bounded_variables[0])]
                            break
                        else:
                            candidate_bounded_variables = _expand_type_binding_arguments(state, regression_rule, candidate_bounded_variables, return_first=False)
                            this_chain_candidate_regression_rules.extend([ApplicableRegressionRuleItem(regression_rule, cbv) for cbv in candidate_bounded_variables])
            candidate_regression_rules.append(ApplicableRegressionRuleGroup(chain_index, subgoal_index, this_chain_candidate_regression_rules))

    # TODO(Jiayuan Mao @ 2024/01/20): implement this within the search process.
    if not return_all_candidates:
        # if we don't need to return all candidates, we can return only the first applicable regression rule if it's always applicable (for each subgoal).
        filtered_candidate_regression_rules = list()
        for item in candidate_regression_rules:
            simplified_regression_rules = list()
            for regression_rule in item.regression_rules:
                if regression_rule.regression_rule.always:
                    simplified_regression_rules = [regression_rule]
                    break
                else:
                    simplified_regression_rules.append(regression_rule)
            filtered_candidate_regression_rules.append(ApplicableRegressionRuleGroup(item.chain_index, item.subgoal_index, simplified_regression_rules))
        candidate_regression_rules = filtered_candidate_regression_rules

    return candidate_regression_rules


def _expand_type_binding_arguments(state: State, regression_rule: RegressionRule, candidate_bounded_variables, return_first: bool = False):
    output_candidate_bounded_variables = list()
    for cbv in candidate_bounded_variables:
        groups_variables = list()
        groups_values = list()
        for k, v in cbv.items():
            if v is QINDEX:
                groups_variables.append(k)
                groups_values.append(state.object_type2name[k.dtype.typename])
        if len(groups_variables) == 0:
            output_candidate_bounded_variables.append(cbv)
            if return_first:
                return output_candidate_bounded_variables
        else:
            for binding_arguments in itertools.product(*groups_values):
                cbv = cbv.copy()
                for variable, value in zip(groups_variables, binding_arguments):
                    cbv[variable] = value
                output_candidate_bounded_variables.append(cbv)
                if return_first:
                    return output_candidate_bounded_variables
    return output_candidate_bounded_variables


def len_candidate_regression_rules(candidate_regression_rules: List[ApplicableRegressionRuleGroup]) -> int:
    """Compute the number of candidate regression rules."""
    return sum(len(x.regression_rules) for x in candidate_regression_rules)


def create_find_expression_csp_variable(variable: Variable, csp: ConstraintSatisfactionProblem, bounded_variables: Dict[Variable, Any]):
    """Create a TensorValue that corresponds to a variable inside a `FindExpression`.

    Args:
        variable: the variable in the FindExpression.
        csp: the current CSP.
        bounded_variables: the already bounded variables.
    """

    if variable.dtype.is_list_type:
        length = -1
        for v in bounded_variables.values():
            if isinstance(v, ListValue):
                length = len(v)
        if length == -1:
            raise ValueError(f'Cannot create a list variable {variable} without specifying the length.')

        return ListValue(variable.dtype, [TensorValue.from_optimistic_value(csp.new_actionable_var(variable.dtype.element_type, wrap=True)) for _ in range(length)])
    else:
        return TensorValue.from_optimistic_value(csp.new_actionable_var(variable.dtype, wrap=True))


def create_find_expression_variable_placeholder(variable: Variable, bounded_variables: Dict[Variable, Any]):
    """Create a TensorValue that corresponds to a variable inside a `FindExpression`. Unlike `_create_find_expression_variable`, this function only creates placeholder variables.

    Args:
        variable: the variable in the FindExpression.
        bounded_variables: the already bounded variables.
    """

    if variable.dtype.is_list_type:
        length = -1
        for v in bounded_variables.values():
            if isinstance(v, ListValue):
                length = len(v)
        if length == -1:
            raise ValueError(f'Cannot create a list variable {variable} without specifying the length.')

        return ListValue(variable.dtype, [UnnamedPlaceholder(variable.dtype) for _ in range(length)])
    else:
        return UnnamedPlaceholder(variable.dtype)


def mark_constraint_group_solver(executor: PDSketchExecutor, state: State, bounded_variables: Dict[Variable, Any], group: GroupConstraint):
    """Mark the solver for the current state.

    Args:
        executor: the executor.
        state: the current state.
        bounded_variables: the already bounded variables.
        group: the current group constraint.
    """

    for generator in executor.domain.generators.values():
        if (matching := surface_fol_downcast(generator.certifies, group.expression)) is not None:
            matching_success = True
            inputs = list()
            outputs = list()
            for var in generator.context:
                value = executor.execute(var, state, bounded_variables, optimistic_execution=True)
                if has_optimistic_value_or_list(value):
                    matching_success = False
                    break
                inputs.append(value)
            for var in generator.generates:
                this_matching_success = False
                if isinstance(var, VariableExpression):
                    if var.name in matching and is_single_optimistic_value_or_list(matching[var.name]):
                        this_matching_success = True
                        outputs.append(cvt_single_optimistic_value_or_list(matching[var.name]))
                if not this_matching_success:
                    matching_success = False
            if matching_success:
                group.candidate_generators.append((generator, inputs, outputs))


def has_optimistic_value_or_list(x: Union[ListValue, TensorValue]) -> bool:
    """Check if there is any optimistic value in the input TensorValue or a list of TensorValue's."""
    if isinstance(x, ListValue):
        return any(has_optimistic_value_or_list(y) for y in x.values)
    elif isinstance(x, TensorValue):
        return x.has_optimistic_value()
    else:
        raise ValueError(f'Unknown value type {type(x)}')


def is_single_optimistic_value_or_list(x: Union[ListValue, TensorValue]) -> bool:
    """Check if the input TensorValue is a single optimistic value or a list of TensorValue's that are all single optimistic values."""
    if isinstance(x, ListValue):
        return all(is_single_optimistic_value_or_list(y) for y in x.values)
    elif isinstance(x, TensorValue):
        return x.is_single_optimistic_value()
    else:
        raise ValueError(f'Unknown value type {type(x)}')


def cvt_single_optimistic_value_or_list(x: Union[ListValue, TensorValue]) -> Union[ListValue, OptimisticValue]:
    """Convert a single optimistic value stored in a TensorValue to an OptimisticValue. If the input is a list of TensorValue's, convert them to a list of OptimisticValue's."""
    if isinstance(x, ListValue):
        return ListValue(x.dtype, [cvt_single_optimistic_value_or_list(y) for y in x.values])
    elif isinstance(x, TensorValue):
        return x.single_elem()
    else:
        raise ValueError(f'Unknown value type {type(x)}')


def has_optimistic_constant_expression(*expressions: Union[ValueOutputExpression, RegressionRuleApplier]):
    """Check if there is a ConstantExpression whose value is an optimistic constant. Useful when checking if the subgoal is fully "grounded." """
    for expression in expressions:
        if isinstance(expression, RegressionRuleApplier):
            expression = expression.goal_expression
        for e in iter_exprs(expression):
            if isinstance(e, ConstantExpression) and has_optimistic_value_or_list(e.constant):
                return True
    return False


def make_rule_applier(rule: RegressionRule, bounded_variables: Dict[str, ValueOutputExpression]) -> RegressionRuleApplier:
    """Make a rule applier from a regression rule and a set of bounded variables."""
    canonized_bounded_variables = dict()
    for k, v in bounded_variables.items():
        if isinstance(k, Variable):
            k = k.name
        if isinstance(v, ObjectConstant):
            v = v.name
        canonized_bounded_variables[k] = v
    arguments = [canonized_bounded_variables[x.name] for x in rule.arguments]
    return RegressionRuleApplier(rule, arguments)


class _ReplaceCSPVariableVisitor(IdentityExpressionVisitor):
    def __init__(self, csp: ConstraintSatisfactionProblem, previous_csp: ConstraintSatisfactionProblem, csp_variable_mapping: Dict[int, Any], reg_variable_mapping: Optional[Dict[str, ObjectConstant]]):
        self.csp = csp
        self.previous_csp = previous_csp
        self.csp_variable_mapping = csp_variable_mapping
        self.reg_variable_mapping = reg_variable_mapping if reg_variable_mapping is not None else dict()

    def _replace_opt_value(self, value: Any):
        if isinstance(value, ListValue):
            return ListValue(value.dtype, [self._replace_opt_value(x) for x in value.values])
        elif isinstance(value, TensorValue):
            if value.is_single_optimistic_value():
                identifier = value.single_elem().identifier
                if identifier in self.csp_variable_mapping:
                    return self.csp_variable_mapping[identifier]
                else:
                    self.csp_variable_mapping[identifier] = TensorValue.from_optimistic_value(self.csp.new_actionable_var(value.dtype, wrap=True))
                    return self.csp_variable_mapping[identifier]
            else:
                return value
        else:
            raise ValueError(f'Unknown value type {type(value)}')

    def visit_constant_expression(self, expr: ConstantExpression) -> ConstantExpression:
        return ConstantExpression(self._replace_opt_value(expr.constant))

    def visit_variable_expression(self, expr: VariableExpression) -> Union[VariableExpression, ObjectConstantExpression, ConstantExpression]:
        if expr.variable.name in self.reg_variable_mapping:
            value = self.reg_variable_mapping[expr.variable.name]
            if isinstance(value, ObjectConstant):
                return ObjectConstantExpression(value)
            elif isinstance(value, TensorValue):
                return ConstantExpression(value)
            else:
                raise ValueError(f'Unknown value type {type(value)}')
        return expr


# subgoal, new_csp_variable_mapping = _map_csp_placeholder_goal(item.goal, new_csp, placeholder_csp, placeholder_bounded_variables, cur_bounded_variables, csp_variable_mapping)
def map_csp_placeholder_goal(
    subgoal: ValueOutputExpression, csp: ConstraintSatisfactionProblem,
    placeholder_csp: ConstraintSatisfactionProblem,
    cur_csp_variable_mapping: Dict[int, TensorValue],
    cur_reg_variable_mapping: Optional[Dict[str, Any]] = None
) -> Tuple[ValueOutputExpression, Dict[int, TensorValue]]:
    """Map the CSP variables in the subgoal to the CSP variables in the placeholder CSP."""

    new_mapping = cur_csp_variable_mapping.copy()
    visitor = _ReplaceCSPVariableVisitor(csp, placeholder_csp, new_mapping, cur_reg_variable_mapping)
    new_subgoal = visitor.visit(subgoal)
    return new_subgoal, new_mapping


def map_csp_placeholder_action(
    action: OperatorApplier, csp: ConstraintSatisfactionProblem,
    placeholder_csp: ConstraintSatisfactionProblem,
    cur_csp_variable_mapping: Dict[int, TensorValue],
    cur_reg_variable_mapping: Optional[Dict[str, Any]] = None,
) -> Tuple[OperatorApplier, Dict[int, TensorValue]]:
    """Map the CSP variables in the action to the CSP variables in the placeholder CSP."""

    new_mapping = cur_csp_variable_mapping.copy()
    new_arguments = list()
    for value in action.arguments:
        if isinstance(value, Variable):
            if cur_reg_variable_mapping is not None and value.name in cur_reg_variable_mapping:
                new_arguments.append(cur_reg_variable_mapping[value.name].name)
            else:
                raise KeyError(f'Unknown variable {value.name}')
        elif isinstance(value, TensorValue):
            if value.is_single_optimistic_value():
                identifier = value.single_elem().identifier
                if identifier in new_mapping:
                    new_arguments.append(new_mapping[identifier])
                else:
                    new_mapping[identifier] = TensorValue.from_optimistic_value(csp.new_actionable_var(value.dtype, wrap=True))
                    new_arguments.append(new_mapping[identifier])
            else:
                new_arguments.append(value)
        else:
            new_arguments.append(value)

    new_action = OperatorApplier(action.operator, new_arguments, regression_rule=action.regression_rule)
    return new_action, new_mapping


def map_csp_placeholder_regression_rule_applier(
    rule: RegressionRuleApplier, csp: ConstraintSatisfactionProblem,
    placeholder_csp: ConstraintSatisfactionProblem,
    cur_csp_variable_mapping: Dict[int, TensorValue],
    cur_reg_variable_mapping: Optional[Dict[str, Any]] = None
) -> Tuple[RegressionRuleApplier, Dict[int, TensorValue]]:
    """Map the CSP variables in the regression rule applier to the CSP variables in the placeholder CSP."""

    new_mapping = cur_csp_variable_mapping.copy()
    new_arguments = list()
    for value in rule.arguments:
        if isinstance(value, Variable):
            if cur_reg_variable_mapping is not None and value.name in cur_reg_variable_mapping:
                new_arguments.append(cur_reg_variable_mapping[value.name].name)
            else:
                raise KeyError(f'Unknown variable {value.name}')
        elif isinstance(value, TensorValue):
            if value.is_single_optimistic_value():
                identifier = value.single_elem().identifier
                if identifier in new_mapping:
                    new_arguments.append(new_mapping[identifier])
                else:
                    new_mapping[identifier] = TensorValue.from_optimistic_value(csp.new_actionable_var(value.dtype, wrap=True))
                    new_arguments.append(new_mapping[identifier])
            else:
                new_arguments.append(value)
        else:
            new_arguments.append(value)
    new_rule = RegressionRuleApplier(rule.regression_rule, new_arguments)
    return new_rule, new_mapping


def map_csp_variable_mapping(
    csp_variable_mapping: Dict[int, TensorValue], csp: ConstraintSatisfactionProblem, assignments: AssignmentDict
) -> Dict[int, TensorValue]:
    """Map the CSP variable mapping to the new variable mapping."""

    new_mapping = dict()
    for identifier, value in csp_variable_mapping.items():
        if isinstance(value, TensorValue):
            if value.is_single_optimistic_value():
                new_identifier = value.single_elem().identifier
                if new_identifier in assignments:
                    new_value = csp.ground_assignment_value_partial(assignments, new_identifier)
                    if isinstance(new_value, OptimisticValue):
                        new_mapping[identifier] = TensorValue.from_optimistic_value(new_value)
                    elif isinstance(new_value, TensorValue):
                        new_mapping[identifier] = new_value
                    else:
                        raise TypeError(f'Unknown value type {type(new_value)}')
            else:
                new_mapping[identifier] = value
        else:
            raise TypeError(f'Unknown value type {type(value)}')
    return new_mapping


def map_csp_variable_state(
    state: State, csp: ConstraintSatisfactionProblem, assignments: AssignmentDict
) -> State:
    """Map the CSP variable state to the new variable state."""

    new_state = state.clone()
    for feature_name, tensor_value in new_state.features.items():
        if tensor_value.tensor_optimistic_values is None:
            continue
        for ind in torch.nonzero(tensor_value.tensor_optimistic_values).tolist():
            ind = tuple(ind)
            identifier = tensor_value.tensor_optimistic_values[ind].item()
            if identifier in assignments:
                new_value = csp.ground_assignment_value_partial(assignments, identifier)
                if isinstance(new_value, OptimisticValue):
                    tensor_value.tensor_optimistic_values[ind] = new_value.identifier
                elif isinstance(new_value, TensorValue):
                    tensor_value.tensor[ind] = new_value.tensor
                    tensor_value.tensor_optimistic_values[ind] = 0
                else:
                    raise TypeError(f'Unknown value type {type(new_value)}')

    return new_state


GroundedSubgoalItem = Union[AchieveExpression, BindExpression, OperatorApplier, RegressionRuleApplier, RegressionCommitFlag]


def _ground_maintains_expressions(maintains: Tuple[ValueOutputExpression, ...], bounded_variables):
    return tuple(ground_fol_expression(e, bounded_variables) for e in maintains)


def gen_grounded_subgoals_with_placeholders(
    executor: PDSketchExecutor, state: State, goal: ValueOutputExpression, constraints: Sequence[ValueOutputExpression],
    candidate_regression_rules: List[ApplicableRegressionRuleItem],
    enable_csp: bool
) -> Dict[int, Tuple[List[GroundedSubgoalItem], Optional[ConstraintSatisfactionProblem], int]]:
    """Generated a set of subgoals with placeholders for CSP variables.

    Args:
        executor: the executor.
        state: the current state.
        goal: the goal expression.
        constraints: the constraints.
        candidate_regression_rules: the candidate regression rules.
        enable_csp: whether to enable the constraint satisfaction problem.

    Returns:
        the grounded subgoals. It is a dictionary mapping from the index of the regression rule to a tuple:
            - the grounded subgoals (which can be AchieveExpression, FindExpression, OperatorApplier, RegressionRuleApplier, or RegressionCommitFlag)
            - the constraint satisfaction problem (for tracking placeholder variables).
            - the length of the prefix that can be reordered.
    """
    grounded_subgoals_cache = dict()
    for regression_rule_index, (rule, bounded_variables) in enumerate(candidate_regression_rules):
        grounded_subgoals = list()
        placeholder_csp = ConstraintSatisfactionProblem() if enable_csp else None
        placeholder_bounded_variables = bounded_variables.copy()
        rule_applier = make_rule_applier(rule, placeholder_bounded_variables)
        for i, item in enumerate(rule.body):
            if isinstance(item, AchieveExpression):
                grounded_subgoals.append(AchieveExpression(
                    ground_fol_expression(item.goal, placeholder_bounded_variables), maintains=_ground_maintains_expressions(item.maintains, placeholder_bounded_variables),
                    serializability=item.serializability, csp_serializability=item.csp_serializability
                ))
            elif isinstance(item, BindExpression):
                if not enable_csp:
                    raise ValueError('FindExpression must be used with a constraint satisfaction problem.')
                for variable in item.variables:
                    if isinstance(variable.dtype, ObjectType):
                        placeholder_bounded_variables[variable] = variable
                    else:
                        placeholder_bounded_variables[variable] = create_find_expression_csp_variable(variable, csp=placeholder_csp, bounded_variables=placeholder_bounded_variables)
                grounded_subgoals.append(BindExpression(item.variables, ground_fol_expression(item.goal, placeholder_bounded_variables), serializability=item.serializability, csp_serializability=item.csp_serializability, ordered=item.ordered))
            elif isinstance(item, OperatorApplicationExpression):
                cur_action = ground_operator_application_expression(item, placeholder_bounded_variables, csp=placeholder_csp, rule_applier=rule_applier)
                grounded_subgoals.append(cur_action)
            elif isinstance(item, RegressionRuleApplicationExpression):
                cur_action = ground_regression_application_expression(item, placeholder_bounded_variables, csp=placeholder_csp)
                grounded_subgoals.append(cur_action)
            elif isinstance(item, ListExpansionExpression):
                if is_and_expr(item.expression) and len(item.expression.arguments) == 1 and item.expression.arguments[0].return_type.is_list_type:
                    # handles ... (and p({x, y, z}, ...))
                    subgoals = executor.execute(item.expression.arguments[0], state, placeholder_bounded_variables, sgc=PDSketchSGC(state, goal, constraints))
                    grounded_subgoals.extend(subgoals.values)
                else:
                    subgoals = executor.execute(item.expression, state, placeholder_bounded_variables, sgc=PDSketchSGC(state, goal, constraints))
                    assert isinstance(subgoals, TotallyOrderedPlan), f'ListExpansionExpression must be used with a TotallyOrderedPlan, got {type(subgoals)}'
                    grounded_subgoals.extend(subgoals.sequence)
            elif isinstance(item, RuntimeAssignExpression):
                placeholder_bounded_variables[item.variable] = item.variable
                grounded_subgoals.append(RuntimeAssignExpression(item.variable, ground_fol_expression(item.value, placeholder_bounded_variables)))
            elif isinstance(item, RegressionCommitFlag):
                grounded_subgoals.append(item)
            else:
                raise ValueError(f'Unknown item type {type(item)} in rule {item}.')

        # pass the serializability information to the previous subgoal.
        max_reorder_prefix_length = 0
        for i, item in enumerate(grounded_subgoals):
            if isinstance(item, RegressionCommitFlag):
                if i > 0 and isinstance(grounded_subgoals[i - 1], (AchieveExpression, BindExpression)):
                    grounded_subgoals[i - 1].serializability = item.goal_serializability
            if isinstance(item, (AchieveExpression, BindExpression)):
                if item.sequential_decomposable is False:
                    max_reorder_prefix_length = i + 1

        grounded_subgoals_cache[regression_rule_index] = (grounded_subgoals, placeholder_csp, max_reorder_prefix_length)
    return grounded_subgoals_cache
