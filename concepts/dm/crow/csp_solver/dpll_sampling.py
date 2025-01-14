#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : dpll_sampling.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/23/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""The DPLL-Sampling algorithm for solving CSPs. This algorithm is specifically designed
for solving CSPs with mixed Boolean and continuous variables. At a high level, the algorithm
uses DPLL-style search to find a solution to Boolean variables. After the value for all Boolean
variables are fixed, the algorithm uses a sampling-based method to find a solution to the continuous variables."""

from typing import Optional, Union, Sequence, Tuple, List, Dict

import itertools
import collections
import jacinle
from jacinle.utils.context import EmptyContext

from concepts.dsl.dsl_types import BOOL, ValueType, NamedTensorValueType, TensorValueTypeBase, PyObjValueType
from concepts.dsl.dsl_functions import Function
from concepts.dsl.tensor_value import TensorValue
from concepts.dsl.constraint import Constraint, GroupConstraint, ConstraintSatisfactionProblem, OptimisticValue, OptimisticValueRecord, Assignment, AssignmentType, AssignmentDict, SimulationFluentConstraintFunction
from concepts.dsl.expression import VariableExpression, ValueOutputExpression, BoolOpType, QuantificationOpType, BoolExpression, PredicateEqualExpression, ValueCompareExpression, CompareOpType, FunctionApplicationExpression
from concepts.dsl.expression_utils import iter_exprs

from concepts.dm.crow.crow_function import CrowFeature, CrowFunction, CrowFunctionBase
from concepts.dm.crow.crow_generator import CrowGeneratorBase, CrowDirectedGenerator, CrowUndirectedGenerator
from concepts.dm.crow.controller import CrowControllerApplier, CrowControllerApplier
from concepts.dm.crow.interfaces.controller_interface import CrowSimulationControllerInterface
from concepts.dm.crow.executors.crow_executor import CrowExecutor
from concepts.dm.crow.executors.generator_executor import CrowGeneratorExecutor
from concepts.dm.crow.csp_solver.csp_utils import csp_ground_action

__all__ = [
    'CSPNotSolvable', 'CSPNoGenerator', 'ConstraintList',
    'dpll_apply_assignments',
    'dpll_filter_deterministic_equal', 'dpll_filter_deterministic_clauses', 'dpll_filter_unused_rhs', 'dpll_filter_unused_simulation_rhs', 'dpll_filter_duplicated_constraints',
    'dpll_find_bool_variable', 'dpll_find_grounded_function_application', 'dpll_find_typegen_variable', 'dpll_find_gen_variable_combined',
    'GeneratorMatchingInputType', 'GeneratorMatchingOutputType', 'GeneratorMatchingIOReturnType', 'GeneratorMatchingReturnType',
    'dpll_solve', 'dpll_simplify'
]

ConstraintList = List[Optional[Union[Constraint, GroupConstraint]]]


class CSPNotSolvable(Exception):
    """An exception raised when the CSP is not solvable."""
    pass


class CSPNoGenerator(Exception):
    """An exception raised when there is no generator that can be matched in order to solve the CSP.
    Note that this does not mean that the CSP is not solvable."""
    pass


def _determined(*args) -> bool:
    """Helper function: if all arguments are determined.

    Args:
        *args: the arguments.

    Returns:
        True if all arguments are determined.
    """
    for x in args:
        if isinstance(x, OptimisticValue):
            return False
    return True


def _ground_assignment_value_partial(assignments: Dict[int, Assignment], dtype: ValueType, identifier: int) -> Union[TensorValue, OptimisticValue]:
    """Get the value of a variable based on the assignment dictionary. It will follow the EQUAL assignment types.

    The key difference between the :meth:`~concepts.dsl.constraint.ground_assignment_value` is the return type of this function.
    Specifically, the :meth:`concepts.dsl.constraint.ground_assignment_value` (exported) function returns the actual Value object.
    This function returns a wrapped Value object. When the value is not determined, it will return an OptimisticValue.

    Args:
        assignments: the assignment dictionary.
        dtype: the type of the variable.

    Returns:
        the value of the variable, wrapped in either :class:`~concepts.dsl.constraint.DeterminedValue` or :class:`~concepts.dsl.constraint.OptimisticValue`.
    """
    while identifier in assignments and assignments[identifier].t is AssignmentType.EQUAL:
        identifier = assignments[identifier].d
    if identifier in assignments:
        return assignments[identifier].d
    return OptimisticValue(dtype, identifier)


def dpll_apply_assignments(executor: CrowExecutor, constraints: ConstraintList, assignments: Dict[int, Assignment], simulation_state_index: Optional[int] = None) -> ConstraintList:
    """Apply the assignments to the constraints. Essentially, it replaces all variables that have been determined in the assignment dictionary with the value.

    This function will also check all the constraints to make sure that the assignments are valid.
    When a constraint is invalid, this function will raises CSPNotSolvable. Otherwise, constraints that have been satisfied will be removed from the list.

    Args:
        executor: the executor.
        constraints: the list of constraints.
        assignments: the dictionary of assignments.

    Returns:
        the list of constraints that have not been satisfied.
    """
    new_constraints = list()
    for c in constraints:
        if c is None:
            continue

        if isinstance(c, GroupConstraint):
            has_unsatisfied_subconstraint = False
            for c2 in constraints:
                if not c2.is_group_constraint and c2.group is not None and c2.group == c:
                    has_unsatisfied_subconstraint = True
                    break

            if has_unsatisfied_subconstraint:
                new_constraints.append(c)
            continue

        # If the return value of the constraint is ignored, simply ignore the entire constraint.
        if isinstance(c.rv, OptimisticValue) and c.rv.identifier in assignments and assignments[c.rv.identifier].t is AssignmentType.IGNORE:
            continue

        # Ground the arguments and the return value.
        new_args = list(c.arguments)
        for i, x in enumerate(c.arguments):
            if isinstance(x, OptimisticValue) and x.identifier in assignments:
                new_args[i] = _ground_assignment_value_partial(assignments, x.dtype, x.identifier)
        new_rv = c.rv
        if isinstance(c.rv, OptimisticValue) and c.rv.identifier in assignments:
            new_rv = _ground_assignment_value_partial(assignments, c.rv.dtype, c.rv.identifier)

        # Evaluate the constraint.
        nc = Constraint(c.function, new_args, new_rv, note=c.note, group=c.group, timestamp=c.timestamp)
        if _determined(nc.rv) and _determined(*nc.arguments) and not isinstance(nc.function, SimulationFluentConstraintFunction):
            if isinstance(nc.function, CrowFunction) and nc.function.is_generator_placeholder:
                raise CSPNotSolvable(f'Constraint {c} is not satisfied: the generator placeholder is not resolved.')
            if isinstance(nc.function, CrowFunction) and nc.function.is_simulation_dependent:
                if simulation_state_index is not None and nc.timestamp == simulation_state_index:
                    # print(f'>>> Matched! {nc}: timestamp={nc.timestamp}, simulation_state_index={simulation_state_index}')
                    run_check = True
                else:
                    run_check = False
            else:
                run_check = True
            if run_check:
                if executor.check_constraint(nc):
                    continue
                else:
                    raise CSPNotSolvable(f'Constraint {c} is not satisfied.')
            else:
                pass  # adding the nc back to the new_constraints
        new_constraints.append(nc)
    return new_constraints


def dpll_filter_deterministic_equal(executor: CrowExecutor, constraints: ConstraintList, assignments: Dict[int, Assignment], simulation_state_index: Optional[int] = None) -> Tuple[bool, ConstraintList]:
    """Filter the constraints to remove the ones that are determined to be equal.

    Args:
        executor: the executor.
        constraints: the list of constraints.
        assignments: the dictionary of assignments.

    Returns:
        a tuple of (whether we have made progress, the list of constraints that have not been satisfied).
    """

    progress = False
    for i, c in enumerate(constraints):
        if not c.is_group_constraint and c.is_equal_constraint:
            if isinstance(c.rv, TensorValue):
                # If the constraint looks like `x == x`, we can simply ignore it.
                if isinstance(c.arguments[0], OptimisticValue) and isinstance(c.arguments[1], OptimisticValue) and c.arguments[0].identifier == c.arguments[1].identifier:
                    if c.rv.item():
                        constraints[i] = None
                        progress = True
                        continue
                    else:
                        raise CSPNotSolvable(f'Constraint {c} can not be satisfied: {c.arguments[0]} is not equal to itself.')
                # If the constraint looks like: (x == y) == True, then we can set x = y.
                if c.rv.item():
                    if isinstance(c.arguments[0], OptimisticValue):
                        if isinstance(c.arguments[1], OptimisticValue):
                            assignments[c.arguments[0].identifier] = Assignment(AssignmentType.EQUAL, c.arguments[1].identifier)
                        else:
                            assignments[c.arguments[0].identifier] = Assignment(AssignmentType.VALUE, c.arguments[1])
                        constraints[i] = None
                    elif isinstance(c.arguments[1], OptimisticValue):
                        if isinstance(c.arguments[0], OptimisticValue):
                            assignments[c.arguments[1].identifier] = Assignment(AssignmentType.EQUAL, c.arguments[0].identifier)
                        else:
                            assignments[c.arguments[1].identifier] = Assignment(AssignmentType.VALUE, c.arguments[0])
                        constraints[i] = None
                    elif isinstance(c.arguments[0], TensorValue) and isinstance(c.arguments[1], TensorValue):
                        if executor.check_eq_constraint(c.arguments[0].dtype, c.arguments[0], c.arguments[1], True):
                            constraints[i] = None
                        else:
                            raise CSPNotSolvable(f'Constraint {c} can not be satisfied: {c.arguments[0]} is not equal to {c.arguments[1]}.')
                    else:
                        raise AssertionError('Sanity check failed.')
                    progress = True
                else:
                    if c.arguments[0].dtype == BOOL:
                        constraints[i] = Constraint(BoolOpType.NOT, [c.arguments[0]], c.arguments[1], note=c.note, group=c.group, timestamp=c.timestamp)
                        progress = True
            else:
                if isinstance(c.arguments[0], OptimisticValue) and isinstance(c.arguments[1], OptimisticValue) and c.arguments[0].identifier == c.arguments[1].identifier:
                    assignments[c.rv.identifier] = Assignment(AssignmentType.VALUE, True)
                    constraints[i] = None
                    progress = True

    if progress:
        return progress, dpll_apply_assignments(executor, constraints, assignments, simulation_state_index)
    return progress, constraints


def dpll_filter_unused_rhs(executor: CrowExecutor, constraints: ConstraintList, assignments: Dict[int, Assignment], index2record: Dict[int, OptimisticValueRecord], simulation_state_index: Optional[int] = None) -> Tuple[bool, ConstraintList]:
    """Filter out constraints that only appear once in the RHS of the constraints. In this case, the variable can be ignored and the related constraints can be removed.

    Args:
        executor: the executor.
        constraints: the list of constraints.
        assignments: the dictionary of assignments.
        index2record: the dictionary of variable records.

    Returns:
        the list of constraints that have not been satisfied, after removing all unused variables.
    """
    used: Dict[int, int] = collections.defaultdict(int)
    for i, record in index2record.items():
        if record.actionable:
            used[i] += 100

    for c in constraints:
        if c.is_group_constraint:
            continue
        for x in c.arguments:
            if isinstance(x, OptimisticValue):
                used[x.identifier] += 100  # as long as a variable has appeared in the lhs of a constraint, it is used.
        if isinstance(c.rv, OptimisticValue):
            used[c.rv.identifier] += 1  # if the variable has only appeared in the rhs of a constraint for once, it is not used.

    if len(used) == 0:
        return False, constraints

    for k, v in used.items():
        if v == 1:
            assignments[k] = Assignment(AssignmentType.IGNORE, None)
    return True, dpll_apply_assignments(executor, constraints, assignments, simulation_state_index)


def dpll_filter_deterministic_clauses(executor: CrowExecutor, constraints: ConstraintList, assignments: Dict[int, Assignment], simulation_state_index: Optional[int] = None) -> Tuple[bool, ConstraintList]:
    """Filter out Boolean constraints that have been determined. For example, AND(x, y, z) == true, then
        x == true, y == true, z == true. This function will remove the AND(x, y, z) constraint.

    There is another case that this function handles: for Boolean constraints, if everything on the LHS is determined, then
    we can determine the RHS.

    Args:
        executor: the executor.
        constraints: the list of constraints.
        assignments: the dictionary of assignments.

    Returns:
        a tuple of (whether we have made progress, the list of constraints that have not been satisfied).
    """
    progress = False
    for i, c in enumerate(constraints):
        if c.is_group_constraint:
            continue
        if isinstance(c.function, (QuantificationOpType, BoolOpType)):
            if _determined(c.rv):
                if (
                    (c.function in (QuantificationOpType.FORALL, BoolOpType.AND)) or
                    (c.function in (QuantificationOpType.EXISTS, BoolOpType.OR) and len(c.arguments) <= 1)
                ):
                    if c.rv.item():
                        for x in c.arguments:
                            if isinstance(x, OptimisticValue):
                                assignments[x.identifier] = Assignment(AssignmentType.VALUE, True)
                                progress = True
                                # print('assign', optimistic_value_id(x), True)
                            elif not x:
                                raise CSPNotSolvable()
                    else:
                        # AND(x, y, z) == false
                        determined_values = [x for x in c.arguments if _determined(x)]
                        if False in determined_values:
                            progress = True
                            constraints[i] = None
                        elif len(determined_values) == len(c.arguments):
                            raise CSPNotSolvable()
                        elif len(determined_values) == len(c.arguments) - 1:
                            for x in c.arguments:
                                if not _determined(x):
                                    progress = True
                                    assignments[x.identifier] = Assignment(AssignmentType.VALUE, False)
                elif c.function in (QuantificationOpType.EXISTS, BoolOpType.OR):
                    if not c.rv.item():
                        for x in c.arguments:
                            if isinstance(x, OptimisticValue):
                                progress = True
                                assignments[x.identifier] = Assignment(AssignmentType.VALUE, False)
                                # print('assign', optimistic_value_id(x), False)
                            elif x:
                                raise CSPNotSolvable()
                    else:
                        # OR(x, y, z) == TRUE
                        determined_values = [x.item() for x in c.arguments if _determined(x)]
                        if True in determined_values:
                            progress = True
                            constraints[i] = None
                        elif len(determined_values) == len(c.arguments):
                            raise CSPNotSolvable()
                        elif len(determined_values) == len(c.arguments) - 1:
                            for x in c.arguments:
                                if not _determined(x):
                                    progress = True
                                    assignments[x.identifier] = Assignment(AssignmentType.VALUE, True)
                elif c.function is BoolOpType.NOT:
                    progress = True
                    assignments[c.arguments[0].identifier] = Assignment(AssignmentType.VALUE, not c.rv.item())
                elif c.function is BoolOpType.IMPLIES:
                    progress = True
                    assignments[c.arguments[0].identifier] = Assignment(AssignmentType.VALUE, not c.rv.item() or c.arguments[1].item())
                elif c.function is BoolOpType.XOR:
                    determined_values = [x for x in c.arguments if _determined(x)]
                    if len(determined_values) == len(c.arguments):
                        raise CSPNotSolvable()
                    elif len(determined_values) == len(c.arguments) - 1:
                        for x in c.arguments:
                            if not _determined(x):
                                progress = True
                                if c.rv.item():
                                    assignments[x.identifier] = Assignment(AssignmentType.VALUE, sum(x.item() for x in c.arguments) % 2 == 0)
                                else:
                                    assignments[x.identifier] = Assignment(AssignmentType.VALUE, sum(x.item() for x in c.arguments) % 2 == 1)
            elif _determined(*c.arguments):
                progress = True
                constraints[i] = None
                if c.function in (QuantificationOpType.FORALL, BoolOpType.AND):
                    assignments[c.rv.identifier] = Assignment(AssignmentType.VALUE, all(x.item() for x in c.arguments))
                elif c.function in (QuantificationOpType.EXISTS, BoolOpType.OR):
                    assignments[c.rv.identifier] = Assignment(AssignmentType.VALUE, any(x.item() for x in c.arguments))
                elif c.function is BoolOpType.NOT:
                    assignments[c.rv.identifier] = Assignment(AssignmentType.VALUE, not c.arguments[0].item())
                elif c.function is BoolOpType.IMPLIES:
                    assignments[c.rv.identifier] = Assignment(AssignmentType.VALUE, not c.arguments[0].item() or c.arguments[1].item())
                elif c.function is BoolOpType.XOR:
                    assignments[c.rv.identifier] = Assignment(AssignmentType.VALUE, sum(x.item() for x in c.arguments) % 2 == 1)
        elif c.is_equal_constraint and _determined(*c.arguments):
            progress = True
            assignments[c.rv.identifier] = Assignment(AssignmentType.VALUE, c.arguments[0].item() == c.arguments[1].item())
    if progress:
        return progress, dpll_apply_assignments(executor, constraints, assignments, simulation_state_index)
    return progress, constraints


def dpll_filter_duplicated_constraints(executor: CrowExecutor, constraints: ConstraintList, simulation_state_index: Optional[int] = None) -> Tuple[bool, ConstraintList]:
    """Filter out duplicated constraints. For example, if we have x == 1 and x == 1, then we can remove one of them.

    Args:
        executor: the executor.
        constraints: the list of constraints.

    Returns:
        a tuple of (whether we have made progress, the list of constraints that have not been satisfied).
    """
    progress = False
    string_set = set()
    for i, c in enumerate(constraints):
        if c.is_group_constraint:
            continue
        # TODO(Jiayuan Mao @ 2023/11/24): since constraint_str contains shortened encodings for TensorValues, we should not use it here as a hash.
        cstr = c.constraint_str()
        if cstr in string_set:
            progress = True
            constraints[i] = None
        else:
            string_set.add(cstr)
    if progress:
        return progress, dpll_apply_assignments(executor, constraints, {}, simulation_state_index)
    return progress, constraints


def dpll_find_bool_variable(executor: CrowExecutor, constraints: ConstraintList, assignments: Dict[int, Assignment]) -> Optional[int]:
    """Find a Boolean variable that is not determined. As a heuristic, we will look for the variable that appear in the maximum number of constraints.

    Args:
        constraints: the list of constraints.
        assignments: the dictionary of assignments.

    Returns:
        the variable that is not determined.
    """
    count: Dict[int, int] = collections.defaultdict(int)
    for c in constraints:
        if c.is_group_constraint:
            continue
        for x in itertools.chain(c.arguments, [c.rv]):
            if isinstance(x, OptimisticValue) and x.identifier not in assignments and x.dtype == BOOL:
                count[x.identifier] += 1
    if len(count) == 0:
        return None
    return max(count, key=count.get)


def dpll_find_grounded_function_application(executor: CrowExecutor, simulation_interface: CrowSimulationControllerInterface, constraints: ConstraintList) -> Optional[Constraint]:
    """Find a function application whose arguments are all determined.

    Args:
        executor: the executor.
        constraints: the list of constraints.

    Returns:
        the function application that is not determined.
    """
    for c in constraints:
        if c.is_group_constraint:
            continue
        if _determined(*c.arguments):
            if isinstance(c.function, Function):
                if isinstance(c.function, CrowFunction) and c.function.is_simulation_dependent:
                    if c.timestamp == simulation_interface.get_action_counter():
                        return c
                else:
                    return c

    return None


def dpll_find_typegen_variable(executor: CrowExecutor, dtype: ValueType) -> Optional[CrowDirectedGenerator]:
    assert isinstance(dtype, TensorValueTypeBase)
    for g in executor.domain.generators.values():
        if isinstance(g, CrowDirectedGenerator):
            if len(g.inputs) == 0 and len(g.outputs) == 1 and g.outputs[0].dtype == dtype:
                if len(g.certifies) == 1 and g.certifies[0].is_null_expression:
                    return g
    return None


GeneratorMatchingInputType = List[Optional[TensorValue]]
GeneratorMatchingOutputType = List[Optional[OptimisticValue]]
GeneratorMatchingIOReturnType = Tuple[Optional[GeneratorMatchingInputType], Optional[GeneratorMatchingOutputType]]


def _match_generator(c: Constraint, g: CrowDirectedGenerator, certifies_expr: Optional[ValueOutputExpression] = None, allow_star_matching: bool = False) -> GeneratorMatchingIOReturnType:
    def gen_input_output(func_arguments, rv_variable=None):
        inputs: GeneratorMatchingInputType = [None for _ in range(len(g.inputs))]
        outputs: GeneratorMatchingOutputType = [None for _ in range(len(g.outputs))]
        for argc, argg in zip(c.arguments, func_arguments):
            if isinstance(argc, OptimisticValue):
                if argg.variable in g.outputs:
                    outputs[g.outputs.index(argg.variable)] = argc
                else:
                    return None, None
            else:
                if argg.variable in g.inputs:
                    inputs[g.inputs.index(argg.variable)] = argc
                elif allow_star_matching and argg.name == '??':
                    continue
                else:
                    return None, None
        if rv_variable is not None:
            if rv_variable.variable in g.inputs:  # If the RV is an output, then this would correspond to a function application, which is already handled.
                inputs[g.inputs.index(rv_variable.variable)] = c.rv
            else:
                return None, None
        return inputs, outputs

    if certifies_expr is None:
        certifies_expr = g.certifies[0]

    if isinstance(c.rv, TensorValue) and c.rv.dtype == BOOL:
        if c.rv.item():  # match (pred ?x ?y) == True
            if isinstance(certifies_expr, FunctionApplicationExpression):
                if c.function.name == certifies_expr.function.name:
                    return gen_input_output(certifies_expr.arguments)
        else:
            if isinstance(certifies_expr, BoolExpression) and certifies_expr.bool_op is BoolOpType.NOT:  # match (pred ?x ?y) == False
                inner_expr = certifies_expr.arguments[0]
                if isinstance(inner_expr, FunctionApplicationExpression):
                    if c.function.name == inner_expr.function.name:
                        return gen_input_output(inner_expr.arguments)
                elif c.is_equal_constraint:  # match (equal ?x ?y) == False
                    inner_expr = _parse_value_compare_expr_into_predicate_equal_expr(inner_expr)
                    if isinstance(inner_expr, PredicateEqualExpression):
                        if c.arguments[0].dtype == inner_expr.predicate.return_type:
                            return gen_input_output([inner_expr.predicate, inner_expr.value])
    if isinstance(c.rv, TensorValue) and isinstance(c.function, Function):  # match (pred ?x ?y) == ?z
        certifies_expr = _parse_value_compare_expr_into_predicate_equal_expr(certifies_expr)
        if isinstance(certifies_expr, PredicateEqualExpression) and c.function.name == certifies_expr.predicate.function.name:
            if isinstance(certifies_expr.value, VariableExpression):
                return gen_input_output(certifies_expr.predicate.arguments, certifies_expr.value)
    return None, None


def _parse_value_compare_expr_into_predicate_equal_expr(expr) -> Optional[PredicateEqualExpression]:
    if isinstance(expr, ValueCompareExpression):
        if expr.compare_op is CompareOpType.EQ:
            if isinstance(expr.arguments[0], FunctionApplicationExpression):
                return PredicateEqualExpression(expr.arguments[0], expr.arguments[1])
    return expr


GeneratorMatchingReturnType = List[Tuple[Union[Constraint, GroupConstraint], CrowDirectedGenerator, Optional[GeneratorMatchingInputType], Optional[GeneratorMatchingOutputType]]]


def _find_gen_variable_group(executor: CrowExecutor, constraints: ConstraintList) -> Optional[GeneratorMatchingReturnType]:
    all_generators = list()
    for c in constraints:
        if c.is_group_constraint:
            for g in c.candidate_generators:
                all_generators.append((c, *g))

    if len(all_generators) == 0:
        return None
    return all_generators


def _find_gen_variable(executor: CrowExecutor, constraints: ConstraintList, state_index: Optional[int] = None) -> Optional[GeneratorMatchingReturnType]:
    # Step 1: find all applicable generators.
    all_generators = list()
    for c in constraints:
        if c.is_group_constraint:
            continue
        if isinstance(c.function, CrowFunction) and c.function.is_simulation_dependent:
            if state_index is not None and c.timestamp == state_index:
                pass
            else:
                continue

        for g in executor.domain.generators.values():
            i, o = _match_generator(c, g)
            # jacinle.log_function.print('matching', c, g, i, o)
            if i is not None:
                all_generators.append((c, g, i, o))

    # Step 2: find if there is any variable with only one generator.
    target_to_generator: Dict[int, list] = collections.defaultdict(list)
    for c, g, i, o in all_generators:
        for target in o:
            target_to_generator[target.identifier].append((c, g, i, o))

    for target, generators in target_to_generator.items():
        if len(target_to_generator[target]) == 1:
            return target_to_generator[target]

    if len(all_generators) > 0:
        max_priority = max([r[1].priority for r in all_generators])
        all_generators = [r for r in all_generators if r[1].priority == max_priority]
        return all_generators

    return None


def _find_gen_variable_advanced(executor: CrowExecutor, constraints: ConstraintList) -> Optional[GeneratorMatchingReturnType]:
    def match_io(list1, list2):
        for x, y in zip(list1, list2):
            if x is None or y is None:
                continue
            if isinstance(x, OptimisticValue):
                if isinstance(y, OptimisticValue):
                    if x.identifier != y.identifier:
                        return False
                else:
                    return False
            elif isinstance(x, TensorValue):
                if isinstance(y, TensorValue):
                    if not executor.check_eq_constraint(x.dtype, x, y, True):
                        return False
                else:
                    return False
            else:
                raise TypeError(f'Invalid type: {type(x)}.')
        return True

    def is_star_expression(sub_certifies):
        for x in iter_exprs(sub_certifies):
            if isinstance(x, VariableExpression) and x.name == '??':
                return True
        return False

    generator2matched = collections.defaultdict(list)
    for c in constraints:
        if c.is_group_constraint:
            continue
        for g in executor.domain.generators.values():
            for sub_certifies_index, sub_certifies in enumerate(g.certifies):
                i, o = _match_generator(c, g, sub_certifies, allow_star_matching=True)
                if i is not None:
                    generator2matched[g].append((c, i, o, sub_certifies_index))

    all_generators = list()
    # TODO: implement the exact matching algorithm.
    for g, matched in generator2matched.items():
        all_matches = list()
        for result_index in range(len(matched)):
            c, i, o, sub_certifies_index = matched[result_index]

            used = False
            # constraints, g, i, o, matched_sub_certifies_index
            for mcs, mg, mi, mo, matched_sci in all_matches:
                if match_io(i, mi) and match_io(o, mo):
                    for j in range(len(mi)):
                        if mi[j] is None:
                            mi[j] = i[j]
                    for j in range(len(mo)):
                        if mo[j] is None:
                            mo[j] = o[j]

                    mcs.append(c)
                    matched_sci.add(sub_certifies_index)
                    used = True
                    break
            if not used:
                all_matches.append(([c], g, i, o, {sub_certifies_index}))

        this_is_star_expression = list()
        for sub_certifies in g.certifies:
            this_is_star_expression.append(is_star_expression(sub_certifies))

        for mcs, mg, mi, mo, matched_sci in all_matches:
            # the matched constraints should cover all sentences in the flatten_expression.
            match_succ = True
            for sub_certifies_index, sub_certifies in enumerate(g.certifies):
                if not this_is_star_expression[sub_certifies_index] and sub_certifies_index not in matched_sci:
                    match_succ = False
                    break
            if not match_succ:
                continue
            if None not in mi and None not in mo:
                all_generators.append((mcs, mg, mi, mo))

    return all_generators if len(all_generators) > 0 else None


def _find_fancy_gen_variable(
    executor: CrowExecutor,
    csp: ConstraintSatisfactionProblem,
    constraints: ConstraintList, assignments: AssignmentDict
) -> Optional[List[Tuple[List[Constraint], Dict[int, Union[TensorValueTypeBase, PyObjValueType]], CrowUndirectedGenerator]]]:
    results = list()
    for g in sorted(executor.domain.generators.values(), key=lambda generator: generator.priority, reverse=True):
        if isinstance(g, CrowUndirectedGenerator):
            this_constraints = list()
            this_variable_dtypes = dict()

            for certifies_expr in g.certifies:
                assert isinstance(certifies_expr, FunctionApplicationExpression)
                for arg in certifies_expr.arguments:
                    assert isinstance(arg, VariableExpression) and arg.name == '??'
                for c in constraints:
                    if c.function.name == certifies_expr.function.name:
                        this_constraints.append(c)
                    for arg in itertools.chain(c.arguments, [c.rv]):
                        if isinstance(arg, OptimisticValue):
                            this_variable_dtypes[arg.identifier] = arg.dtype
            if len(this_constraints) == 0:
                continue
            results.append((this_constraints, this_variable_dtypes, g))
    if len(results) > 0:
        return results
    return None


def dpll_find_gen_variable_combined(executor: CrowExecutor, simulation_interface: CrowSimulationControllerInterface, csp: ConstraintSatisfactionProblem, constraints: ConstraintList, assignments: AssignmentDict) -> GeneratorMatchingReturnType:
    """Combine the generator matching in the following order:

    1. Use :func:`_find_gen_variable` to find the generator with the highest priority.
    2. Use :func:`_find_gen_variable_advanced` to find the generator with the highest priority, using star-matching.
    3. Use :func:`_find_typegen_variable` to find the generator with the highest priority, using type-matching.
    """
    state_index = simulation_interface.get_action_counter() if simulation_interface is not None else None

    rv = _find_gen_variable_group(executor, constraints)
    if rv is not None:
        return rv
    rv = _find_gen_variable(executor, constraints, state_index=state_index)
    if rv is not None:
        return rv

    # rv = _find_gen_variable_advanced(executor, constraints)
    # if rv is not None:
    #     return rv

    # Implement _find_gen_variable_advanced
    # Implement _find_fancy_gen_variable

    rv = list()
    for name, record in csp.index2record.items():
        dtype = record.dtype
        if name not in assignments and isinstance(dtype, TensorValueTypeBase):
            g = dpll_find_typegen_variable(executor, dtype)
            if g is not None:
                rv.append((None, g, [], [OptimisticValue(dtype, name)]))
    if len(rv) > 0:
        return rv
    return None


def dpll_filter_unused_simulation_rhs(executor: CrowExecutor, constraints: ConstraintList, assignments: Dict[int, Assignment], simulation_state_index: Optional[int] = None) -> Tuple[bool, ConstraintList]:
    """Filter out simulation constraints that only appear once in the RHS of the constraints. In this case, the variable can be ignored and the related constraints can be removed.

    Args:
        executor: the executor.
        constraints: the list of constraints.
        assignments: the dictionary of assignments.

    Returns:
        the list of constraints that have not been satisfied, after removing all unused variables.
    """
    used = collections.defaultdict(int)
    for c in constraints:
        if c.is_group_constraint:
            continue
        for x in c.arguments:
            if isinstance(x, OptimisticValue):
                used[x.identifier] += 100  # as long as a variable has appeared in the lhs of a constraint, it is used.
        if isinstance(c.rv, OptimisticValue) and isinstance(c.function, SimulationFluentConstraintFunction):
            used[c.rv.identifier] += 1  # if the variable has only appeared in the rhs of a constraint for once, it is not used.

    if len(used) == 0:
        return False, constraints

    for k, v in used.items():
        if v == 1:
            assignments[k] = Assignment(AssignmentType.IGNORE, None)

    return True, dpll_apply_assignments(executor, constraints, assignments, simulation_state_index)


def dpll_apply_assignments_with_simulation(
    executor: CrowExecutor,
    constraints: ConstraintList, assignments: Dict[int, Assignment],
    simulation_interface: Optional[CrowSimulationControllerInterface], actions: Sequence[CrowControllerApplier],
    verbose: bool = False
) -> Tuple[ConstraintList, AssignmentDict]:
    simulation_state_index = simulation_interface.get_action_counter() if simulation_interface is not None else None
    new_constraints = dpll_apply_assignments(executor, constraints, assignments, simulation_state_index)
    new_assignments = None

    if simulation_interface is None:
        return new_constraints, assignments

    while True:
        next_action_index = simulation_interface.get_action_counter()
        if next_action_index < len(actions):
            action = actions[next_action_index]

            try:
                grounded_action = csp_ground_action(executor, action, assignments)
            except AssertionError:
                # The action is not applicable.
                break

            if verbose:
                jacinle.log_function.print(f'Executing grounded action: #{next_action_index}')

            succ = simulation_interface.step_without_error(grounded_action)
            if not succ:
                if verbose:
                    jacinle.log_function.print(jacinle.colored(f'Action {grounded_action} failed.', 'red'))
                raise CSPNotSolvable(f'Unable to perform action {action}.')
            # jacinle.log_function.print(f'Action {grounded_action} succeeded.')

            state = simulation_interface.get_crow_state()

            if new_assignments is None:
                new_assignments = assignments.copy()

            resolve_simulation_fluent_constraints_inplace(simulation_interface, new_constraints, new_assignments)
            new_constraints = dpll_apply_assignments(executor, new_constraints, new_assignments, simulation_state_index=next_action_index + 1)
        else:
            break

    return new_constraints, new_assignments if new_assignments is not None else assignments


def resolve_simulation_fluent_constraints_inplace(
    simulation_interface: CrowSimulationControllerInterface, constraints_: ConstraintList, assignments_: Dict[int, Assignment],
) -> Tuple[ConstraintList, AssignmentDict]:
    state = simulation_interface.get_crow_state()
    state_index = simulation_interface.get_action_counter()

    # Update the assignments.
    for i, c in enumerate(constraints_):
        if isinstance(c.function, SimulationFluentConstraintFunction):
            function: SimulationFluentConstraintFunction = c.function
            if function.state_index == state_index:  # After the "next_action_index" has been executed.
                if isinstance(c.rv, TensorValue):
                    if c.rv.dtype != BOOL:
                        raise NotImplementedError('Only bool is supported for simulation constraints.')

                    # print('Simulation constraint', c)
                    # print(state.features[function.predicate.name])
                    # print('Desired value =', c.rv.value, 'Actual value =', state.features[function.predicate.name][function.arguments].item())
                    # import pybullet
                    # pybullet.stepSimulation()
                    # import ipdb; ipdb.set_trace()

                    if state.features[function.predicate.name][function.arguments].item() != c.rv.item():
                        raise CSPNotSolvable()
                    else:
                        constraints_[i] = None
                else:
                    assignments_[c.rv.identifier] = Assignment(
                        AssignmentType.VALUE,
                        state.features[function.predicate.name][function.arguments]
                    )
                    constraints_[i] = None

    return constraints_, assignments_


def dpll_solve(
    executor: CrowExecutor, csp: ConstraintSatisfactionProblem, *,
    generator_manager: Optional[CrowGeneratorExecutor] = None,
    simulation_interface: Optional[CrowSimulationControllerInterface] = None,
    actions: Optional[Sequence[CrowControllerApplier]] = None,
    simulation_state: Optional[int] = None, simulation_state_index: Optional[int] = None,
    max_generator_trials: int = 3,
    enable_ignore: bool = False, solvable_only: bool = False,
    verbose: bool = False
) -> Optional[Union[bool, AssignmentDict]]:
    """Solve the constraint satisfaction problem using the DPLL-sampling algorithm.

    Args:
        executor: the executor.
        csp: the constraint satisfaction problem.
        generator_manager: the generator manager.
        simulation_interface: the simulation interface.
        actions: the list of actions. Only used when `simulation_interface` is not None.
        max_generator_trials: the maximum number of trials for each generator.
        enable_ignore: whether to ignore constraints whose RHS value is not determined.
        solvable_only: whether to only return whether the problem is solvable, without returning the solution.
        verbose: whether to print verbose information.

    Returns:
        When `solvable_only` is True, return a single Boolean value indicating whether the problem is solvable.
        When `solvable_only` is False, return an assignment dictionary.
        When the problem is not solvable, return None.

    Raises:
        CSPNotSolvable: when the problem is not solvable.
        CSPNoGenerator: when no generator can be found to solve the problem. However, the problem may still be solvable.
    """
    if generator_manager is None:
        generator_manager = CrowGeneratorExecutor(executor, store_history=False)

    constraints = csp.constraints.copy()
    if simulation_interface is not None:
        simulation_interface.reset_action_counter()

    def restore_context(verbose=verbose):
        if simulation_interface is not None:
            return simulation_interface.restore_context(verbose)
        return jacinle.EmptyContext()

    @jacinle.log_function(verbose=False)
    def dfs(constraints, assignments):
        if len(constraints) == 0:
            return assignments

        simulation_state_index = simulation_interface.get_action_counter() if simulation_interface is not None else None

        progress = True
        while progress:
            progress, constraints = dpll_filter_deterministic_equal(executor, constraints, assignments, simulation_state_index)
        if enable_ignore:
            _, constraints = dpll_filter_unused_rhs(executor, constraints, assignments, csp.index2record, simulation_state_index)
        else:
            # NB(Jiayuan Mao @ 2023/03/11): for simulation constraints, we can remove them if they are not used in any other constraints.
            # TODO(Jiayuan Mao @ 2023/03/15): actually, I just noticed that this is a "bug" in the implementation of `_find_bool_variable`.
            # Basically, if the variable is a simulation variable and it only appears in the RHS of a constraint, then it will be ignored.
            # The current handling will work, but probably I need to think about a better way to handle this.

            # TODO(Jiayuan Mao @ 2025/01/03): I forgot what was the bug in _find_bool_variable. But apparently this is causing
            # another bug in csp_ground_state. So here I comment out the following line.
            # _, constraints = dpll_filter_unused_simulation_rhs(executor, constraints, assignments, simulation_state_index)
            pass

        progress = True
        while progress:
            progress, constraints = dpll_filter_deterministic_clauses(executor, constraints, assignments, simulation_state_index)

        if verbose:
            jacinle.log_function.print('Remaining constraints:', len(constraints))
            jacinle.log_function.print(*constraints, sep='\n')

        if len(constraints) == 0:
            return assignments

        if (next_bool_var := dpll_find_bool_variable(executor, constraints, assignments)) is not None:
            assignments_true = assignments.copy()
            assignments_true[next_bool_var] = Assignment(AssignmentType.VALUE, True)

            with restore_context():
                try:
                    constraints_true, assignments_true = dpll_apply_assignments_with_simulation(executor, constraints, assignments_true, simulation_interface, actions, verbose=verbose)
                    return dfs(constraints_true, assignments_true)
                except CSPNotSolvable:
                    pass

            assignments_false = assignments.copy()
            assignments_false[next_bool_var] = Assignment(AssignmentType.VALUE, False)
            with restore_context():
                try:
                    constraints_false, assignments_false = dpll_apply_assignments_with_simulation(executor, constraints, assignments_false, simulation_interface, actions, verbose=verbose)
                    return dfs(constraints_false, assignments_false)
                except CSPNotSolvable:
                    pass

            raise CSPNotSolvable()
        elif (next_fapp := dpll_find_grounded_function_application(executor, simulation_interface, constraints)) is not None:
            function = next_fapp.function
            arguments = next_fapp.arguments
            target = next_fapp.rv

            external_function = executor.get_function_implementation(function.name)
            output = external_function(*arguments, return_type=target.dtype if isinstance(target, (TensorValue, OptimisticValue)) else None)

            new_assignments = assignments.copy()
            new_assignments[target.identifier] = Assignment(AssignmentType.VALUE, output)
            with restore_context():
                try:
                    new_constraints = constraints.copy()
                    new_constraints[new_constraints.index(next_fapp)] = None
                    new_constraints, new_assignments = dpll_apply_assignments_with_simulation(executor, new_constraints, new_assignments, simulation_interface, actions, verbose=verbose)
                    return dfs(new_constraints, new_assignments)
                except CSPNotSolvable:
                    pass

            raise CSPNotSolvable()
        elif (next_gen_vars := dpll_find_gen_variable_combined(executor, simulation_interface, csp, constraints, assignments)) is not None:
            if len(next_gen_vars) >= 1:
                if verbose:
                    jacinle.log_function.print('Generator orders', *[str(vv[1]).split('\n')[0] for vv in next_gen_vars], sep='\n  ')

            for vv in next_gen_vars:
                c, g, args, outputs_target = vv
                # TODO(Jiayuan Mao @ 2024/04/22): implement the "not solvable" handling.
                # if g.unsolvable:
                #     raise CSPNotSolvable('Hit unsolvable generator.')

                if verbose:
                    jacinle.log_function.print(f'Generator: {g}\nArgs: {args}')
                generator = generator_manager.call(g, max_generator_trials, args, c)

                # NB(Jiayuan Mao @ 2023/03/03): I didn't write for x in generator in order to make the verbose output more readable.
                generator = iter(generator)
                for j in range(max_generator_trials):
                    if verbose:
                        jacinle.log_function.print('Running generator', g.name, f'count = {j + 1} / {max_generator_trials}')

                    try:
                        output_index, outputs = next(generator)
                    except StopIteration:
                        if verbose:
                            jacinle.log_function.print('Generator', g.name, 'exhausted.')
                        break

                    new_assignments = assignments.copy()
                    for output, target in zip(outputs, outputs_target):
                        new_assignments[target.identifier] = Assignment(AssignmentType.VALUE, output, generator_index=output_index)
                        # jacinle.log_function.print('Assigned', target, output)

                    with restore_context():
                        try:
                            new_constraints = constraints.copy()
                            if c is None:  # If g is a type-only generator function, c === None
                                pass
                            else:
                                if isinstance(c, list):
                                    for cc in c:
                                        new_constraints[new_constraints.index(cc)] = None
                                else:
                                    new_constraints[new_constraints.index(c)] = None

                            new_constraints, new_assignments = dpll_apply_assignments_with_simulation(executor, new_constraints, new_assignments, simulation_interface, actions, verbose=verbose)
                            return dfs(new_constraints, new_assignments)
                        except CSPNotSolvable as e:
                            if verbose:
                                jacinle.log_function.print('Failed to apply assignments. Reason:', e)
                            pass

            raise CSPNotSolvable()
        else:
            jacinle.log_function.print('Can not find a generator. Constraints:\n  ' + '\n  '.join([str(x) for x in constraints]))
            from IPython import embed; embed()
            raise CSPNoGenerator('Can not find a generator. Constraints:\n  ' + '\n  '.join([str(x) for x in constraints]))

    with restore_context():
        init_assignments = dict()
        if simulation_state is not None:
            simulation_interface.restore_state_keep(simulation_state, simulation_state_index)
            resolve_simulation_fluent_constraints_inplace(simulation_interface, constraints, init_assignments)
            constraints = dpll_apply_assignments(executor, constraints, init_assignments, simulation_state_index)
            actions = tuple(None for _ in range(simulation_state_index)) + tuple(actions)

        try:
            assignments = dfs(constraints, init_assignments)
            if not solvable_only:
                for name, record in csp.index2record.items():
                    dtype = record.dtype
                    if name not in assignments:
                        g = dpll_find_typegen_variable(executor, dtype)
                        if g is None:
                            raise NotImplementedError('Can not find a generator for unbounded variable {}, type {}.'.format(name, dtype))
                        else:
                            try:
                                output_index, (output, ) = next(iter(generator_manager.call(g, 1, [], None)))
                                assignments[name] = Assignment(AssignmentType.VALUE, output, generator_index=output_index)
                            except StopIteration:
                                raise CSPNotSolvable

            if generator_manager.store_history:
                generator_manager.mark_success(assignments)
            if solvable_only:
                return True
            return assignments
        except CSPNotSolvable:
            return None
        except CSPNoGenerator:
            raise


def dpll_simplify(
    executor: CrowExecutor,
    csp: ConstraintSatisfactionProblem,
    enable_ignore: bool = True, return_assignments: bool = False
) -> Union[ConstraintSatisfactionProblem, Tuple[ConstraintSatisfactionProblem, AssignmentDict]]:
    """Simplify the CSP using DPLL algorithm.

    Args:
        executor: the executor.
        csp: the CSP.
        enable_ignore: whether to ignore constraints whose RHS value is not determined.
        return_assignments: whether to return the assignments.

    Returns:
        the simplified CSP.
    """

    constraints = csp.constraints.copy()
    assignments = dict()

    if len(constraints) == 0:
        return csp

    while True:
        nr_constraints = len(constraints)

        progress = True
        while progress:
            progress, constraints = dpll_filter_deterministic_equal(executor, constraints, assignments)
        if enable_ignore:
            constraints = dpll_filter_unused_rhs(executor, constraints, assignments, csp.index2record)
        progress = True
        while progress:
            progress, constraints = dpll_filter_deterministic_clauses(executor, constraints, assignments)

        if len(constraints) == nr_constraints:
            break

    if return_assignments:
        return csp.clone(constraints), assignments
    return csp.clone(constraints)

