#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : dpll_sampling.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 11/20/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""The DPLL-Sampling algorithm for solving CSPs. This algorithm is specifically designed
for solving CSPs with mixed Boolean and continuous variables. At a high level, the algorithm
uses DPLL-style search to find a solution to Boolean variables. After the value for all Boolean
variables are fixed, the algorithm uses a sampling-based method to find a solution to the continuous variables.
"""

from typing import Optional, Union, Tuple, List, Dict

import itertools
import collections
import jacinle

from concepts.dsl.dsl_types import BOOL, ValueType, NamedTensorValueType, TensorValueTypeBase, PyObjValueType
from concepts.dsl.dsl_functions import Function
from concepts.dsl.value import ListValue
from concepts.dsl.tensor_value import TensorValue
from concepts.dsl.constraint import Constraint, GroupConstraint, ConstraintSatisfactionProblem, OptimisticValue, OptimisticValueRecord, Assignment, AssignmentType, AssignmentDict, SimulationFluentConstraintFunction
from concepts.dsl.expression import VariableExpression, ValueOutputExpression, BoolOpType, QuantificationOpType, BoolExpression, PredicateEqualExpression, FunctionApplicationExpression
from concepts.dsl.expression import is_and_expr
from concepts.dsl.expression_utils import iter_exprs

from concepts.dm.pdsketch.executor import PDSketchExecutor, GeneratorManager
from concepts.dm.pdsketch.predicate import Predicate
from concepts.dm.pdsketch.generator import Generator, FancyGenerator

__all__ = [
    'CSPNotSolvable', 'CSPNoGenerator', 'ConstraintList',
    'dpll_apply_assignments',
    'dpll_filter_deterministic_equal', 'dpll_filter_deterministic_clauses', 'dpll_filter_unused_rhs',
    'dpll_find_bool_variable', 'dpll_find_grounded_function_application', 'dpll_find_typegen_variable', 'dpll_find_gen_variable_combined',
    'GeneratorMatchingInputType', 'GeneratorMatchingOutputType', 'GeneratorMatchingIOReturnType', 'GeneratorMatchingReturnType',
    'csp_dpll_sampling_solve', 'csp_dpll_simplify'
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


def dpll_apply_assignments(executor: PDSketchExecutor, constraints: ConstraintList, assignments: Dict[int, Assignment]) -> ConstraintList:
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
        nc = Constraint(c.function, new_args, new_rv, note=c.note, group=c.group)
        if _determined(nc.rv) and _determined(*nc.arguments) and not isinstance(nc.function, SimulationFluentConstraintFunction):
            if _check_constraint(executor, nc):
                continue
            else:
                raise CSPNotSolvable(f'Constraint {c} is not satisfied.')
        new_constraints.append(nc)
    return new_constraints


def _check_constraint(executor: PDSketchExecutor, c: Constraint) -> bool:
    """Helper function: check if a constraint has been satisfied based on the current assignments.

    Args:
        executor: the executor.
        c: the constraint.

    Returns:
        True if the constraint has been satisfied.
    """
    if c.function is BoolOpType.NOT:
        return c.arguments[0].item() == (not c.rv.item())
    elif c.function in (QuantificationOpType.FORALL, BoolOpType.AND):
        return all([x.item() for x in c.arguments]) == c.rv.item()
    elif c.function in (QuantificationOpType.EXISTS, BoolOpType.OR):
        return any([x.item() for x in c.arguments]) == c.rv.item()
    elif c.is_equal_constraint:
        if c.arguments[0].dtype == BOOL:
            return (c.arguments[0].item() == c.arguments[1].item()) == c.rv.item()
        else:
            return _check_eq(executor, c.arguments[0].dtype, c.arguments[0], c.arguments[1]) == c.rv.item()
    elif isinstance(c.function, SimulationFluentConstraintFunction):
        return False
    else:
        assert isinstance(c.function, Predicate)
        # NB(Jiayuan Mao @ 09/05): for generator placeholders, they can only be set true through the corresponding generators.
        if c.function.is_generator_placeholder:
            return False
        func = executor.get_function_implementation(c.function.name)
        rv = func(*c.arguments, return_type=c.function.return_type)
        if rv.dtype == BOOL:
            return (rv.item() > 0.5) == c.rv.item()
        else:
            return _check_eq(executor, c.function.return_type, rv, c.rv.item())


def _check_eq(executor: PDSketchExecutor, dtype: Union[TensorValueTypeBase, PyObjValueType], v1: TensorValue, v2: TensorValue) -> bool:
    """Helper function: check if two values are equal. Internally used by :meth:`_check_constraint`.

    Args:
        executor: the executor.
        dtype: the type of the values.
        v1: the first value.
        v2: the second value.

    Returns:
        True if the two values are equal.
    """
    if isinstance(dtype, TensorValueTypeBase) and dtype.is_intrinsically_quantized():
        return (v1.tensor == v2.tensor).item()
    assert isinstance(dtype, NamedTensorValueType)
    eq_function = executor.get_function_implementation('type::' + dtype.typename + '::equal')
    return bool(eq_function(v1, v2).item())


def dpll_filter_deterministic_equal(executor: PDSketchExecutor, constraints: ConstraintList, assignments: Dict[int, Assignment]) -> Tuple[bool, ConstraintList]:
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
                    else:
                        raise AssertionError('Sanity check failed.')
                    progress = True
                else:
                    if c.arguments[0].dtype == BOOL:
                        constraints[i] = Constraint(BoolOpType.NOT, [c.arguments[0]], c.arguments[1], note=c.note, group=c.group)
                        progress = True
            else:
                if isinstance(c.arguments[0], OptimisticValue) and isinstance(c.arguments[1], OptimisticValue) and c.arguments[0].identifier == c.arguments[1].identifier:
                    assignments[c.rv.identifier] = Assignment(AssignmentType.VALUE, True)
                    constraints[i] = None
                    progress = True

    if progress:
        return progress, dpll_apply_assignments(executor, constraints, assignments)
    return progress, constraints


def dpll_filter_unused_rhs(executor: PDSketchExecutor, constraints: ConstraintList, assignments: Dict[int, Assignment], index2record: Dict[int, OptimisticValueRecord]) -> ConstraintList:
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
    for k, v in used.items():
        if v == 1:
            assignments[k] = Assignment(AssignmentType.IGNORE, None)
    return dpll_apply_assignments(executor, constraints, assignments)


def dpll_filter_deterministic_clauses(executor: PDSketchExecutor, constraints: ConstraintList, assignments: Dict[int, Assignment]) -> Tuple[bool, ConstraintList]:
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
            elif _determined(*c.arguments):
                progress = True
                if c.function in (QuantificationOpType.FORALL, BoolOpType.AND):
                    assignments[c.rv.identifier] = Assignment(AssignmentType.VALUE, all(x.item() for x in c.arguments))
                elif c.function in (QuantificationOpType.EXISTS, BoolOpType.OR):
                    assignments[c.rv.identifier] = Assignment(AssignmentType.VALUE, any(x.item() for x in c.arguments))
                elif c.function is BoolOpType.NOT:
                    assignments[c.rv.identifier] = Assignment(AssignmentType.VALUE, not c.arguments[0].item())
        elif c.is_equal_constraint and _determined(*c.arguments):
            progress = True
            assignments[c.rv.identifier] = Assignment(AssignmentType.VALUE, c.arguments[0].item() == c.arguments[1].item())
    if progress:
        return progress, dpll_apply_assignments(executor, constraints, assignments)
    return progress, constraints


def dpll_filter_duplicated_constraints(executor: PDSketchExecutor, constraints: ConstraintList) -> Tuple[bool, ConstraintList]:
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
        return progress, dpll_apply_assignments(executor, constraints, {})
    return progress, constraints


def dpll_find_bool_variable(executor: PDSketchExecutor, constraints: ConstraintList, assignments: Dict[int, Assignment]) -> Optional[int]:
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


def dpll_find_grounded_function_application(executor: PDSketchExecutor, constraints: ConstraintList) -> Optional[Constraint]:
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
        if _determined(*c.arguments) and isinstance(c.function, Function):
            return c

    return None


def dpll_find_typegen_variable(executor: PDSketchExecutor, dtype: ValueType) -> Optional[Generator]:
    assert isinstance(dtype, NamedTensorValueType)
    for g in executor.domain.generators.values():
        if len(g.function.arguments) == 0 and g.function.return_type[0] == dtype:
            if isinstance(g.certifies, BoolExpression) and g.certifies.bool_op is BoolOpType.AND:
                return g
    return None


GeneratorMatchingInputType = List[Optional[TensorValue]]
GeneratorMatchingOutputType = List[Optional[OptimisticValue]]
GeneratorMatchingIOReturnType = Tuple[Optional[GeneratorMatchingInputType], Optional[GeneratorMatchingOutputType]]


def _match_generator(c: Constraint, g: Generator, certifies_expr: Optional[ValueOutputExpression] = None, allow_star_matching: bool = False) -> GeneratorMatchingIOReturnType:
    def gen_input_output(func_arguments, rv_variable=None):
        inputs: GeneratorMatchingInputType = [None for _ in range(len(g.input_vars))]
        outputs: GeneratorMatchingOutputType = [None for _ in range(len(g.output_vars))]
        for argc, argg in zip(c.arguments, func_arguments):
            if isinstance(argc, OptimisticValue):
                if argg.name.startswith('?g'):
                    outputs[int(argg.name[2:])] = argc
                else:
                    return None, None
            else:
                if argg.name.startswith('?c'):
                    inputs[int(argg.name[2:])] = argc
                elif allow_star_matching and argg.name == '??':
                    continue
                else:
                    return None, None
        if rv_variable is not None:
            if rv_variable.name.startswith('?c'):
                inputs[int(rv_variable.name[2:])] = c.rv
            else:
                return None, None
        return inputs, outputs

    if certifies_expr is None:
        certifies_expr = g.flatten_certifies

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
                elif c.is_equal_constraint and isinstance(inner_expr, PredicateEqualExpression):  # match (equal ?x ?y) == False
                    if c.arguments[0].dtype == inner_expr.predicate.return_type:
                        return gen_input_output([inner_expr.predicate, inner_expr.value])
    if isinstance(c.rv, TensorValue) and isinstance(c.function, Function):  # match (pred ?x ?y) == ?z
        if isinstance(certifies_expr, PredicateEqualExpression) and c.function.name == certifies_expr.predicate.function.name:
            if isinstance(certifies_expr.value, VariableExpression):
                return gen_input_output(certifies_expr.predicate.arguments, certifies_expr.value)
    return None, None


GeneratorMatchingReturnType = Optional[List[Tuple[Constraint, Generator, Optional[GeneratorMatchingInputType], Optional[GeneratorMatchingOutputType]]]]


def _find_gen_variable_group(executor: PDSketchExecutor, constraints: ConstraintList) -> GeneratorMatchingReturnType:
    all_generators = list()
    for c in constraints:
        if c.is_group_constraint:
            for g in c.candidate_generators:
                all_generators.append((c, *g))

    if len(all_generators) == 0:
        return None
    return all_generators


def _find_gen_variable(executor: PDSketchExecutor, constraints: ConstraintList) -> GeneratorMatchingReturnType:
    # Step 1: find all applicable generators.
    all_generators = list()
    for c in constraints:
        if c.is_group_constraint:
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


def _find_gen_variable_advanced(executor: PDSketchExecutor, constraints: ConstraintList) -> GeneratorMatchingReturnType:
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
                    if not _check_eq(executor, x.dtype, x, y):
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
            if is_and_expr(g.flatten_certifies):
                for sub_certifies_index, sub_certifies in enumerate(g.flatten_certifies.arguments):
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
        for sub_certifies in g.flatten_certifies.arguments:
            this_is_star_expression.append(is_star_expression(sub_certifies))

        for mcs, mg, mi, mo, matched_sci in all_matches:
            # the matched constraints should cover all sentences in the flatten_expression.
            match_succ = True
            for sub_certifies_index, sub_certifies in enumerate(g.flatten_certifies.arguments):
                if not this_is_star_expression[sub_certifies_index] and sub_certifies_index not in matched_sci:
                    match_succ = False
                    break
            if not match_succ:
                continue
            if None not in mi and None not in mo:
                all_generators.append((mcs, mg, mi, mo))

    return all_generators if len(all_generators) > 0 else None


def _find_fancy_gen_variable(
    executor: PDSketchExecutor,
    csp: ConstraintSatisfactionProblem,
    constraints: ConstraintList, assignments: AssignmentDict
) -> Optional[List[Tuple[List[Constraint], Dict[int, Union[TensorValueTypeBase, PyObjValueType]], FancyGenerator]]]:
    results = list()
    for g in sorted(executor.domain.fancy_generators.values(), key=lambda generator: generator.priority, reverse=True):
        g: FancyGenerator
        this_constraints = list()
        this_variable_dtypes = dict()

        assert is_and_expr(g.flatten_certifies)
        for certifies_expr in g.flatten_certifies.arguments:
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


def dpll_find_gen_variable_combined(executor: PDSketchExecutor, csp: ConstraintSatisfactionProblem, constraints: ConstraintList, assignments: AssignmentDict) -> GeneratorMatchingReturnType:
    """Combine the generator matching in the following order:

    1. Use :func:`_find_gen_variable` to find the generator with the highest priority.
    2. Use :func:`_find_gen_variable_advanced` to find the generator with the highest priority, using star-matching.
    3. Use :func:`_find_typegen_variable` to find the generator with the highest priority, using type-matching.
    """
    rv = _find_gen_variable_group(executor, constraints)
    if rv is not None:
        return rv
    rv = _find_gen_variable(executor, constraints)
    if rv is not None:
        return rv
    rv = _find_gen_variable_advanced(executor, constraints)
    if rv is not None:
        return rv
    for name, record in csp.index2record.items():
        dtype = record.dtype
        if name not in assignments and isinstance(dtype, NamedTensorValueType):
            g = dpll_find_typegen_variable(executor, dtype)
            if g is not None:
                rv = [(None, g, [], [OptimisticValue(dtype, name)])]
                return rv
    return None


def csp_dpll_sampling_solve(
    executor: PDSketchExecutor, csp: ConstraintSatisfactionProblem, *,
    generator_manager: Optional[GeneratorManager] = None,
    max_generator_trials: int = 3,
    enable_ignore: bool = False, solvable_only: bool = False,
    verbose: bool = False
) -> Optional[Union[bool, AssignmentDict]]:
    """Solve the constraint satisfaction problem using the DPLL-sampling algorithm.

    Args:
        executor: the executor.
        csp: the constraint satisfaction problem.
        generator_manager: the generator manager.
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
        generator_manager = GeneratorManager(executor, store_history=False)

    constraints = csp.constraints.copy()

    if verbose:
        jacinle.log_function.print('csp_dpll_sampling_solve: max_generator_trials =', max_generator_trials)
        jacinle.log_function.print('Constraints:', len(constraints))
        jacinle.log_function.print(*[jacinle.indent_text(str(c)) for c in constraints], sep='\n')

    @jacinle.log_function(verbose=False)
    def dfs(constraints, assignments):
        if len(constraints) == 0:
            return assignments

        progress = True
        while progress:
            progress, constraints = dpll_filter_deterministic_equal(executor, constraints, assignments)
        if enable_ignore:
            constraints = dpll_filter_unused_rhs(executor, constraints, assignments, csp.index2record)
        progress = True
        while progress:
            progress, constraints = dpll_filter_deterministic_clauses(executor, constraints, assignments)
        progress, constraints = dpll_filter_duplicated_constraints(executor, constraints)

        if verbose:
            jacinle.log_function.print('Remaining constraints:', len(constraints))
            jacinle.log_function.print(*constraints, sep='\n')

        if len(constraints) == 0:
            return assignments

        if (next_bool_var := dpll_find_bool_variable(executor, constraints, assignments)) is not None:
            assignments_true = assignments.copy()
            assignments_true[next_bool_var] = Assignment(AssignmentType.VALUE, True)
            try:
                constraints_true = dpll_apply_assignments(executor, constraints, assignments_true)
                return dfs(constraints_true, assignments_true)
            except CSPNotSolvable:
                pass

            assignments_false = assignments.copy()
            assignments_false[next_bool_var] = Assignment(AssignmentType.VALUE, False)
            try:
                constraints_false = dpll_apply_assignments(executor, constraints, assignments_false)
                return dfs(constraints_false, assignments_false)
            except CSPNotSolvable:
                pass

            raise CSPNotSolvable()
        elif (next_fapp := dpll_find_grounded_function_application(executor, constraints)) is not None:
            function: Predicate = next_fapp.function
            arguments = next_fapp.arguments

            external_function = executor.get_function_implementation(function.name)
            output = external_function(*arguments, auto_broadcast=False)

            target = next_fapp.rv
            new_assignments = assignments.copy()
            new_assignments[target.identifier] = Assignment(AssignmentType.VALUE, output)
            try:
                new_constraints = constraints.copy()
                new_constraints[new_constraints.index(next_fapp)] = None
                new_constraints = dpll_apply_assignments(executor, new_constraints, new_assignments)
                return dfs(new_constraints, new_assignments)
            except CSPNotSolvable:
                pass

            raise CSPNotSolvable()
        elif (next_gen_vars := _find_fancy_gen_variable(executor, csp, constraints, assignments)) is not None:
            if len(next_gen_vars) > 0 and next_gen_vars[0][1].unsolvable:
                raise CSPNotSolvable()

            for vv in next_gen_vars:
                c, dtype_mapping, g = vv

                generator = generator_manager.call(g, max_generator_trials, tuple(), c)
                generator = iter(generator)
                for j in range(max_generator_trials):
                    try:
                        output_index, outputs = next(generator)
                    except StopIteration:
                        break
                    if outputs is None:
                        break

                    assert isinstance(outputs, dict)
                    new_assignments = assignments.copy()
                    for k, v in outputs.items():
                        # v = TensorValue.from_scalar(v, dtype_mapping[k])
                        new_assignments[k] = Assignment(AssignmentType.VALUE, v)
                    try:
                        new_constraints = constraints.copy()
                        for cc in c:
                            new_constraints[new_constraints.index(cc)] = None

                        new_constraints = dpll_apply_assignments(executor, new_constraints, new_assignments)
                        return dfs(new_constraints, new_assignments)
                    except CSPNotSolvable:
                        pass

            raise CSPNotSolvable()
        elif (next_gen_vars := dpll_find_gen_variable_combined(executor, csp, constraints, assignments)) is not None:
            if len(next_gen_vars) > 1:
                # jacinle.log_function.print('generator orders', *[str(vv[1]).split('\n')[0] for vv in next_gen_vars], sep='\n  ')
                pass

            if len(next_gen_vars) > 0 and next_gen_vars[0][1].unsolvable:
                raise CSPNotSolvable()

            for vv in next_gen_vars:
                c, g, args, outputs_target = vv

                generator = generator_manager.call(g, max_generator_trials, tuple(args), c)
                generator = iter(generator)
                for j in range(max_generator_trials):
                    try:
                        output_index, outputs = next(generator)
                    except StopIteration:
                        break
                    if outputs is None:
                        break
                    if not isinstance(outputs, tuple) and g.function.ftype.is_singular_return:
                        outputs = (outputs, )

                    if not g.function.ftype.is_singular_return:
                        assert len(outputs) == len(g.function.return_type)

                    # jacinle.log_function.print('running generator', g, f'count = {j}')
                    new_assignments = assignments.copy()
                    for output, target in zip(outputs, outputs_target):
                        if isinstance(target, ListValue):
                            assert isinstance(output, ListValue)
                            assert len(output) == len(target)
                            for k, v in zip(target.values, output.values):
                                new_assignments[k.identifier] = Assignment(AssignmentType.VALUE, v)
                        else:
                            # output = TensorValue.from_scalar(output, target.dtype)
                            new_assignments[target.identifier] = Assignment(AssignmentType.VALUE, output)
                            # jacinle.log_function.print('assigned', target, output)
                    try:
                        new_constraints = constraints.copy()
                        if isinstance(c, list):
                            for cc in c:
                                new_constraints[new_constraints.index(cc)] = None
                        else:
                            new_constraints[new_constraints.index(c)] = None

                            if isinstance(c, GroupConstraint):
                                for i, cc in enumerate(new_constraints):
                                    if cc is not None and not cc.is_group_constraint and cc.group is not None and id(cc.group) == id(c):
                                        new_constraints[i] = None

                        # jacinle.log_function.print('new assignments', new_assignments)
                        # jacinle.log_function.print('new_constraints', new_constraints)
                        new_constraints = dpll_apply_assignments(executor, new_constraints, new_assignments)
                        return dfs(new_constraints, new_assignments)
                    except CSPNotSolvable:
                        pass

            raise CSPNotSolvable()
        else:
            # jacinle.log_function.print('Can not find a generator. Constraints:\n  ' + '\n  '.join([str(x) for x in constraints]))
            raise CSPNoGenerator('Can not find a generator. Constraints:\n  ' + '\n  '.join([str(x) for x in constraints]))

    try:
        assignments = dfs(constraints, {})
        if solvable_only:
            return True
        for name, record in csp.index2record.items():
            dtype = record.dtype
            if name not in assignments:
                g = dpll_find_typegen_variable(executor, dtype)
                if g is None:
                    raise NotImplementedError('Can not find a generator for unbounded variable {}, type {}.'.format(name, dtype))
                else:
                    output, = executor.get_function_implementation(g.function.name)()
                    assignments[name] = Assignment(AssignmentType.VALUE, TensorValue.from_scalar(output, dtype))
        return assignments
    except CSPNotSolvable:
        return None
    except CSPNoGenerator:
        raise


def csp_dpll_simplify(
    executor: PDSketchExecutor,
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

