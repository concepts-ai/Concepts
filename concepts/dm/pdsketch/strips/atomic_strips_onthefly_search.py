#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : atomic_strips_onthefly_search.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/14/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import time
from typing import Optional, Union, Iterator, Tuple, List, Dict
from collections import defaultdict, deque

from concepts.dsl.dsl_types import Variable
from concepts.dm.pdsketch.strips.strips_expression import SStateDict, SBoolPredicateApplicationExpression
from concepts.dm.pdsketch.strips.atomic_strips_domain import AtomicStripsDomain, AtomicStripsProblem, AtomicStripsOperator, AtomicStripsOperatorApplier


def _bind_arguments(predicate: SBoolPredicateApplicationExpression, bound_arguments: Dict[str, Union[int, str]]):
    return predicate.name, tuple(bound_arguments[arg.name] if isinstance(arg, Variable) else arg for arg in predicate.arguments)


def _gen_applicable_actions(domain: AtomicStripsDomain, objects: Dict[str, List[str]], state: SStateDict, check_negation: bool = False) -> Iterator[Tuple[AtomicStripsOperator, Dict[str, str]]]:
    TOO_MANY, FAILED, PASS = object(), object(), object()

    def compute_possible_grounding(predicate: SBoolPredicateApplicationExpression, bound_arguments: Dict[str, Union[int, str]]):
        unbound_arguments = [arg for arg in predicate.arguments if isinstance(arg, Variable) and arg.name not in bound_arguments]
        if len(unbound_arguments) == 0:
            name, arguments = _bind_arguments(predicate, bound_arguments)
            rv = state.contains(name, arguments, predicate.negated, check_negation=check_negation)
            if not rv:
                return '', FAILED
            return '', PASS
        elif len(unbound_arguments) == 1:
            arg = unbound_arguments[0]
            valid_arguments = list()

            options = objects[arg.typename]
            for o in options:
                bound_arguments[arg.name] = o
                name, arguments = _bind_arguments(predicate, bound_arguments)
                rv = state.contains(name, arguments, predicate.negated, check_negation=check_negation)
                if rv:
                    valid_arguments.append(o)
                del bound_arguments[arg.name]
            return arg.name, valid_arguments
        else:
            return '', TOO_MANY

    # @jacinle.log_function(verbose=False)
    def dfs(preconditions: Tuple[SBoolPredicateApplicationExpression, ...], bound_arguments: Dict[str, int]):
        """Inner DFS function.

        Args:
            preconditions: the preconditions to be satisfied.
            bound_arguments: a mapping from variable name to object.
        """

        # jacinle.log_function.print('dfs', bound_arguments, 'remaining preconditions:', len(preconditions))
        # import ipdb; ipdb.set_trace()

        for i, precondition in enumerate(preconditions):
            name, valid_arguments = compute_possible_grounding(precondition, bound_arguments)
            if valid_arguments == FAILED:
                # jacinle.log_function.print('Failed.')
                return list()
            elif valid_arguments == PASS:
                # jacinle.log_function.print('Pass.')
                return dfs(preconditions[:i] + preconditions[i + 1:], bound_arguments)
            elif valid_arguments == TOO_MANY:
                pass
            else:
                outputs = list()
                for arg in valid_arguments:
                    bound_arguments[name] = arg
                    outputs.extend(dfs(preconditions[:i] + preconditions[i + 1:], bound_arguments))
                    del bound_arguments[name]
                return outputs

        unbound_arguments = [arg for arg in operator.arguments if isinstance(arg, Variable) and arg.name not in bound_arguments]
        # print('unbound_arguments', unbound_arguments, bound_arguments)
        if len(unbound_arguments) == 0:
            # jacinle.log_function.print('Found a grounding:', bound_arguments)
            return [bound_arguments.copy()]

        unbound_arguments_possible_values = {arg.name: objects[arg.typename] for arg in unbound_arguments}

        name, valid_arguments = min(unbound_arguments_possible_values.items(), key=lambda x: len(x[1]))
        outputs = list()
        for arg in valid_arguments:
            bound_arguments[name] = arg
            # jacinle.log_function.print('{} = {}'.format(name, arg))
            outputs.extend(dfs(preconditions, bound_arguments))
        del bound_arguments[name]
        return outputs

    for operator in domain.operators.values():
        # jacinle.log_function.print(f'operator: {operator.name}')
        for bound_arguments in dfs(operator.preconditions, dict()):
            # jacinle.log_function.print('yield bound_arguments:', bound_arguments)
            yield operator, bound_arguments


def _check_precondition(state: SStateDict, operator: AtomicStripsOperator, bound_arguments: Dict[str, Union[int, str]]):
    for precondition in operator.preconditions:
        name, arguments = _bind_arguments(precondition, bound_arguments)
        if not state.contains(name, arguments, precondition.negated):
            return False
    return True


def _apply_operator(state: SStateDict, operator: AtomicStripsOperator, bound_arguments: Dict[str, Union[int, str]]):
    new_state = state.clone()
    for predicate in operator.del_effects:
        name, arguments = _bind_arguments(predicate, bound_arguments)
        new_state.remove(name, arguments)
    for predicate in operator.add_effects:
        name, arguments = _bind_arguments(predicate, bound_arguments)
        new_state.add(name, arguments)
    return new_state


def _ground_actions(actions: Tuple[Tuple[AtomicStripsOperator, Dict[str, str]], ...]) -> Tuple[AtomicStripsOperatorApplier, ...]:
    ground_operators = list()
    for operator, bound_arguments in actions:
        ground_operators.append(operator.ground(bound_arguments))
    return tuple(ground_operators)


def astrips_onthefly_search(domain: AtomicStripsDomain, problem: AtomicStripsProblem, verbose: bool = False, timeout: float = 300, max_expanded_nodes: int = 1000000) -> Optional[Tuple[AtomicStripsOperatorApplier, ...]]:
    objects = defaultdict(list)
    object2index = dict()

    for name, constant in domain.constants.items():
        objects[constant.dtype.typename].append(name)
        object2index[name] = len(object2index) - 1
    for name, typename in problem.objects.items():
        objects[typename].append(name)
        object2index[name] = len(object2index) - 1

    initial_state = SStateDict()
    for predicate in problem.initial_state:
        name, *args = predicate.split()
        initial_state.add(name, args)

    goal_conditions = problem.conjunctive_goal

    frontier = deque()
    frontier.append((initial_state, tuple()))
    explored = set()

    start_time = time.time()
    nr_expanded_nodes = 0
    while len(frontier) > 0:
        nr_expanded_nodes += 1
        if nr_expanded_nodes > max_expanded_nodes:
            break

        if nr_expanded_nodes % 100 == 0:
            if time.time() - start_time > timeout:
                print('astrips_onthefly_search::Timeout.')
                break

        state, plan = frontier.popleft()

        # action_strings = [f"{operator.name}({', '.join(bound_arguments.values())})" for operator, bound_arguments in plan]
        # print('State', state, 'Plan', action_strings)
        # print('Plan', action_strings)

        for operator, bound_arguments in _gen_applicable_actions(domain, objects, state):
            new_state = _apply_operator(state, operator, bound_arguments)
            new_state_set = new_state.as_state()
            if new_state_set not in explored:
                if new_state_set.issuperset(goal_conditions):
                    return _ground_actions(plan + ((operator, bound_arguments), ))

                frontier.append((new_state, plan + ((operator, bound_arguments), )))
                explored.add(new_state_set)

    return None

