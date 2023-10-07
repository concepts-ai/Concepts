#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : atomic_strips_regression_search.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/13/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import itertools
from typing import List, Set

import jacinle

from concepts.pdsketch.strips.strips_expression import SProposition
from concepts.pdsketch.strips.atomic_strips_domain import (
    AtomicStripsDomain, AtomicStripsProblem,
    AtomicStripsOperator, AtomicStripsRegressionRule,
    AtomicStripsOperatorApplier, AtomicStripsRegressionRuleApplier,
    AtomicStripsGroundedAchieveExpression
)


def gen_all_grounded_actions_and_rules(domain: AtomicStripsDomain, problem: AtomicStripsProblem, mode: str) -> List[AtomicStripsOperatorApplier]:
    assert mode in ('action', 'rule')
    all_actions = list()

    if mode == 'action':
        operator_list = domain.operators
    elif mode == 'rule':
        operator_list = domain.regression_rules
    else:
        raise ValueError('Unknown mode: {}.'.format(mode))

    for operator in operator_list.values():
        candidate_arguments = [
            problem.objects_type2names[operator.arguments[i].dtype.typename]
            for i in range(len(operator.arguments))
        ]
        for arg_list in itertools.product(*candidate_arguments):
            action = operator(*arg_list)
            static_check = True
            for i, prop in enumerate(action.preconditions):
                if domain.predicates[operator.preconditions[i].name].is_static:
                    if prop not in problem.initial_state:
                        static_check = False
                        break
            if not static_check:
                continue
            all_actions.append(action)

    return all_actions


def astrips_regression_search_1(domain: AtomicStripsDomain, problem: AtomicStripsProblem, verbose: bool = False) -> List[AtomicStripsOperatorApplier]:
    """Search for a plan using regression search."""

    assert problem.conjunctive_goal is not None, "Only conjunctive goals are supported."
    assert len(problem.conjunctive_goal) == 1, "Only single-goal problems are supported."

    for operator in domain.operators.values():
        for precondition in operator.preconditions:
            if precondition.negated:
                raise NotImplementedError('astrips_regression_search does not support negated preconditions.')
    for regression_rule in domain.regression_rules.values():
        for precondition in regression_rule.preconditions:
            if precondition.negated:
                raise NotImplementedError('astrips_regression_search does not support negated preconditions.')
        if len(regression_rule.preconstraints) > 0:
            raise NotImplementedError('astrips_regression_search does not support preconstraints.')

    all_rules = gen_all_grounded_actions_and_rules(domain, problem, 'rule')

    def find_applicable_rules(state: Set[SProposition], goal: SProposition, maintains: Set[SProposition]):
        applicable_rules = list()
        for rule in all_rules:
            if rule.goal == goal and state.issuperset(rule.preconditions):
                applicable_rules.append(rule)
        assert len(applicable_rules) == 1, "Only one applicable rule is allowed."
        return applicable_rules[0]

    @jacinle.log_function(verbose=False)
    def dfs(state: Set[SProposition], goal: SProposition, maintains: Set[SProposition]):
        rule = find_applicable_rules(state, goal, maintains)
        if verbose:
            jacinle.log_function.print(f'Current state: {state}, goal: {goal}, maintains: {maintains} => rule: {rule}')
        actions = list()
        for item in rule.body:
            if isinstance(item, AtomicStripsGroundedAchieveExpression):
                if item.goal not in state:
                    if verbose:
                        jacinle.log_function.print(f'{str(item)}')
                    state, sub_actions = dfs(state, item.goal, maintains.union(item.maintains))
                    actions.extend(sub_actions)
                else:
                    if verbose:
                        jacinle.log_function.print(f'{str(item)} (skipped)')
            elif isinstance(item, AtomicStripsOperatorApplier):
                if verbose:
                    jacinle.log_function.print(f'do({str(item)})')
                actions.append(item)
                assert state.issuperset(item.preconditions), "The preconditions of an action should be satisfied."
                state = (state - frozenset(item.del_effects)).union(frozenset(item.add_effects))
            else:
                raise ValueError('Unknown item: {}.'.format(item))
        return state, actions

    end_state, actions = dfs(problem.initial_state, problem.conjunctive_goal[0], set())
    return actions

