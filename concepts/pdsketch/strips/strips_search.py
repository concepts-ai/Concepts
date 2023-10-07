#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : strips_search.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/20/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import jacinle
import jactorch
from typing import Callable

from concepts.pdsketch.strips.strips_expression import SState
from concepts.pdsketch.strips.strips_grounding import GStripsProblem
from concepts.pdsketch.strips.strips_heuristics import StripsHeuristic
from concepts.algorithm.search.heuristic_search import run_heuristic_search

__all__ = ['strips_brute_force_search', 'strips_heuristic_search', 'get_priority_func']


def print_task(task: GStripsProblem, func):
    func_name = func.__name__

    print(f'{func_name}::task.goal={task.goal}')
    print(f'{func_name}::task.facts={len(task.facts) if task.facts is not None else "N/A"}')
    print(f'{func_name}::task.operators={len(task.operators)}')


@jactorch.no_grad_func
def strips_brute_force_search(
    task: GStripsProblem, max_depth=5, verbose=True
):
    if verbose:
        print_task(task, strips_brute_force_search)

    goal_func = task.goal.compile()
    states = [(task.state, tuple())]
    visited = set()
    visited.add(task.state)

    pbar = None
    if verbose:
        pbar = jacinle.tqdm_pbar(desc='strips_brute_force_search::depth=0')
    with jacinle.cond_with(pbar, verbose):
        for depth in range(max_depth):
            next_states = list()
            for s, traj in states:
                for a in task.operators:
                    if verbose:
                        pbar.update()
                    if a.applicable(s):
                        ns = a.apply(s)

                        if ns not in visited:
                            visited.add(ns)
                        nt = traj + (a, )
                        next_states.append((ns, nt))
                        if goal_func(ns):
                            return nt
            states = next_states
            if verbose:
                pbar.set_description(f'strips_brute_force_search::depth={depth}, states={len(states)}')
    return None


def get_priority_func(heuristic: Callable[[SState], float], weight: float) -> Callable[[SState, int], float]:
    if weight == 1:
        def priority_fun(state, g, heuristic=heuristic):
            return heuristic(state) + g
    elif weight == float('inf'):
        def priority_fun(state, g, heuristic=heuristic):
            return heuristic(state)
    else:
        def priority_fun(state, g, heuristic=heuristic, weight=weight):
            return heuristic(state) + g * weight
    return priority_fun


@jactorch.no_grad_func
def strips_heuristic_search(
    task: GStripsProblem, heuristic: StripsHeuristic, *,
    max_expansions=int(1e9), verbose=False,
    heuristic_weight=float('inf')
):
    if verbose:
        print_task(task, strips_heuristic_search)
        print('strips_heuristic_search::init_heuristic={}'.format(heuristic.compute(task.state)))

    goal_func = task.goal.compile()

    pbar = None
    if verbose:
        pbar = jacinle.tqdm_pbar(desc='strips_heuristic_search::expanding')

    def check_goal(state: SState, gf=goal_func):
        return gf(state)

    def get_successors(state: GStripsProblem, actions=task.operators):
        for action in actions:
            if verbose:
                pbar.update()
            if action.applicable(state):
                yield action, action.apply(state), 1

    with jacinle.cond_with(pbar, verbose):
        return run_heuristic_search(
            task.state,
            check_goal,
            get_priority_func(heuristic.compute, heuristic_weight),
            get_successors,
            max_expansions=max_expansions
        )[1]

