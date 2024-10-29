#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : 2-solve-blocksworld.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/04/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import jacinle
import concepts.dm.crow as crow

parser = jacinle.JacArgumentParser()
parser.add_argument('--domain', default='blocksworld.cdl')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()


def main():
    domain = crow.load_domain_file(args.domain)
    problem = crow.load_problem_file('blocksworld-problem-sussman.cdl', domain=domain)

    state = problem.state
    print('=' * 80)
    print('Initial state:')
    print(state)

    plan(domain, problem, 'clear(A)')
    plan(domain, problem, 'clear(A) and clear(C)')
    plan(domain, problem, 'on(B, C) and holding(A)')
    plan(domain, problem, 'on(B, C) and on(A, B)')


def plan(domain, problem, goal):
    candidate_plans, search_stat = crow.crow_regression(
        domain, problem, goal=goal, min_search_depth=5, max_search_depth=7,
        is_goal_ordered=True, is_goal_serializable=False, always_commit_skeleton=True
    )
    table = list()
    for p in candidate_plans:
        table.append('; '.join(map(str, p)))
    print()
    print('=' * 80)
    print('Goal:', goal)
    for i, row in enumerate(table):
        print(f'Plan {i}:', row)
    print(search_stat)
    input('Press Enter to continue...')


if __name__ == '__main__':
    main()

