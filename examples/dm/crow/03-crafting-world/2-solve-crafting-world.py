#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : 2-solve-crafting-world.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 06/21/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import time
import jacinle

import concepts.dm.crow as crow
from concepts.benchmark.gridworld.crafting_world.crow_domains import get_crafting_world_domain_filename
from concepts.benchmark.gridworld.crafting_world.crafting_world_crow_gen.cw_20230913_mixed import gen_v20230913_instance_record, gen_v20240622_instance_record, gen_v20240625_instance_record

parser = jacinle.JacArgumentParser()
parser.add_argument('--target', choices=['single-goal', 'double-goal', 'custom'], default='custom')
args = parser.parse_args()


def main_single_goal():
    domain = crow.load_domain_file(get_crafting_world_domain_filename())
    domain.print_summary()

    problem_func = gen_v20230913_instance_record

    for i in range(23):
        record = problem_func(f'test-{i}', 'test', goal_index=i, station_agnostic=True)
        plan(domain, record)


def main_double_goal():
    domain = crow.load_domain_file(get_crafting_world_domain_filename())
    domain.print_summary()

    problem_func = gen_v20240622_instance_record

    for i in range(23):
        for j in range(23):
            record = problem_func(f'test-{i}', f'test-{j}', 'test', goal_index1=i, goal_index2=j, station_agnostic=True)
            plan(domain, record)


def main_custom():
    domain = crow.load_domain_file(get_crafting_world_domain_filename())
    problem_func = gen_v20240625_instance_record

    record = problem_func('test-1', ['boat', 'cooked_potato', 'beetroot', 'sword'])
    plan(domain, record)


def plan(domain, record):
    problem_cdl = record['problem_cdl']
    print(problem_cdl)
    problem = crow.load_problem_string(problem_cdl, domain=domain)

    start_time = time.time()
    rv, stat = crow.crow_regression(
        domain, problem, min_search_depth=15, max_search_depth=15,
        is_goal_ordered=False, is_goal_serializable=True, always_commit_skeleton=True,
        verbose=False
    )

    if len(rv) == 0:
        raise ValueError(f'No plan found for goal {problem.goal}.')

    table = list()
    solution = rv[0]
    table.append(('goal', str(problem.goal)))
    table.append(('plan_length', len(solution)))
    table.append(('plan', '\n'.join([str(a) for a in solution])))
    table.append(('time', f'{time.time() - start_time:.3f}s'))
    table.extend(stat.items())
    print(jacinle.tabulate(table, tablefmt='simple'))


if __name__ == '__main__':
    mapping = {
        'single-goal': main_single_goal,
        'double-goal': main_double_goal,
        'custom': main_custom,
    }
    mapping[args.target]()
