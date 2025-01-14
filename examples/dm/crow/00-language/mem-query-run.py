#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : mem-query-run.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/08/2025
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os.path as osp
from typing import Dict, Optional, Tuple, Union

import concepts.dsl.all as T
import concepts.dsl.expression as E
import concepts.dm.crow as crow
from concepts.dm.crow import CrowState
from concepts.dsl.constraint import ConstraintSatisfactionProblem
from concepts.dsl.dsl_types import ObjectConstant
from concepts.dsl.expression import ObjectOrValueOutputExpression
from concepts.dsl.tensor_value import TensorValue


class CustomPerceptionInterface(crow.CrowPerceptionInterface):
    def __init__(self, domain: crow.CrowDomain):
        super().__init__()
        self.domain = domain

    def mem_query(
        self, expression: ObjectOrValueOutputExpression, state: CrowState, csp: Optional[ConstraintSatisfactionProblem] = None,
        bounded_variables: Optional[Dict[str, Union[TensorValue, ObjectConstant]]] = None,
        state_index: Optional[int] = None
    ) -> Tuple[CrowState, ConstraintSatisfactionProblem, Dict[str, Union[TensorValue, ObjectConstant]]]:
        return self.simple_mem_query(expression, state), csp, bounded_variables

    def simple_mem_query(self, expression: ObjectOrValueOutputExpression, state: CrowState) -> CrowState:
        if isinstance(expression, E.FindAllExpression):
            return self._mem_query_find_all(expression.variable, expression.expression, state)
        else:
            raise NotImplementedError(f"Unsupported expression type: {type(expression)}")

    def _mem_query_find_all(self, variable: T.Variable, expression: E.ValueOutputExpression, state: CrowState) -> CrowState:
        if isinstance(expression, E.FunctionApplicationExpression):
            if expression.function.nr_arguments == 1 and isinstance(expression.arguments[0], E.VariableExpression) and expression.arguments[0].name == variable.name:
                object_list = self._get_object_list_by_property(expression.function.name)
                new_object_list = [x for x in object_list if x not in state.object_names]
                new_state = state.clone_with_new_objects(self.domain, new_object_list, [self.domain.types['Object'] for _ in new_object_list])
                self._annotate_new_state(new_state, new_object_list)
                return new_state
            else:
                raise NotImplementedError(f"Unsupported function application: {expression}")
        else:
            raise NotImplementedError(f"Unsupported expression type: {type(expression)}")

    def _get_object_list_by_property(self, property_name: str) -> list[str]:
        return ['A', 'B', 'C', 'D']

    def _annotate_new_state(self, new_state: CrowState, new_object_list: list[str]):
        for obj in new_object_list:
            if obj == 'D':
                new_state.fast_set_value('is_good', ['D'], True)


def main():
    problem = crow.load_problem_file(osp.join(osp.dirname(__file__), 'mem-query.cdl'))
    domain = problem.domain
    perception_interface = CustomPerceptionInterface(domain)

    results = crow.crow_regression(problem, return_results=True, algo='priority_tree_v1', perception_interface=perception_interface)

    if len(results) == 0:
        print("No results found.")
        return

    for result in results:
        print(result.controller_actions)


if __name__ == '__main__':
    main()
