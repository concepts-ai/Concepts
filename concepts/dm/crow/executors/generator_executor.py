#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : generator_executor.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/23/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import functools
import itertools

from typing import Any, Optional, Union, Tuple, List, Dict, Iterator, Callable
from collections import defaultdict
from tabulate import tabulate

from jacinle.logging import get_logger
from concepts.dsl.constraint import Constraint, AssignmentDict
from concepts.dm.crow.crow_generator import CrowGeneratorBase, CrowDirectedGenerator
from concepts.dm.crow.executors.crow_executor import CrowExecutor
from concepts.dm.crow.executors.python_function import CrowPythonFunctionRef

logger = get_logger(__file__)

__all__ = ['CrowGeneratorExecutor', 'wrap_singletime_function_to_iterator']


class CrowGeneratorExecutor(object):
    """The :class:`CrowGeneratorExecutor` is used to manage calls to generators in the function domain. It is particularly useful for keep tracking of historical values generated by the generators."""

    def __init__(self, executor: CrowExecutor, store_history: bool = True):
        """Initialize the generator manager.

        Args:
            executor: the executor.
            store_history: whether to store the historical values generated by the generators.
        """

        self.executor = executor
        self.generator_calls = defaultdict(list)
        self.generator_calls_successful = defaultdict(list)
        self.generator_calls_count = defaultdict(int)

        self._store_history = store_history

    executor: CrowExecutor
    """The executor."""

    generator_calls: Dict[str, List[Tuple[Tuple[Any, ...], Tuple[Any, ...]]]]
    """Mappings from generator names to the list of calls made to the generator, including a tuple of the arguments and a tuple of the return values."""

    generator_calls_successful: Dict[str, List[bool]]
    """Mappings from generator names to the list of Boolean values indicating whether the generated values lead to successful solution."""

    @property
    def store_history(self) -> bool:
        """Whether to store the historical values generated by the generators."""
        return self._store_history

    def call(self, g: CrowDirectedGenerator, max_generator_trials: int, args: Tuple[Any, ...], constraint_list: Optional[List[Constraint]] = None) -> Iterator[Tuple[Tuple[str, int], Any]]:
        """Call a generator.

        Args:
            g: the generator.
            max_generator_trials: the maximum number of trials to generate values.
            args: the arguments of the generator.
            constraint_list: the list of constraints to be satisfied by the generated values. This will be passed to the generator function if the list contains more than one constraint.

        Yields:
            A tuple of (index, generated value). The index is a tuple of (generator_name, value_index).
        """

        generator_name = g.name
        generator = wrap_singletime_function_to_iterator(
            self.executor.get_function_implementation(generator_name),
            max_generator_trials
        )
        if constraint_list is not None or not isinstance(constraint_list, list):
            generator = generator(*args, return_type=g.ftype.return_type)
        else:
            generator = generator(*args, constraint_list, return_type=g.ftype.return_type)

        self.generator_calls_count[generator_name] += 1
        first = True
        for result in generator:
            if self._store_history:
                self.generator_calls[generator_name].append((args, result))
                self.generator_calls_successful[generator_name].append(False)
            if not first:
                self.generator_calls_count[generator_name] += 1
            else:
                first = False
            index = generator_name, len(self.generator_calls[generator_name]) - 1

            if not isinstance(result, tuple) and g.ftype.is_singular_return:
                result = (result, )

            if not g.ftype.is_singular_return:
                assert len(result) == len(g.ftype.return_type.element_types)

            yield index, result

    def mark_success(self, assignment_dict: AssignmentDict):
        """Mark the values in an assignment dictionary as successful.

        Args:
            assignment_dict: the assignment dictionary.
        """
        assert self._store_history, 'Cannot mark success if history is not stored.'
        for _, value in assignment_dict.items():
            if value.generator_index is not None:
                name, index = value.generator_index
                self.generator_calls_successful[name][index] = True

    def export_generator_calls(self) -> Dict[str, List[Tuple[Tuple[Any, ...], Tuple[Any, ...], bool]]]:
        """Export the generator calls.

        Returns:
            a dictionary mapping from generator names to the list of calls made to the generator, including a tuple of the arguments
            and a tuple of the return values, and a Boolean value indicating whether the generated values lead to successful solution.
        """

        output_dict = defaultdict(list)
        for name, calls in self.generator_calls.items():
            for index, (args, result) in enumerate(calls):
                output_dict[name].append((args, result, self.generator_calls_successful[name][index]))
        return output_dict

    def export_generator_stats(self, divide_by: float = 1) -> str:
        """Export the generator statistics.

        Returns:
            a string containing the generator statistics.
        """

        rows = list()
        for name, count in self.generator_calls_count.items():
            rows.append((name, count / divide_by))
        rows.append(('Total', sum(count / divide_by for count in self.generator_calls_count.values())))
        return tabulate(rows, headers=['Generator', 'Calls'])


def wrap_singletime_function_to_iterator(function: CrowPythonFunctionRef, max_examples: int) -> Callable[..., Iterator[Any]]:
    """Wrap a function that returns a single value to an iterator function.

    Args:
        function: the function.
        max_examples: the maximum number of examples.

    Returns:
        the iterator function.
    """

    if function.is_iterator:
        @functools.wraps(function)
        def wrapped(*args, **kwargs) -> Iterator[Any]:
            try:
                yield from itertools.islice(function.iter_from(*args, **kwargs), max_examples)
            except Exception as e:
                logger.warning(f'Exception raised when calling generator {function}: {e}')

        return wrapped

    @functools.wraps(function)
    def wrapped(*args, **kwargs) -> Iterator[Any]:
        for _ in range(max_examples):
            rv = function(*args, **kwargs)
            if rv is None:
                break
            yield rv

    return wrapped

