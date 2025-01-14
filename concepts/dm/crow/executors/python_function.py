#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : python_function.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/17/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import itertools
from dataclasses import dataclass
from typing import Any, Optional, Union, Iterator, Sequence, Tuple, Dict, Callable, TYPE_CHECKING

import numpy as np
import torch

from concepts.dsl.dsl_functions import Function
from concepts.dsl.dsl_types import BOOL, PyObjValueType, ObjectType, UniformSequenceType, TupleType, QINDEX, TensorValueTypeBase
from concepts.dsl.executors.tensor_value_executor import TensorValueExecutorReturnType
from concepts.dsl.expression import Expression
from concepts.dsl.tensor_value import TensorValue, TensorizedPyObjValues
from concepts.dsl.tensor_state import StateObjectReference
from concepts.dsl.tensor_value_utils import expand_argument_values
from concepts.dsl.value import ListValue

from concepts.dm.crow.crow_domain import CrowState

if TYPE_CHECKING:
    from concepts.dm.crow.executors.crow_executor import CrowExecutor

__all__ = ['CrowPythonFunctionRef', 'CrowPythonFunctionCrossRef', 'config_function_implementation', 'CrowSGC']


class CrowPythonFunctionRef(object):
    """A reference to a Python function.

    This class is used to wrap external function implementations in domains.
    """

    def __init__(
        self, function: Callable, function_quantized: Optional[Callable] = None,
        *,
        return_type: Optional[Union[TensorValueTypeBase, PyObjValueType, Tuple[Union[TensorValueTypeBase, PyObjValueType], ...]]] = None,
        support_batch: bool = False, auto_broadcast: bool = False,
        use_object_names: bool = True, unwrap_values: Optional[bool] = None,
        include_executor_args: bool = False, is_iterator: bool = False, is_sgc_function: bool = False,
        executor: Optional['CrowExecutor'] = None
    ):
        """Initialize a Python function reference.

        Args:
            function: the function to be wrapped.
            function_quantized: the quantized version of the function (can be None).
            support_batch: whether the function supports batched inputs.
            auto_broadcast: whether the executor should automatically broadcast the arguments before calling the function.
            use_object_names: whether the executor should use the object names in the state (instead of the index).
            unwrap_values: whether the executor should unwrap the tensor values before calling the function.
            include_executor_args: whether the caller should include the executor as the first argument.
            is_iterator: whether the function is an iterator.
            is_sgc_function: whether the function is an SGC function (state-goal-constraints).
            executor: the executor that is using this function reference.
        """

        self.function = function
        self.function_quantized = function_quantized
        self.return_type = return_type
        self.support_batch = support_batch
        self.auto_broadcast = auto_broadcast
        self.use_object_names = use_object_names
        if unwrap_values is None:
            unwrap_values = not support_batch or auto_broadcast
        self.unwrap_values = unwrap_values
        self.include_executor_args = include_executor_args
        self.is_iterator = is_iterator
        self.is_sgc_function = is_sgc_function
        self._executor = executor

    function: Callable
    """The internal implementation of the function."""

    function_quantized: Optional[Callable]
    """The quantized version of the function (can be None)."""

    return_type: Optional[Union[TensorValueTypeBase, PyObjValueType, Tuple[Union[TensorValueTypeBase, PyObjValueType], ...]]]
    """The return type of the function."""

    support_batch: bool
    """Whether the function supports batched inputs."""

    auto_broadcast: bool
    """Whether the executor should automatically broadcast the arguments before calling the function."""

    use_object_names: bool
    """Whether the executor should use the object names in the state (instead of the index)."""

    unwrap_values: bool
    """Whether the executor should unwrap the tensor values before calling the function."""

    include_executor_args: bool
    """Whether the caller should include the executor as the first argument."""

    is_iterator: bool
    """Whether the function is an iterator."""

    is_sgc_function: bool
    """Whether the function is an SGC function (state-goal-constraints)."""

    def set_executor(self, executor: 'CrowExecutor') -> 'CrowPythonFunctionRef':
        """Set the executor that is using this function reference.

        Args:
            executor: the executor that is using this function reference.

        Returns:
            the function reference itself.
        """
        self._executor = executor
        return self

    def __str__(self) -> str:
        return (
            'PythonFunctionRef('
            f'{self.function}, support_batch={self.support_batch}, auto_broadcast={self.auto_broadcast}, '
            f'use_object_names={self.use_object_names}, unwrap_values={self.unwrap_values}, include_executor_args={self.include_executor_args}, '
            f'is_iterator={self.is_iterator}), is_sgc_function={self.is_sgc_function}'
        )

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def flags(self) -> Dict[str, bool]:
        return {
            'support_batch': self.support_batch,
            'auto_broadcast': self.auto_broadcast,
            'use_object_names': self.use_object_names,
            'unwrap_values': self.unwrap_values,
            'include_executor_args': self.include_executor_args,
            'is_iterator': self.is_iterator,
            'is_sgc_function': self.is_sgc_function
        }

    def forward(
        self, argument_values: Sequence[TensorValueExecutorReturnType], return_type: Optional[Union[TensorValueTypeBase, PyObjValueType]] = None,
        additional_parameters: Optional[Sequence[Any]] = None, auto_broadcast: bool = True, wrap_rv: bool = True,
        function_def: Optional[Function] = None
    ) -> Union[TensorValue, Tuple[TensorValue, ...]]:
        """Call the function.

        Args:
            argument_values: the arguments to the function.
            return_type: the type of the return value.
            additional_parameters: the additional parameters to the function.
            auto_broadcast: whether the executor should automatically broadcast the arguments before calling the function.
            wrap_rv: whether the executor should wrap the return value.
            function_def: the function definition, used to wrap return values and to handle QINDEX.

        Returns:
            the result of the function.
        """
        function = self.function

        if self.unwrap_values:
            if self.use_object_names:
                argument_values = [v.name if isinstance(v, StateObjectReference) else v for v in argument_values]
            else:
                argument_values = [v.index if isinstance(v, StateObjectReference) else v for v in argument_values]

        if self.support_batch:
            if self.auto_broadcast and auto_broadcast:
                argument_values = expand_argument_values(argument_values)
        else:
            # If the function does not support batch, we will have to automatically batch the processing.
            argument_values = expand_argument_values(argument_values)

        argument_values_flat = list(argument_values)

        if additional_parameters is not None:
            additional_parameters = list(additional_parameters)
        else:
            additional_parameters = []

        if self.is_sgc_function:
            assert self._executor is not None, 'Executor is None.'
            additional_parameters.insert(0, self._executor.sgc)

        if self.include_executor_args:
            assert self._executor is not None, 'Executor is None.'
            additional_parameters.insert(0, self._executor)

        argument_values_flat = additional_parameters + argument_values_flat

        if QINDEX in argument_values_flat:
            if not self.support_batch:
                if function_def is None:
                    raise RuntimeError('For functions with QINDEX, the function_def argument must be provided.')
                rv = self.forward_internal_autobatch(function_def, function, argument_values_flat)
            else:
                if self.unwrap_values:
                    argument_values_flat = [v.tensor if isinstance(v, TensorValue) else v for v in argument_values_flat]
                    if not self.support_batch:
                        argument_values_flat = [v.item() if isinstance(v, TensorizedPyObjValues) else v for v in argument_values_flat]
                rv = function(*argument_values_flat)
        else:
            if self.unwrap_values:
                argument_values_flat = [v.tensor if isinstance(v, TensorValue) else v for v in argument_values_flat]
                if not self.support_batch:
                    argument_values_flat = [v.item() if isinstance(v, TensorizedPyObjValues) else v for v in argument_values_flat]
            rv = function(*argument_values_flat)

        if not wrap_rv:
            return rv
        return self._wrap_rv(rv, return_type, argument_values, auto_broadcast)

    def forward_internal_autobatch(self, function_def, function, argument_values_flat):
        options_per_argument = list()
        output_dims = list()
        batch_variables = list()
        for i, arg in enumerate(argument_values_flat):
            if arg is QINDEX:
                objects = self._executor.state.object_type2name[function_def.ftype.argument_types[i].typename]
                if self.use_object_names:
                    options_per_argument.append(objects)
                else:
                    options_per_argument.append(list(range(len(objects))))
                output_dims.append(len(objects))
                batch_variables.append(function_def.ftype.argument_names[i])
            else:
                options_per_argument.append([arg])

        rtype = function_def.ftype.return_type
        if rtype != BOOL:
            raise TypeError('Only BOOL is supported for auto-batch functions with QINDEX.')
        output = torch.zeros(np.prod(output_dims), dtype=torch.bool)
        options = list(itertools.product(*options_per_argument))
        for i, option in enumerate(options):
            rv = function(*option)
            if isinstance(rv, TensorValue):
                rv = rv.item()
            if isinstance(rv, torch.Tensor):
                if output.device != rv.device:
                    output.tensor = output.to(rv.device)
            output[i] = rv
        output = TensorValue.from_tensor(output.reshape(output_dims), rtype, batch_variables=batch_variables)
        return output

    def forward_internal_autobatch2(self, function_def, function, argument_values_flat):
        options_per_argument = list()
        output_dims = list()
        batch_variables = list()
        other_batch_variables = None
        for i, arg in enumerate(argument_values_flat):
            if arg is QINDEX:
                objects = self._executor.state.object_type2name[function_def.ftype.argument_types[i].typename]
                if self.use_object_names:
                    options_per_argument.append(objects)
                else:
                    options_per_argument.append(list(range(len(objects))))
                output_dims.append(len(objects))
                batch_variables.append(function_def.ftype.argument_names[i])
            else:
                options_per_argument.append([arg])
                if isinstance(arg, TensorValue):
                    if other_batch_variables is None:
                        other_batch_variables = arg.batch_variables
                    else:
                        assert tuple(other_batch_variables) == tuple(arg.batch_variables), 'Inconsistent batch variables.'

        rtype = function_def.ftype.return_type
        if not isinstance(rtype, (TensorValueTypeBase, PyObjValueType)):
            raise TypeError('Only TensorValueTypeBase and PyObjValueType are supported for auto-batch functions with QINDEX.')

        options = list(itertools.product(*options_per_argument))
        for i, option in enumerate(options):
            option = list(option)
            pass

    def forward_generator(
        self, argument_values: Sequence[TensorValueExecutorReturnType], return_type: Optional[Union[TensorValueTypeBase, PyObjValueType]] = None,
        auto_broadcast: bool = True, wrap_rv: bool = True
    ) -> Union[Iterator[TensorValue], Iterator[Tuple[TensorValue, ...]]]:
        """Call the function and return a generator.

        Args:
            argument_values: the arguments to the function.
            return_type: the type of the return value.
            auto_broadcast: whether the executor should automatically broadcast the arguments before calling the function.
            wrap_rv: whether the executor should wrap the
        """

        generator = self.forward(argument_values, return_type=return_type, auto_broadcast=auto_broadcast, wrap_rv=False)
        if not wrap_rv:
            yield from generator
        else:
            for v in generator:
                yield self._wrap_rv(v, return_type, argument_values, auto_broadcast)

    def forward_sgc_function(
        self, state: CrowState, goal: Expression, constraints: Sequence[Expression], additional_arguments: Sequence[TensorValueExecutorReturnType],
        return_type: Optional[Union[TensorValueTypeBase, PyObjValueType]] = None,
        auto_broadcast: bool = True, wrap_rv: bool = True
    ):
        """Call an SGC function (state-goal-constraints) function.

        Args:
            state: the current state.
            goal: the goal expression.
            constraints: the constraints, as a list of expressions.
            additional_arguments: the additional arguments.
            return_type: the type of the return value.
            auto_broadcast: whether the executor should automatically broadcast the arguments before calling the function.
            wrap_rv: whether the executor should wrap the return value.
        """
        return self.forward(additional_arguments, return_type=return_type, additional_parameters=(state, goal, constraints), auto_broadcast=auto_broadcast, wrap_rv=wrap_rv)

    def __call__(self, *args, return_type: Optional[Union[TensorValueTypeBase, PyObjValueType]] = None, auto_broadcast: bool = True, wrap_rv: bool = True) -> Union[TensorValue, Tuple[TensorValue, ...]]:
        assert not self.is_iterator, 'Use iter_from to call an iterator function.'
        return self.forward(args, return_type=return_type, auto_broadcast=auto_broadcast, wrap_rv=wrap_rv)

    def iter_from(self, *args, return_type: Optional[Union[TensorValueTypeBase, PyObjValueType]] = None, auto_broadcast: bool = True, wrap_rv: bool = True) -> Union[Iterator[TensorValue], Iterator[Tuple[TensorValue, ...]]]:
        assert self.is_iterator, 'Use __call__ to call a non-iterator function.'
        return self.forward_generator(args, return_type=return_type, auto_broadcast=auto_broadcast, wrap_rv=wrap_rv)

    def _wrap_rv(self, rv, return_type, argument_values, auto_broadcast):
        if isinstance(rv, (TensorValue, ListValue)):
            return rv
        elif isinstance(rv, tuple) and all(isinstance(v, (TensorValue, ListValue)) for v in rv):
            return rv

        if return_type is None:
            return_type = self.return_type
        if return_type is None:
            raise RuntimeError('Return type can not be None if the function return is not a TensorValue.')

        if isinstance(return_type, TupleType):
            if not isinstance(rv, tuple) and len(return_type) == 1:
                rv = (rv, )
            return tuple(self._wrap_single_rv(v, t, argument_values, auto_broadcast) for v, t in zip(rv, return_type.element_types))
        else:
            return self._wrap_single_rv(rv, return_type, argument_values, auto_broadcast)

    def _wrap_single_rv(self, rv, return_type, argument_values, auto_broadcast):
        if isinstance(rv, (TensorValue, ListValue)):
            return rv
        # TODO(Jiayuan Mao @ 2023/11/18): have an actual type check.
        if return_type.alias is not None and return_type.alias.startswith('__') and return_type.alias.endswith('__'):
            return rv
        if not self.support_batch:
            if isinstance(return_type, PyObjValueType):
                if isinstance(rv, TensorizedPyObjValues):
                    return TensorValue.from_tensorized_pyobj(rv, return_type)
                return TensorValue.from_scalar(rv, return_type)
            elif isinstance(return_type, ObjectType):
                if isinstance(rv, str):
                    return self._executor.state.get_state_object_reference(return_type, name=rv)
                elif isinstance(rv, int):
                    return self._executor.state.get_state_object_reference(return_type, index=rv)
                else:
                    return rv
            elif isinstance(return_type, UniformSequenceType) and isinstance(return_type.element_type, ObjectType):
                if isinstance(rv, (list, tuple)):
                    if len(rv) == 0:
                        return self._executor.state.get_state_object_list(return_type.element_type, [])
                    else:
                        if isinstance(rv[0], str):
                            return self._executor.state.get_state_object_list(return_type.element_type, names=rv)
                        elif isinstance(rv[0], int):
                            return self._executor.state.get_state_object_list(return_type.element_type, indices=rv)
                        else:
                            raise ValueError(f'Unsupported return type: {rv}')
                else:
                    return rv
            else:
                if isinstance(rv, torch.Tensor):
                    return TensorValue.from_tensor(rv, return_type)
                elif isinstance(rv, (bool, int, float)):
                    return TensorValue.from_scalar(rv, return_type)
                else:
                    raise ValueError(f'Unsupported return type: {type(rv)}')
        else:
            if isinstance(return_type, PyObjValueType):
                raise TypeError('Cannot return a PyObjValueType for a batched function.')
            else:
                if isinstance(rv, torch.Tensor):
                    first_tensor_arg = None
                    for arg in argument_values:
                        if isinstance(arg, TensorValue):
                            first_tensor_arg = arg
                            break
                    if not self.auto_broadcast or not auto_broadcast or first_tensor_arg is None:
                        raise ValueError('Cannot return a raw PyTorch tensor for a batched function without auto_broadcast.')
                    return TensorValue.from_tensor(rv, return_type, batch_variables=first_tensor_arg.batch_variables, batch_dims=first_tensor_arg.batch_dims)
                else:
                    raise ValueError(f'Unsupported return type: {type(rv)}')


class CrowPythonFunctionCrossRef(object):
    def __init__(self, cross_ref_name: str):
        self.cross_ref_name = cross_ref_name


def config_function_implementation(
    function: Optional[Callable] = None, *, function_quantized: Optional[Callable] = None,
    support_batch: bool = False, auto_broadcast: bool = True, use_object_names: bool = True, unwrap_values: Optional[bool] = None, include_executor_args: bool = False,
    is_iterator: bool = False, is_sgc_function: bool = False
) -> Callable:
    """Configure the implementation of a function in a domain.

    Args:
        function: the function to be wrapped.
        function_quantized: the quantized version of the function (can be None).
        support_batch: whether the function supports batched inputs.
        auto_broadcast: whether the executor should automatically broadcast the arguments before calling the function.
        use_object_names: whether the executor should use object names instead of indices.
        unwrap_values: whether the executor should unwrap the values before calling the function.
        include_executor_args: whether the executor should include itself as the first argument.
        is_iterator: whether the function is an iterator.
        is_sgc_function: whether the function is an SGC function.

    Returns:
        the decorator.
    """

    function_implementation_configs = {
        'function_quantized': function_quantized,
        'support_batch': support_batch,
        'auto_broadcast': auto_broadcast,
        'use_object_names': use_object_names,
        'unwrap_values': unwrap_values,
        'include_executor_args': include_executor_args,
        'is_iterator': is_iterator,
        'is_sgc_function': is_sgc_function
    }

    def wrapper(function: Callable, configs=function_implementation_configs):
        return CrowPythonFunctionRef(function, **configs)

    if function is None:
        return wrapper
    return wrapper(function)


def _check_no_quantized_arguments(arguments):
    """A helper function to check that there are no quantized arguments. This function handles the migration from quantized tensor CSP computation to non-quantized tensor CSP computation."""
    # TODO(Jiayuan Mao @ 2023/08/15): remove this after the migration.
    for arg in arguments:
        if isinstance(arg, TensorValue):
            if isinstance(arg.dtype, TensorValueTypeBase):
                if arg.quantized and not arg.dtype.is_intrinsically_quantized():
                    raise RuntimeError('Quantized arguments are not supported.')


@dataclass
class CrowSGC(object):
    state: CrowState
    goal: Expression
    constraints: Sequence[Expression]
