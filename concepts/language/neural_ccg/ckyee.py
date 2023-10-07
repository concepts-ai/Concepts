#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : ckyee.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/11/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""NeuralCCG with CKY-EE (CKY with Expected Execution)."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Union, Iterable, Tuple, List, Dict, Callable

import torch
import torch.nn.functional as F

import jacinle.random as jacrandom
import jactorch

from concepts.dsl.function_domain import FunctionDomain
from concepts.dsl.value import Value
from concepts.dsl.executors.function_domain_executor import FunctionDomainExecutor
from concepts.language.ccg.composition import CCGCompositionSystem, CCGCoordinationImmNode
from concepts.language.neural_ccg.grammar import NeuralCCGSemanticsPartialTypeLex, NeuralCCGSemantics, NeuralCCGNode, NeuralCCG
from concepts.language.neural_ccg.search import NeuralCCGLexiconSearchResult


__all__ = ['CKYEEExpectationFunction', 'CKYEEExpectationConfig', 'NeuralCKYEE']

_profile = getattr(__builtins__, 'profile', lambda x: x)


class CKYEEExpectationFunction(object):
    """A collection of functions to perform expected execution. This class maintains a set of functions of the following form:

    .. code-block:: python

        def expectation_<typename>(self, values: List[Value], weights: List[torch.Tensor]) -> Tuple[Optional[Value], Optional[Tuple[torch.Tensor, float]]]:
            ...

    where ``typename`` is the name of a type, and ``values`` is a list of values of the type, and ``weights`` is a list of corresponding weights.
    The function should return a tuple of (expectation, (sum_weight, max_weight)), where ``expectation`` is the expectation of the values,
    as a :class:`~concepts.dsl.value.Value` instance, and ``sum_weight`` is the sum of the weights (as a PyTorch tensor),
    and ``max_weight`` is the maximum of the weights (as a float).
    """

    def __init__(self, domain: FunctionDomain):
        """Initialize the expectation function.

        Args:
            domain: the function domain.
        """
        self._domain = domain
        self._expectation_functions = dict()
        self._init_expectation_functions()

    _domain: FunctionDomain
    _expectation_functions: Dict[str, Callable[[List[Value], List[torch.Tensor]], Tuple[torch.Tensor, Tuple[torch.Tensor, float]]]]

    def register_function(self, typename: str, function: Callable[[List[Value], List[torch.Tensor]], Tuple[torch.Tensor, Tuple[torch.Tensor, float]]]):
        """Register a function to compute the expectation of a list of values.

        Args:
            typename: the name of the type.
            function: the function to compute the expectation of a list of values.
        """
        self._expectation_functions[typename] = function

    def get_function(self, typename: str) -> Optional[Callable[[List[Value], List[torch.Tensor]], Tuple[torch.Tensor, Tuple[torch.Tensor, float]]]]:
        """Get the registered function for a type.

        Args:
            typename: the name of the type.

        Returns:
            The registered function.
        """
        return self._expectation_functions.get(typename, None)

    def _init_expectation_functions(self):
        for typename in self._domain.types:
            funcname = 'expectation_' + typename
            if hasattr(self, funcname):
                self.register_function(typename, getattr(self, funcname))

    def expectation(self, values: List[Value], weights: List[torch.Tensor]) -> Tuple[Optional[Value], Optional[Tuple[torch.Tensor, float]]]:
        """Compute the expectation of a list of values.

        Args:
            values: a list of values.
            weights: a list of corresponding weights.

        Returns:
            A tuple of (expectation, (sum_weight, max_weight)).
            The ``expectation`` is the expectation of the values, as a :class:`~concepts.dsl.value.Value` instance.
            The second element is a tuple of (sum_weight, max_weight), where ``sum_weight`` is the sum of the weights (as a PyTorch tensor),
            and ``max_weight`` is the maximum of the weights (as a float).
        """
        if not isinstance(values[0], Value):
            return None, None
        function = self.get_function(values[0].dtype.typename)
        if function is None:
            return None, None
        return function(values, weights)

    def _expectation_set_tensors(self, sets: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], weights: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], log: bool = False) -> Tuple[torch.Tensor, Tuple[torch.Tensor, float]]:
        """A helper function to compute the expectation of a list of tensors representing sets.
        Typically, the tensor representation of a set is a one-hot vector, where each entry represents the probabbility of the corresponding element in the set.

        Args:
            sets: a list of tensors representing sets.
            weights: a list of corresponding weights.
            log: whether the probabilities in sets are in log-space.

        Returns:
            A tuple of (set, (sum_weight, max_weight)).
            The ``set`` is a tensor representing the expectation of the list of sets, as a PyTorch tensor.
            The second element is a tuple of (sum_weight, max_weight), where ``sum_weight`` is the sum of the weights (as a PyTorch tensor),
            and ``max_weight`` is the maximum of the weights (as a float).
        """
        sets_tensor = torch.stack(sets, dim=-1)
        weights_tensor = torch.stack(weights, dim=0)

        weights_sum = jactorch.logsumexp(weights_tensor)
        weights_max = weights_tensor.argmax().item()

        if log:
            weights_tensor = F.log_softmax(weights_tensor, dim=-1)
            weights_tensor = jactorch.add_dim_as_except(weights_tensor, sets_tensor, -1)
            output = jactorch.logsumexp(sets_tensor + weights_tensor, dim=-1)
        else:
            weights_tensor = F.softmax(weights_tensor, dim=-1)
            output = torch.matmul(sets_tensor, weights_tensor)

        return output, (weights_sum, weights_max)


def aggregate_weights(weights: Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]], log=True) -> Tuple[torch.Tensor, float]:
    """Sum up a list of parsing weights (probabilities).

    Args:
        weights: a list of weights. It can be a single tensor, or a list of tensors.
        log: whether the weights are in log space. When this is True, the output will be computed as ``logsumexp``.

    Returns:
        A tuple of (sum, max).
    """
    if torch.is_tensor(weights):
        weights_tensor = weights
    elif isinstance(weights, (tuple, list)):
        weights_tensor = torch.stack(weights, dim=0)
    else:
        weights_tensor = torch.tensor(weights)

    if log:
        return jactorch.logsumexp(weights_tensor), weights_tensor.argmax().item()
    else:
        return weights_tensor.sum(), weights_tensor.argmax().item()


@dataclass
class CKYEEExpectationConfig(object):
    """Configurations for CKYEE expectation computation."""
    compress_values: bool = True
    compress_0varbinding_functions: bool = True
    compress_1varbinding_functions: bool = True
    sample: bool = False


def _get_typename(x):
    return x.typename if x is not None else None


class NeuralCKYEE(NeuralCCG):
    """The neural CCG grammar with CKY-EE for chart parsing."""

    def __init__(
        self,
        domain: FunctionDomain,
        executor: FunctionDomainExecutor,
        candidate_lexicon_entries: Iterable[NeuralCCGLexiconSearchResult],
        expectation_function: CKYEEExpectationFunction,
        composition_system: Optional[CCGCompositionSystem] = None,
        *,
        expectation_config: Optional[CKYEEExpectationConfig] = None,
        joint_execution: bool = True,
        allow_none_lexicon: bool = False,
        reweight_meaning_lex: bool = False
    ):
        """Initialize the neural CCG grammar with CKY-EE.

        Args:
            domain: the function domain of the grammar.
            executor: the executor for the function domain.
            candidate_lexicon_entries:  a list of candidate lexicon entries.
            exepctation_function: the expectation function for CKY-EE. See :class:`CKYEEExpectationFunction`.
            composition_system: the composition system. If None, the default composition system will be used.
            expectation_config: the configuration for CKY-EE expectation computation.
            joint_execution: whether to execute the partial programs during CKY.
            allow_none_lexicon: whether to allow None lexicon.
            reweight_meaning_lex: whether to reweight the meaning lexicon entries. Specifically, if there are two parsings
                share the same set of lexicon entries (i.e., caused by ambiguities in combination), specifying this flag
                will reweight both of them to be 1 / (number of parsings that used this set of lexicon entries).
        """
        super().__init__(
            domain, executor, candidate_lexicon_entries, composition_system,
            joint_execution=joint_execution, allow_none_lexicon=allow_none_lexicon, reweight_meaning_lex=reweight_meaning_lex
        )
        self.expectation_function = expectation_function
        self.expectation_config = expectation_config if expectation_config is not None else CKYEEExpectationConfig()

    training: bool
    domain: FunctionDomain
    executor: FunctionDomainExecutor
    composition_system: CCGCompositionSystem
    candidate_lexicon_entries: Tuple['NeuralCCGLexiconSearchResult', ...]
    joint_execution: bool
    allow_none_lexicon: bool
    reweight_meaning_lex: bool

    def _unique_lexicon(self, nodes: List[NeuralCCGNode], syntax_only: bool) -> List[NeuralCCGNode]:
        if syntax_only:
            return self._unique_syntax_only(nodes)
        return self._collect_nodes_by_type(nodes)

    def _unique(self, nodes: List[NeuralCCGNode], syntax_only: bool) -> List[NeuralCCGNode]:
        nodes = super()._unique(nodes, syntax_only)
        if syntax_only:
            return nodes
        nodes = self._collect_nodes_by_type(nodes)
        return nodes

    def _reweight_parse_trees(self, parsings: List[NeuralCCGNode]) -> List[NeuralCCGNode]:
        raise NotImplementedError('Reweight pr[meaning | lex_i] is not supported for NeuralSoftCCG.')

    def _collect_nodes_by_type(self, nodes: List[NeuralCCGNode]) -> List[NeuralCCGNode]:
        output_nodes = nodes

        if self.expectation_config.compress_values:
            output_nodes = self._collect_values(output_nodes)
        if self.expectation_config.compress_0varbinding_functions:
            output_nodes = self._collect_0varbinding_functions(output_nodes)
        if self.expectation_config.compress_1varbinding_functions:
            output_nodes = self._collect_1varbinding_functions(output_nodes)
        if self.expectation_config.sample:
            output_nodes = self._collect_by_sample(output_nodes)

        return output_nodes

    @_profile
    def _collect_values(self, nodes):
        output_nodes = list()
        value_nodes = defaultdict(lambda: (list(), list(), list()))
        for node in nodes:
            if not isinstance(node.syntax, CCGCoordinationImmNode) and node.syntax.is_value:
                rec = value_nodes[(node.syntax.return_type.typename, _get_typename(node.syntax.lang_syntax_type))]
                rec[0].append(node)
                if self.joint_execution:
                    rec[1].append(node.execution_result)
                rec[2].append(node.weight)
            else:
                output_nodes.append(node)

        for typename, rec in value_nodes.items():
            output_nodes.extend(self._collect_values_once(*rec))
        return output_nodes

    @_profile
    def _collect_0varbinding_functions(self, nodes):
        output_nodes = list()

        function_nodes = defaultdict(lambda: (list(), list(), list()))
        for node in nodes:
            sem = node.semantics
            if not isinstance(node.syntax, CCGCoordinationImmNode) and sem.partial_type is not None and len(sem.partial_type) == 1 and isinstance(sem.partial_type[0], NeuralCCGSemanticsPartialTypeLex):
                rec = function_nodes[(sem.partial_type[0], _get_typename(node.syntax.lang_syntax_type))]
                rec[0].append(node)
                rec[2].append(node.weight)
            else:
                output_nodes.append(node)

        for typename, rec in function_nodes.items():
            output_nodes.extend(self._collect_0varbinding_functions_once(*rec))

        return output_nodes

    @_profile
    def _collect_1varbinding_functions(self, nodes):
        output_nodes = list()

        function_nodes = defaultdict(lambda: (list(), list(), list()))
        for node in nodes:
            sem = node.semantics
            if not isinstance(node.syntax, CCGCoordinationImmNode) and sem.partial_type is not None and len(sem.partial_type) == 2 and isinstance(sem.partial_type[0], NeuralCCGSemanticsPartialTypeLex):
                rec = function_nodes[(sem.partial_type[0], _get_typename(node.syntax.lang_syntax_type))]
                rec[0].append(node)
                if self.joint_execution:
                    rec[1].append(node.semantics.execution_buffer[1])
                rec[2].append(node.weight)
            else:
                output_nodes.append(node)

        for typename, rec in function_nodes.items():
            output_nodes.extend(self._collect_1varbinding_functions_once(*rec))

        return output_nodes

    @_profile
    def _collect_by_sample(self, nodes):
        output_nodes = list()
        value_nodes = defaultdict(lambda: (list(), list()))
        for node in nodes:
            rec = value_nodes[str(node.syntax)]
            rec[0].append(node)
            rec[1].append(node.weight)

        for typename, rec in value_nodes.items():
            output_nodes.extend(self._collect_by_sample_once(*rec))
        return output_nodes

    def _collect_values_once(self, nodes: List[NeuralCCGNode], all_results: List[Value], all_weights: List[torch.Tensor]) -> List[NeuralCCGNode]:
        if len(nodes) > 1:
            if self.joint_execution:
                compressed_node, weights = self.expectation_function.expectation(all_results, all_weights)
            else:
                compressed_node = None
                weights = aggregate_weights(all_weights, log=True)

            if self.joint_execution and compressed_node is None:
                return nodes
            else:
                sample_node = nodes[weights[1]]
                new_node = NeuralCCGNode(
                    composition_system=sample_node.composition_system,
                    syntax=sample_node.syntax,
                    semantics=NeuralCCGSemantics(
                        sample_node.semantics.value,
                        execution_buffer=[compressed_node] if compressed_node is not None else None,
                        partial_type=sample_node.semantics.partial_type,
                        nr_execution_steps=self.count_nr_execution_steps(nodes)
                    ),
                    composition_type=sample_node.composition_type,
                    lexicon=sample_node.lexicon, lhs=sample_node.lhs, rhs=sample_node.rhs,
                    composition_str=sample_node.composition_str,
                    weight=weights[0]
                )
                new_node.set_used_lexicon_entries(self.gen_valid_lexicons(nodes))
                return [new_node]
        else:
            return nodes

    def _collect_0varbinding_functions_once(self, nodes: List[NeuralCCGNode], all_results: List[Value], all_weights: List[torch.Tensor]) -> List[NeuralCCGNode]:
        if len(nodes) > 1:
            weights = aggregate_weights(all_weights, log=True)
            sample_node = nodes[0]
            new_node = NeuralCCGNode(
                composition_system=sample_node.composition_system,
                syntax=sample_node.syntax,
                semantics=sample_node.semantics,
                composition_type=sample_node.composition_type,
                lexicon=sample_node.lexicon, lhs=sample_node.lhs, rhs=sample_node.rhs,
                composition_str=sample_node.composition_str,
                weight=weights[0]
            )
            new_node.set_used_lexicon_entries(self.gen_valid_lexicons(nodes))
            return [new_node]
        else:
            return nodes

    def _collect_1varbinding_functions_once(self, nodes: List[NeuralCCGNode], all_results: List[Value], all_weights: List[torch.Tensor]) -> List[NeuralCCGNode]:
        if len(nodes) > 1:
            if self.joint_execution:
                compressed_node, weights = self.expectation_function.expectation(all_results, all_weights)
            else:
                compressed_node = None
                weights = aggregate_weights(all_weights, log=True)

            if self.joint_execution and compressed_node is None:
                return nodes
            else:
                sample_node = nodes[weights[1]]
                new_node = NeuralCCGNode(
                    composition_system=sample_node.composition_system,
                    syntax=sample_node.syntax,
                    semantics=NeuralCCGSemantics(
                        sample_node.semantics.value,
                        execution_buffer=[
                            sample_node.semantics.execution_buffer[0],
                            compressed_node
                        ] if compressed_node is not None else None,
                        partial_type=sample_node.semantics.partial_type,
                        nr_execution_steps=self.count_nr_execution_steps(nodes)
                    ),
                    composition_type=sample_node.composition_type,
                    lexicon=sample_node.lexicon, lhs=sample_node.lhs, rhs=sample_node.rhs,
                    composition_str=sample_node.composition_str,
                    weight=weights[0]
                )
                new_node.set_used_lexicon_entries(self.gen_valid_lexicons(nodes))
                return [new_node]
        else:
            return nodes

    def _collect_by_sample_once(self, nodes: List[NeuralCCGNode], all_weights: List[torch.Tensor]) -> List[NeuralCCGNode]:
        if len(nodes) > 1:
            weights_tensor = torch.stack(all_weights, dim=0).detach()
            index = jacrandom.choice(weights_tensor.shape[0], p=jactorch.as_numpy(F.softmax(weights_tensor)))
            weights_tensor.data[index] = -1e9
            total_weights = jactorch.logaddexp(all_weights[index], jactorch.logsumexp(weights_tensor))

            sample_node = nodes[index]
            new_node = NeuralCCGNode(
                composition_system=sample_node.composition_system,
                syntax=sample_node.syntax,
                semantics=NeuralCCGSemantics(
                    sample_node.semantics.value,
                    execution_buffer=sample_node.semantics.execution_buffer,
                    partial_type=sample_node.semantics.partial_type,
                    nr_execution_steps=self.count_nr_execution_steps(nodes)
                ),
                composition_type=sample_node.composition_type,
                lexicon=sample_node.lexicon, lhs=sample_node.lhs, rhs=sample_node.rhs,
                composition_str=sample_node.composition_str,
                weight=total_weights
            )
            new_node.set_used_lexicon_entries(self.gen_valid_lexicons(nodes))
            return [new_node]
        else:
            return nodes
