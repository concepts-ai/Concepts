#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : tensorized_syntax.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 06/21/2021
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import heapq
import collections
from typing import Optional, Union, Iterable, Tuple, List, Dict, Callable

import torch
import torch.nn as nn

import jactorch
from jacinle.utils.cache import cached_property

from concepts.language.ccg.composition import CCGCompositionType, CCGCompositionContext, CCGCompositionError, CCGCompositionSystem
from concepts.language.ccg.syntax import CCGSyntaxType
from concepts.language.ccg.grammar import Lexicon, CCGNode

__all__ = ['TensorizedSyntaxOnlyCCG']


class TensorizedSyntaxOnlyCCG(nn.Module):
    """Data structures and parsing implementation for a syntax-only CCG grammar. It uses a tensor-based representation for the chart, therefore is highly parallelizable."""

    def __init__(self, candidate_syntax_types: Iterable[CCGSyntaxType], composition_system: Optional[CCGCompositionSystem] = None):
        """Initialize the CCG grammar.

        Args:
            candidate_syntax_types: the candidate syntax types.
            composition_system: the composition system. If None, the default composition system will be used.
        """
        super().__init__()

        self.candidate_syntax_types = tuple(candidate_syntax_types)
        self.composition_system = composition_system
        if self.composition_system is None:
            self.composition_system = CCGCompositionSystem.make_default()

    training: bool

    candidate_syntax_types: Tuple[CCGSyntaxType, ...]
    """A list of candidate syntax types."""

    composition_system: CCGCompositionSystem
    """The composition system."""

    @cached_property
    def syntax_name2index(self) -> Dict[str, int]:
        """A mapping from syntax type name to the corresponding index."""
        return {
            (s.typename if isinstance(s, CCGSyntaxType) else s): i
            for i, s in enumerate(self.candidate_syntax_types)
        }

    def parse(
        self,
        words: torch.Tensor,  # [batch_size, max_seqlen]
        words_length: torch.Tensor,  # [batch_size]
        distribution_over_syntax_types: torch.Tensor,  # [batch_size, max_seqlen, nr_types],
        transition_matrix: torch.Tensor,  # [A: nr_types, B: nr_types, C: nr_types]  AB -> C
        allowed_root_type_indices: Union[int, List[int]]
    ) -> torch.Tensor:
        """Parse a batch of sentences.

        Args:
            words: the word indices, should have shape ``[batch_size, max_seqlen]``.
            words_length: the length of each sentence, should have shape ``[batch_size]``.
            distribution_over_syntax_types: the distribution over syntax types for each word, should have shape ``[batch_size, max_seqlen, nr_types]``.
            transition_matrix: the transition matrix, should have shape [A: nr_types, B: nr_types, C: nr_types]. Basically, each entry [A, B, C] means the probability of composing a node of type A and a node of type B to a node of type C.
            allowed_root_type_indices: the indices of the syntax types that are allowed to be the root of the tree. It can be either a single integer or a list of integers.

        Returns:
            The root nodes of the trees, with shape ``[batch_size]`` (when ``allowed_root_type_indices`` is a single integer) or ``[batch_size, nr_allowed_root_types]`` (when ``allowed_root_type_indices`` is a list of integers).
            Each entry i can be interpreted as the log-probability of the root node being syntax type ``allowed_root_type_indices[i]``.
        """
        batch_size, length = words.size()

        dp = [[None for _ in range(length + 1)] for _ in range(length)]

        for i in range(length):
            dp[i][i + 1] = distribution_over_syntax_types[:, i]

        for span_length in range(2, length + 1):
            for i in range(0, length + 1 - span_length):
                j = i + span_length

                scores = list()
                for k in range(i + 1, j):
                    scores.append(self.merge(dp[i][k], dp[k][j], transition_matrix))

                scores = torch.stack(scores, dim=-1)  # [batch_size, nr_types, K]
                scores = jactorch.logsumexp(scores, dim=-1, keepdim=False)
                dp[i][j] = scores

        ret = list()
        for i in range(words_length.size(0)):
            ret.append(dp[0][words_length[i].item()][i])
        ret = torch.stack(ret, dim=0)

        if isinstance(allowed_root_type_indices, int):
            ret = ret[:, allowed_root_type_indices]
        else:
            ret = jactorch.logsumexp(ret[:, allowed_root_type_indices], dim=-1)

        return ret

    @jactorch.no_grad_func
    def parse_beamsearch(
        self,
        words: torch.Tensor,
        words_length: torch.Tensor,  # [batch_size]
        words_tokenized: List[List[str]],
        distribution_over_syntax_types: torch.Tensor,  # [batch_size, max_seqlen, nr_types],
        transition_matrix: torch.Tensor,  # [A: nr_types, B: nr_types, C: nr_types]  AB -> C
        allowed_root_types: List[str],
        beam_size: Optional[int] = 3
    ) -> List[List[CCGNode]]:
        """Parse a batch of sentences using beam search. This function will return candidate parsing trees (instead of just the probability
        distribution of the root note). Therefore, this function is significantly slower than :meth:`parse` and should not be used for training.

        Args:
            words: the word indices, should have shape ``[batch_size, max_seqlen]``.
            words_length: the length of each sentence, should have shape ``[batch_size]``.
            words_tokenized: the tokenized words.
            distribution_over_syntax_types: the distribution over syntax types for each word, should have shape ``[batch_size, max_seqlen, nr_types]``.
            transition_matrix: the transition matrix, should have shape [A: nr_types, B: nr_types, C: nr_types]. Basically, each entry [A, B, C] means the probability of composing a node of type A and a node of type B to a node of type C.
            allowed_root_types: the syntax types that are allowed to be the root of the tree.
            beam_size: the beam size.

        Returns:
            A list of candidate parsing trees for each sentence. Each entry is a list of trees, where each tree is a :class:`CCGNode` object.
        """
        assert not self.training

        batch_size = words.size(0)
        batch_ret = list()

        with CCGCompositionContext(semantics=False).as_default():
            for batch_index in range(batch_size):
                length = words_length[batch_index].item()
                dp = [[None for _ in range(length + 1)] for _ in range(length)]

                for i in range(length):
                    dp[i][i + 1] = list()
                    for j in range(1, distribution_over_syntax_types.size(2)):  # skip <PAD>
                        weight = distribution_over_syntax_types[batch_index, i, j]
                        dp[i][i + 1].append(CCGNode(
                            self.composition_system,
                            syntax=self.candidate_syntax_types[j],
                            semantics=None,
                            composition_type=CCGCompositionType.LEXICON,
                            lexicon=Lexicon(
                                self.candidate_syntax_types[j],
                                None,
                                weight=weight,
                                extra=(words_tokenized[batch_index][i], j)
                            ),
                            weight=weight
                        ))

                    dp[i][i + 1] = self._beamsearch_nlargest(beam_size, dp[i][i + 1], key_func=lambda node: node.weight.item())

                for span_length in range(2, length + 1):
                    for i in range(0, length + 1 - span_length):
                        j = i + span_length

                        dp[i][j] = list()
                        for k in range(i + 1, j):
                            dp[i][j].extend(self.merge_beamsearch(dp[i][k], dp[k][j], transition_matrix))
                        dp[i][j] = self._beamsearch_nlargest(beam_size, dp[i][j], key_func=lambda node: node.weight.item())

                ret = dp[0][length]
                ret = list(filter(lambda node: node.syntax.typename in allowed_root_types, ret))
                batch_ret.append(ret)

        return batch_ret

    def _beamsearch_nlargest(self, beam_size: int, input_list: List[CCGNode], key_func: Callable[[CCGNode], float]) -> List[CCGNode]:
        """Return the nlargest elements in the list, using the key function to determine the order."""
        input_by_type = collections.defaultdict(list)
        for node in input_list:
            input_by_type[node.syntax.typename].append(node)
        for key, value in input_by_type.items():
            input_by_type[key] = heapq.nlargest(beam_size, value, key_func)
        return [v for value in input_by_type.values() for v in value]

    def merge(self, A: torch.Tensor, B: torch.Tensor, transition_matrix: torch.Tensor):
        """The underlying merge function for chart parsing.

        Args:
            A: the distribution over syntax types for the left child, should have shape ``[batch_size, nr_types]``.
            B: the distribution over syntax types for the right child, should have shape ``[batch_size, nr_types]``.
            transition_matrix: the transition matrix, should have shape [A: nr_types, B: nr_types, C: nr_types]. Basically, each entry [A, B, C] means the probability of composing a node of type A and a node of type B to a node of type C.

        Returns:
            The distribution over syntax types for the merged node, should have shape ``[batch_size, nr_types]``.
        """
        # A : [batch_size, nr_types]
        # B : [batch_size, nr_types]
        # transition_matrix: [A: nr_types, B: nr_types, C: nr_types]

        # return _cky_merge(A, B, transition_matrix)

        T = transition_matrix
        T = jactorch.logmatmulexp(A, T)
        T = jactorch.batch_logmatmulexp(B.unsqueeze(1), T).squeeze(1)
        return T

    def merge_beamsearch(self, a_list: List[CCGNode], b_list: List[CCGNode], transition_matrix: torch.Tensor) -> List[CCGNode]:
        """The underlying merge function for chart parsing with beam search.

        Args:
            a_list: the list of nodes for the left child.
            b_list: the list of nodes for the right child.
            transition_matrix: the transition matrix, should have shape [A: nr_types, B: nr_types, C: nr_types]. Basically, each entry [A, B, C] means the probability of composing a node of type A and a node of type B to a node of type C.

        Returns:
            The list of merged nodes.
        """
        for x in a_list:
            for y in b_list:
                try:
                    _, z = x.compose(y)
                except CCGCompositionError:
                    continue

                z.weight = x.weight + y.weight + transition_matrix[
                    self.syntax_name2index[x.syntax.typename],
                    self.syntax_name2index[y.syntax.typename],
                    self.syntax_name2index[z.syntax.typename]
                ]

                yield z


@torch.jit.script
def _cky_logsumexp(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    tensor_max = tensor.max(dim=dim, keepdim=True)[0]
    tensor = tensor - tensor_max
    tensor_max = tensor_max.squeeze(dim)
    tensor = tensor.exp().sum(dim=dim).log() + tensor_max
    return tensor


@torch.jit.script
def _cky_merge(A: torch.Tensor, B: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    batch_size, nr_types = A.size()
    C = torch.zeros_like(A)
    for bs in range(batch_size):
        c = A[bs].unsqueeze(-1).unsqueeze(-1) + B[bs].unsqueeze(-1).unsqueeze(0) + T
        c = c.reshape(nr_types * nr_types, nr_types)
        c = _cky_logsumexp(c)
        C[bs, :] = c
    return C

