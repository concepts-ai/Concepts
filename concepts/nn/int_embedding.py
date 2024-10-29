#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : int_embedding.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/12/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['IntEmbedding', 'FloatEmbedding', 'FallThroughEmbedding', 'ConcatIntEmbedding']


class IntEmbedding(nn.Module):
    def __init__(self, embedding_dim, input_dim=3, value_range=(0, 16), attach_input=False, concat_input=False):
        super().__init__()
        if isinstance(value_range, tuple):
            assert len(value_range) == 2
        elif isinstance(value_range, int):
            value_range = (0, value_range)
        else:
            raise TypeError('Value range should be either tuple or int, got {}.'.format(type(value_range)))

        self.input_dim = input_dim
        self.value_range = value_range
        self.embedding_dim = embedding_dim + int(concat_input) * input_dim
        self.embedding = nn.Embedding((self.value_range[1] - self.value_range[0]) ** self.input_dim, embedding_dim)
        self.attach_input = attach_input
        self.concat_input = concat_input

    training: bool

    input_dim: int
    """The input dimension of the embedding layer."""

    value_range: tuple[int, int]
    """The value range of the input tensor."""

    embedding_dim: int
    """The output dimension of the embedding layer."""

    attach_input: bool
    """Whether to attach the input tensor to the output tensor. The difference between `attach_input` and `concat_input` is that `attach_input` will overwrite
    the last `input_dim` dimensions of the output tensor, while `concat_input` will concatenate the input tensor to the output tensor."""

    concat_input: bool
    """Whether to concatenate the input tensor to the output tensor."""

    def forward(self, input):
        input = input - self.value_range[0]  # shift the input first.

        if self.input_dim == 1:
            index = input[..., 0]
        elif self.input_dim == 2:
            x, y = input.split(1, dim=-1)
            index = (x * 16 + y).squeeze(-1)
        elif self.input_dim == 3:
            x, y, z = input.split(1, dim=-1)
            index = ((x * 16 + y) * 16 + z).squeeze(-1)
        elif self.input_dim == 4:
            x, y, z, t = input.split(1, dim=-1)
            index = ((((x * 16) + y) * 16 + z) * 16 + t).squeeze(-1)
        else:
            index = input[..., 0]
            for i in range(1, self.input_dim):
                index = index * 16 + input[..., i]

        rv = self.embedding(index.long())
        if self.attach_input:
            rv[..., :self.input_dim] = input
        if self.concat_input:
            rv = torch.cat((rv, input), dim=-1)
        return rv


class FloatEmbedding(nn.Module):
    def __init__(self, embedding_dim, input_dim=2):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.mapping = nn.Linear(input_dim, embedding_dim)

    training: bool

    input_dim: int
    """The input dimension of the embedding layer."""

    embedding_dim: int
    """The output dimension of the embedding layer."""

    def forward(self, input):
        return self.mapping(input)


class PadEmbedding(nn.Module):
    def __init__(self, embedding_dim, input_dim=2):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        assert self.input_dim < self.embedding_dim

    training: bool

    input_dim: int
    """The input dimension of the embedding layer."""

    embedding_dim: int
    """The output dimension of the embedding layer."""

    def forward(self, input):
        return F.pad(input, (0, self.embedding_dim - input.shape[-1]))


class FallThroughEmbedding(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = input_dim

    training: bool

    input_dim: int
    """The input dimension of the embedding layer."""

    embedding_dim: int
    """The output dimension of the embedding layer. It is the same as the input dimension for this module."""

    def forward(self, input):
        if self.input_dim is not None:
            assert input.shape[-1] == self.input_dim
        return input


class ConcatIntEmbedding(nn.Module):
    """
    The dimension mapping is a dictionary from int to int. For example, if the mapping is:

    .. code-block:: python

        emb = ConcatIntEmbedding({
            3: IntEmbedding(64),
            2: IntEmbedding(32, value_range=(-1, 15)),
            1: IntEmbedding(32, value_range=(0, 4))
        })

        print(emb.input_dim)   # 6
        print(emb.output_dim)  # 128 = 64 + 32 + 32

    This mapping indicates that the input tensor has dimension 3+2+1=6.
    - The first 3 dimensions will be embedded to a 64-dim latent vector.
    - The next 2 dimensions will be embedded to a 32-dim latent vector.
    - The last dimension will be embedded to a 32-dim latent vector.

    Thus, the total output dimension is 64+32+32 = 128.
    """
    def __init__(self, dimension_mapping: dict[int, nn.Module]):
        super().__init__()
        self.dimension_mapping = dimension_mapping
        self.embeddings = nn.ModuleList()

        self.input_dim = 0
        self.output_dim = 0
        for k, v in dimension_mapping.items():
            self.input_dim += k
            self.output_dim += v.embedding_dim
            self.embeddings.append(v)

    training: bool

    input_dim: int
    """The input dimension of the embedding layer."""

    output_dim: int
    """The output dimension of the embedding layer."""

    dimension_mapping: dict[int, nn.Module]
    """The mapping from input dimension to embedding layer."""

    def forward(self, input):
        dims = list(self.dimension_mapping)
        input_splits = torch.split(input, dims, dim=-1)
        outputs = list()
        for v, e in zip(input_splits, self.embeddings):
            outputs.append(e(v))
        return torch.cat(outputs, dim=-1)
