#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : graph_dataset.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
# Date   : 05/07/2018
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import numpy as np

from torch.utils.data.dataset import Dataset
from torchvision import datasets

import jacinle.random as random

from concepts.benchmark.algorithm_env.graph import random_generate_graph, random_generate_graph_dnc
from concepts.benchmark.logic_induction.family import random_generate_family

__all__ = ['GraphOutDegreeDataset', 'GraphConnectivityDataset', 'GraphAdjacentDataset', 'FamilyTreeDataset']


class GraphDatasetBase(Dataset):
    def __init__(self, nr_nodes, p, epoch_size, directed=False, gen_method='dnc'):
        if type(nr_nodes) is int:
            self.nr_nodes = (max(nr_nodes // 2, 1), nr_nodes)
        else:
            self.nr_nodes = tuple(nr_nodes)
        self.p = p
        self.epoch_size = epoch_size
        self.directed = directed
        self.gen_method = gen_method

    def _gen_graph(self, item):
        nr_nodes = item % (self.nr_nodes[1] - self.nr_nodes[0] + 1) + self.nr_nodes[0]
        if type(self.p) is float:
            p = self.p
        else:
            p = self.p[0] + random.rand() * (self.p[1] - self.p[0])
        gen_graph = random_generate_graph_dnc if self.gen_method == 'dnc' else random_generate_graph
        return gen_graph(nr_nodes, p, directed=self.directed)

    def __len__(self):
        return self.epoch_size


class GraphOutDegreeDataset(GraphDatasetBase):
    def __init__(self, nr_nodes, p, epoch_size, degree=2, directed=False, gen_method='dnc'):
        super().__init__(nr_nodes, p, epoch_size, directed, gen_method)
        self.degree = degree

    def __getitem__(self, item):
        graph = self._gen_graph(item)
        return dict(
            n=graph._nr_nodes,
            relations=np.expand_dims(graph.get_edges(), axis=-1),
            target=(graph.get_out_degree() == self.degree).astype('float'),
        )


class GraphConnectivityDataset(GraphDatasetBase):
    def __init__(self, nr_nodes, p, epoch_size, dist_limit=None, directed=False, gen_method='dnc'):
        super().__init__(nr_nodes, p, epoch_size, directed, gen_method)
        self.dist_limit = dist_limit

    def __getitem__(self, item):
        graph = self._gen_graph(item)
        return dict(
            n=graph._nr_nodes,
            relations=np.expand_dims(graph.get_edges(), axis=-1),
            # relations=graph.get_relations(),
            target=graph.get_connectivity(self.dist_limit, exclude_self=True),
        )


class GraphAdjacentDataset(GraphDatasetBase):
    def __init__(self, nr_nodes, p, epoch_size, nr_colors, directed=False, gen_method='dnc',
                 is_mnist_colors=False, is_train=True):

        super().__init__(nr_nodes, p, epoch_size, directed, gen_method)
        self._nr_colors = nr_colors
        self._mnist_colors = is_mnist_colors
        if is_mnist_colors:
            assert nr_colors == 10
            transform = None
            self.mnist = datasets.MNIST('../data', train=is_train, download=True, transform=transform)

    def __getitem__(self, item):
        graph = self._gen_graph(item)
        n = graph._nr_nodes
        if self._mnist_colors:
            m = self.mnist.__len__()
            digits = []
            colors = []
            for i in range(n):
                x = random.randint(m)
                digit, color = self.mnist.__getitem__(x)
                digits.append(np.array(digit)[np.newaxis])
                colors.append(color)
            digits, colors = np.array(digits), np.array(colors)
        else:
            colors = random.randint(self._nr_colors, size=n)
        states = np.zeros((n, self._nr_colors))
        adjacent = np.zeros((n, self._nr_colors))
        for i in range(n):
            states[i, colors[i]] = 1
            adjacent[i, colors[i]] = 1
            for j in range(n):
                if graph.has_edge(i, j):
                    adjacent[i, colors[j]] = 1
        if self._mnist_colors:
            states = digits
        return dict(
            n=n,
            relations=np.expand_dims(graph.get_edges(), axis=-1),
            states=states,
            colors=colors,
            target=adjacent,
            # connectivity=graph.get_connectivity(self.dist_limit, exclude_self=True),
        )


class FamilyTreeDataset(Dataset):
    def __init__(self, nr_people, epoch_size, task, p_marriage=0.8, balance_sample=False):
        super().__init__()
        if type(nr_people) is int:
            self.nr_people = (max(nr_people // 2, 1), nr_people)
        else:
            self.nr_people = tuple(nr_people)
        self.epoch_size = epoch_size
        self.task = task
        self.p_marriage = p_marriage
        self.balance_sample = balance_sample
        self.data = []

        assert task in ['has-father', 'has-daughter', 'has-sister', 'parents', 'grandparents', 'uncle', 'maternal-great-uncle']

    def _gen_family(self, item):
        nr_people = item % (self.nr_people[1] - self.nr_people[0] + 1) + self.nr_people[0]
        return random_generate_family(nr_people, self.p_marriage)

    def __getitem__(self, item):
        while len(self.data) == 0:
            family = self._gen_family(item)
            relations = family._relations[:, :, 2:]
            if self.task == 'has-father':
                target = family.has_father()
            elif self.task == 'has-daughter':
                target = family.has_daughter()
            elif self.task == 'has-sister':
                target = family.has_sister()
            elif self.task == 'parents':
                target = family.get_parents()
            elif self.task == 'grandparents':
                target = family.get_grandparents()
            elif self.task == 'uncle':
                target = family.get_uncle()
            elif self.task == 'maternal-great-uncle':
                target = family.get_maternal_great_uncle()
            else:
                assert False, "{} is not supported.".format(self.task)

            if not self.balance_sample:
                return dict(n=family._n, relations=relations, target=target)

            def get_position(x):
                return list(np.vstack(np.where(x)).T)

            def append_data(pos, target):
                states = np.zeros((family._n, 2))
                states[pos[0], 0] = states[pos[1], 1] = 1
                self.data.append(dict(n=family._n, relations=relations, states=states, target=target))

            positive = get_position(target == 1)
            if len(positive) == 0:
                continue
            negative = get_position(target == 0)
            np.random.shuffle(negative)
            negative = negative[:len(positive)]
            for i in positive:
                append_data(i, 1)
            for i in negative:
                append_data(i, 0)

        return self.data.pop()

    def __len__(self):
        return self.epoch_size
