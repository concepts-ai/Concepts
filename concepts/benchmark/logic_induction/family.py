#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : family.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
# Date   : 05/07/2018
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import numpy as np
import jacinle.random as random
from torch.utils.data.dataset import Dataset

__all__ = ['FamilyTreeDataset', 'Family', 'random_generate_family']


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


class Family(object):
    """A data structure that stores the relationship between N people in a family."""

    def __init__(self, nr_people: int, relations: np.ndarray):
        """Initialize a family with relations.

        Args:
            nr_people: number of people in the family.
            relations: a 3D array of shape (nr_people, nr_people, 6), where
                relations[i, j, 0] = 1 if j is the husband of i, 0 otherwise.
                relations[i, j, 1] = 1 if j is the wife of i, 0 otherwise.
                relations[i, j, 2] = 1 if j is the father of i, 0 otherwise.
                relations[i, j, 3] = 1 if j is the mother of i, 0 otherwise.
                relations[i, j, 4] = 1 if j is the son of i, 0 otherwise.
                relations[i, j, 5] = 1 if j is the daughter of i, 0 otherwise.
        """
        self._n = nr_people
        self._relations = relations

    @property
    def father(self) -> np.ndarray:
        return self._relations[:, :, 2]

    @property
    def mother(self) -> np.ndarray:
        return self._relations[:, :, 3]

    @property
    def son(self) -> np.ndarray:
        return self._relations[:, :, 4]

    @property
    def daughter(self) -> np.ndarray:
        return self._relations[:, :, 5]

    def has_father(self) -> np.ndarray:
        return self.father.max(axis=1)

    def has_daughter(self) -> np.ndarray:
        return self.daughter.max(axis=1)

    def has_sister(self) -> np.ndarray:
        return _clip_mul(self.father, self.daughter).max(axis=1)

    def get_parents(self) -> np.ndarray:
        return np.clip(self.father + self.mother, 0, 1)

    def get_grandfather(self) -> np.ndarray:
        return _clip_mul(self.get_parents(), self.father)

    def get_grandmother(self) -> np.ndarray:
        return _clip_mul(self.get_parents(), self.mother)

    def get_grandparents(self) -> np.ndarray:
        parents = self.get_parents()
        return _clip_mul(parents, parents)

    def get_uncle(self) -> np.ndarray:
        return _clip_mul(self.get_grandparents(), self.son)

    def get_maternal_great_uncle(self) -> np.ndarray:
        return _clip_mul(_clip_mul(self.get_grandmother(), self.mother), self.son)


def random_generate_family(n, p_marriage=0.8, verbose=False) -> Family:
    assert n > 0
    ids = list(random.permutation(n))

    single_m = []
    single_w = []
    couples = [None]
    rel = np.zeros((n, n, 6))  # husband, wife, father, mother, son, daughter
    fathers = [None for i in range(n)]
    mothers = [None for i in range(n)]

    def add_couple(man, woman):
        couples.append((man, woman))
        rel[woman, man, 0] = 1  # husband
        rel[man, woman, 1] = 1  # wife
        if verbose:
            print('couple', man, woman)

    def add_child(parents, child, gender):
        father, mother = parents
        fathers[child] = father
        mothers[child] = mother
        rel[child, father, 2] = 1  # father
        rel[child, mother, 3] = 1  # mother
        if gender == 0:  # son
            rel[father, child, 4] = 1
            rel[mother, child, 4] = 1
        else:  # daughter
            rel[father, child, 5] = 1
            rel[mother, child, 5] = 1
        if verbose:
            print('child', father, mother, child, gender)

    def check_relations(man, woman):
        if fathers[man] is None or fathers[woman] is None:
            return True
        if fathers[man] == fathers[woman]:
            return False

        def same_parent(x, y):
            return fathers[x] is not None and fathers[y] is not None and fathers[x] == fathers[y]

        for x in [fathers[man], mothers[man]]:
            for y in [fathers[woman], mothers[woman]]:
                if same_parent(man, y) or same_parent(woman, x) or same_parent(x, y):
                    return False
        return True

    while len(ids) > 0:
        x = ids.pop()
        gender = random.randint(2)
        parents = random.choice_list(couples)
        if gender == 0:
            single_m.append(x)
        else:
            single_w.append(x)
        if parents is not None:
            add_child(parents, x, gender)

        if random.rand() < p_marriage and len(single_m) > 0 and len(single_w) > 0:
            mi = random.randint(len(single_m))
            wi = random.randint(len(single_w))
            man = single_m[mi]
            woman = single_w[wi]
            if check_relations(man, woman):
                add_couple(man, woman)
                del single_m[mi]
                del single_w[wi]

    return Family(n, rel)


def _clip_mul(x, y):
    return np.clip(np.matmul(x, y), 0, 1)
