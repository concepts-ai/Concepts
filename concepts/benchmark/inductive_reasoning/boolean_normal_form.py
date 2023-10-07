#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : boolean_normal_form.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/18/2018
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import collections
import jacinle.random as random

__all__ = ['CNF', 'DNF', 'random_generate_cnf']


class NormalForm(object):
    merge1 = None
    merge2 = None
    init1 = None
    init2 = None
    split1 = None
    split2 = None

    def __init__(self, nr_variables, exprs, varnames=None):
        self.nr_variables = nr_variables
        self.exprs = exprs
        self.idx2name = varnames

    def __call__(self, assigns=None, **kwargs):
        if assigns is None:
            return self.eval(kwargs)
        return self.eval(assigns)

    def eval(self, assigns):
        assert self.nr_variables == len(assigns)

        if isinstance(assigns, collections.Mapping):
            assert self.idx2name is not None
            assigns = [assigns[self.idx2name[k]] for k in range(self.nr_variables)]
        else:
            assert isinstance(assigns, collections.Sequence)
        assigns = list(map(bool, assigns))

        v1 = type(self).init1
        for e1, neg1 in self.exprs:
            v2 = type(self).init2
            for e2, neg2 in e1:
                if neg2:
                    v2 = type(self).merge2(v2, not assigns[e2])
                else:
                    v2 = type(self).merge2(v2, assigns[e2])
            if neg1:
                v1 = type(self).merge1(v1, not v2)
            else:
                v1 = type(self).merge1(v1, v2)
        return v1

    @classmethod
    def from_string(cls, expr):
        expr = expr.strip().replace(' ', '')  # canonize
        closures = []
        variables = set()

        ands = expr.split(cls.split1)
        for a in ands:
            this_closure = []
            if a[0] == '!':
                a = a[1:]
                nega = True
            else:
                nega = False

            assert a[0] == '(' and a[-1] == ')'
            a = a[1:-1]
            ors = a.split(cls.split2)
            for o in ors:
                if o.startswith('!'):
                    o = o[1:]
                    nego = True
                else:
                    nego = False
                this_closure.append([o, nego])
                variables.add(o)
            closures.append([this_closure, nega])

        variables = sorted(variables)
        var2idx = {v: i for i, v in enumerate(variables)}
        nr_variables = len(variables)
        for a, _ in closures:
            for o in a:
                o[0] = var2idx[o[0]]
        return cls(nr_variables, closures, varnames=variables)

    def to_string(self):
        if self.idx2name is not None:
            idx2name = self.idx2name
        else:
            assert self.nr_variables <= 26
            idx2name = [chr(97 + i) for i in range(26)]

        s1 = ''
        for e1, neg1 in self.exprs:
            if len(s1) != 0:
                s1 += type(self).split1
            s2 = ''
            for e2, neg2 in e1:
                if len(s2) != 0:
                    s2 += type(self).split2
                if neg2:
                    s2 += '!' + idx2name[e2]
                else:
                    s2 += idx2name[e2]
            if neg1:
                s2 = '!(' + s2 + ')'
            else:
                s2 = '(' + s2 + ')'
            s1 += s2
        return s1


class CNF(NormalForm):
    init1 = True
    init2 = False
    split1 = '&'
    split2 = '|'
    merge1 = lambda x, y: x and y
    merge2 = lambda x, y: x or y


class DNF(NormalForm):
    init1 = False
    init2 = True
    split1 = '|'
    split2 = '&'
    merge1 = lambda x, y: x or y
    merge2 = lambda x, y: x and y


def _canonize_max(m):
    if type(m) is int:
        m = [max(m // 2, 1), max(m, 1)]
    else:
        assert m[0] > 0 and m[1] > 0
    return m


def random_generate_cnf(nr_variables, max_ands, max_ors):
    max_ands = _canonize_max(max_ands)
    max_ors = _canonize_max(max_ors)

    closures = []
    nr_ands = random.randint(max_ands[0], max_ands[1] + 1)
    for i in range(nr_ands):
        nr_ors = random.randint(max_ors[0], max_ors[1] + 1)
        vars = random.randint(0, nr_variables, size=nr_ors)
        nots = random.randint(0, 2, size=nr_ors)
        this_closures = [(v, n) for v, n in zip(vars, nots)]
        closures.append((this_closures, random.randint(0, 2)))
    return CNF(nr_variables, closures)
