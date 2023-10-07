#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : boolean_normal_form_dataset.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/18/2018
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import numpy as np
from torch.utils.data.dataset import Dataset


class NormalFormDataset(Dataset):
    """Learning a general normal form."""

    def __init__(self, normal_form, include=None, exclude=None):
        assert normal_form.nr_variables <= 16

        self.normal_form = normal_form
        self.include = include
        self.exclude = exclude

        if self.include is not None:
            self.include = list(self.include)
        elif self.exclude is not None:
            self.exclude = set(self.exclude)
            self.include = [i for i in range(1 << self.normal_form.nr_variables) if i not in self.exclude]
        else:
            self.include = list(range(1 << self.normal_form.nr_variables))

    def __getitem__(self, item):
        assigns = _binary_decomposition(self.include[item], self.normal_form.nr_variables)
        result = self.normal_form(assigns)
        return dict(input=np.array(assigns, dtype=np.float32), label=float(result))

    def __len__(self):
        return len(self.include)


class TruthTableDataset(Dataset):
    """Learning a truth table."""
    def __init__(self, nr_variables, table):
        assert nr_variables <= 16
        assert 1 << nr_variables == len(table)
        self.nr_variables = nr_variables
        self.table = table

    def __getitem__(self, item):
        assigns = _binary_decomposition(item, self.nr_variables)
        result = self.table[item]
        return dict(input=np.array(assigns, dtype=np.float32), label=float(result))

    def __len__(self):
        return len(self.table)


class ParityDataset(Dataset):
    """Learning the parity function."""
    def __init__(self, nr_variables):
        assert nr_variables <= 16
        self.nr_variables = nr_variables

    def __getitem__(self, item):
        assigns = _binary_decomposition(item, self.nr_variables)
        result = sum(assigns) % 2
        return dict(input=np.array(assigns, dtype=np.float32), label=float(result))

    def __len__(self):
        return 1 << self.nr_variables


def _binary_decomposition(v, n):
    return [bool(v & (1 << i)) for i in range(n)]
