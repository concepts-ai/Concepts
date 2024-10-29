#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : crow_state.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 11/09/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Optional, Union, Iterator, Sequence, Tuple, List

import jacinle
from concepts.dsl.expression import ValueOutputExpression
from concepts.dm.pdsketch.regression_rule import AchieveExpression, BindExpression, RegressionRuleApplier

__all__ = ['TotallyOrderedPlan', 'PartiallyOrderedPlan', 'RegressionNode']


class TotallyOrderedPlan(object):
    """A totally ordered plan sequence."""

    def __init__(self, sequence: Sequence[Union[ValueOutputExpression, RegressionRuleApplier]], return_all_skeletons_flags: Optional[Union[Sequence[bool], bool]] = None, is_ordered: bool = True):
        """Initialize the totally ordered plan sequence.

        Args:
            sequence: the sequence of the plan.
            is_ordered: whether the sequence is ordered.
        """
        self.sequence = tuple(sequence)
        if type(return_all_skeletons_flags) is bool:
            return_all_skeletons_flags = [return_all_skeletons_flags] * len(sequence)
        else:
            self.return_all_skeletons_flags = tuple(return_all_skeletons_flags) if return_all_skeletons_flags is not None else None
        self.is_ordered = is_ordered

    sequence: Tuple[Union[ValueOutputExpression, RegressionRuleApplier], ...]
    """The sequence of the plan."""

    return_all_skeletons_flags: Optional[Tuple[bool, ...]]
    """For each item in the sequence, whether to return all the skeletons. This value can be set to None when is_ordered is False."""

    is_ordered: bool
    """Whether the sequence is ordered. If it's not ordered, then the sequence is treated as a set."""

    def exclude(self, index: int):
        """Exclude the given index from the sequence; return a new sequence."""
        return TotallyOrderedPlan(
            self.sequence[:index] + self.sequence[index + 1:],
            self.return_all_skeletons_flags[:index] + self.return_all_skeletons_flags[index + 1:] if self.return_all_skeletons_flags is not None else None,
            self.is_ordered
        )

    def get_return_all_skeletons_flag(self, index: int) -> bool:
        """Get the return_all_skeletons flag of the given index. If the flag is not set, then return True."""
        if self.return_all_skeletons_flags is None:
            return True
        return self.return_all_skeletons_flags[index]

    def gen_string(self):
        """Generate the string representation of the plan."""
        if self.is_ordered:
            return '(then ' + ' '.join([str(e) for e in self.sequence]) + ')'
        else:
            return '(and ' + ' '.join([str(e) for e in self.sequence]) + ')'

    def __len__(self):
        return len(self.sequence)

    def __str__(self):
        return self.gen_string()

    __repr__ = jacinle.repr_from_str


class PartiallyOrderedPlan(object):
    """A partially ordered plan is a set of totally ordered plan sequences."""

    def __init__(self, chains: Sequence[TotallyOrderedPlan]):
        """Initialize the partially ordered plan.

        Args:
            chains: a collection of the totally ordered plan sequences.
        """
        self.chains = tuple(chains)
        self.infeasible_chain_index = None

    chains: Tuple[TotallyOrderedPlan, ...]
    """The totally ordered plan sequences."""

    infeasible_index: Optional[int]
    """One can optionally mark one of the chain as infeasible to be the last chain. Currently this flag is not used."""

    def exclude(self, chain_index: int, item_index: int) -> 'PartiallyOrderedPlan':
        """Exclude the given item from the given chain; return a new plan."""
        if len(self.chains[chain_index]) == 1:
            assert item_index == 0
            return PartiallyOrderedPlan(self.chains[:chain_index] + self.chains[chain_index + 1:])
        return PartiallyOrderedPlan(self.chains[:chain_index] + (self.chains[chain_index].exclude(item_index),) + self.chains[chain_index + 1:])

    def add_chain(self, chain: Sequence[Union[ValueOutputExpression, RegressionRuleApplier]], return_all_skeletons_flags: Optional[Sequence[bool]] = False) -> 'PartiallyOrderedPlan':
        """Add a new chain to the plan; return a new plan."""
        plan = PartiallyOrderedPlan(self.chains + (TotallyOrderedPlan(chain, return_all_skeletons_flags),))
        return plan

    @classmethod
    def from_single_goal(cls, goal: Union[ValueOutputExpression, RegressionRuleApplier], return_all_skeletons_flag: bool = False) -> 'PartiallyOrderedPlan':
        """Create a plan from a single goal."""
        return cls((TotallyOrderedPlan((goal,), [return_all_skeletons_flag], is_ordered=True),))

    @property
    def nr_chains(self):
        return len(self.chains)

    @property
    def total_length(self):
        return sum(len(chain) for chain in self.chains)

    def set_infeasible_index(self, chain_index: int):
        self.infeasible_chain_index = chain_index

    def iter_feasible_chains(self) -> Iterator[Tuple[int, TotallyOrderedPlan]]:
        for i, seq in enumerate(self.chains):
            if i == self.infeasible_chain_index:
                continue
            yield i, seq

    def iter_goals(self) -> Iterator[ValueOutputExpression]:
        for chain in self.chains:
            for goal in chain.sequence:
                yield goal if isinstance(goal, ValueOutputExpression) else goal.goal_expression

    def gen_string(self):
        return '(and ' + ' '.join([chain.gen_string() for chain in self.chains]) + ')'

    def __len__(self):
        return self.total_length

    def __str__(self):
        return self.gen_string()

    __repr__ = jacinle.repr_from_str


class RegressionNode(object):
    def __init__(self, goal_expression: Union[BindExpression, AchieveExpression], associated_regression_rule: Optional[RegressionRuleApplier] = None):
        self.goal_expression = goal_expression
        self.associated_regression_rule = associated_regression_rule
        self.children = []

    goal_expression: Union[BindExpression, AchieveExpression]
    """The goal expression of this node."""

    children: List['RegressionNode']
    """The children of this node."""

    def add_child(self, node: 'RegressionNode'):
        self.children.append(node)

