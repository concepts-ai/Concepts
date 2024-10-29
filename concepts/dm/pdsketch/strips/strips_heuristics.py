#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : strips_heuristics.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/20/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Optional, Union, Tuple, List, Set, Dict, Callable

from concepts.dm.pdsketch.strips.strips_expression import SProposition, SState
from concepts.dm.pdsketch.strips.strips_grounded_expression import GSBoolForwardDiffReturn, GSBoolOutputExpression, GSSimpleBoolAssignExpression, GSConditionalAssignExpression
from concepts.dm.pdsketch.strips.strips_grounding import GStripsProblem, GStripsTranslatorBase

__all__ = ['StripsHeuristic', 'StripsBlindHeuristic', 'StripsRPGHeuristic', 'StripsHFFHeuristic']


class StripsHeuristic(object):
    """The base class for STRIPS heuristics."""

    def __init__(self, task: GStripsProblem, translator: Optional[GStripsTranslatorBase] = None):
        """Initialize the heuristic.

        Args:
            task: the task to compute the heuristic value (including the goal and the operators)
            translator: the translator to translate the task representation into the `GS*` (grounded STRIPS) representation.
        """
        self._task = task
        self._goal_func = task.goal.compile()
        self._translator = translator

    @property
    def task(self) -> GStripsProblem:
        """The grounded STRIPS planning task."""
        return self._task

    @property
    def goal_func(self) -> Callable[[SState], Union[bool, GSBoolForwardDiffReturn]]:
        """The (compiled version) of goal function of the task."""
        return self._goal_func

    @property
    def translator(self) -> Optional[GStripsTranslatorBase]:
        """The translator used to translate the task representation into the `GS*` (grounded STRIPS) representation."""
        return self._translator

    @classmethod
    def from_type(cls, type_identifier: str, task: GStripsProblem, translator: Optional[GStripsTranslatorBase] = None, **kwargs) -> 'StripsHeuristic':
        """Create a heuristic from the given type identifier.

        Args:
            type_identifier: the type identifier of the heuristic.
            task: the grounded STRIPS planning task.
            translator: the translator used to translate the planning task to the `GS*` (grounded STRIPS) representation.
            **kwargs: additional arguments to pass to the constructor.

        Returns:
            the created heuristic.
        """
        if type_identifier == 'hff':
            return StripsHFFHeuristic(task, translator, **kwargs)
        elif type_identifier == 'blind':
            return StripsBlindHeuristic(task, translator)
        else:
            raise ValueError(f'Unknown heuristic type {type_identifier}.')

    def compute(self, state: SState) -> int:
        """Compute the heuristic value of the given state.

        Args:
            state: the state to compute the heuristic value.

        Returns:
            the heuristic value of the given state.
        """
        raise NotImplementedError()


class StripsBlindHeuristic(StripsHeuristic):
    """A blind heuristic that always returns 1 if the goal is not satisfied, and 0 otherwise."""

    def compute(self, state: SState) -> int:
        goal_rv = self._goal_func(state)
        return 0 if goal_rv else 1


class StripsRPGHeuristic(StripsHeuristic):
    """RPG heuristic (relaxed planning graph)."""

    def __init__(self, task: GStripsProblem, translator: Optional[GStripsTranslatorBase] = None, forward_relevance_analysis: bool = True, backward_relevance_analysis: bool = True):
        """Initialize an RPG heuristic.

        Args:
            task: the grounded STRIPS planning task.
            translator: the translator used to translate the planning task to the `GS*` (grounded STRIPS) representation.
            forward_relevance_analysis: whether to perform forward relevance analysis.
            backward_relevance_analysis: whether to perform backward relevance analysis.
        """
        super().__init__(task, translator)

        self._forward_relevance_analysis = forward_relevance_analysis
        self._backward_relevance_analysis = backward_relevance_analysis

        if task.is_relaxed:
            self._relaxed = task
        else:
            assert translator is not None
            self._relaxed = translator.recompile_relaxed_task(task, forward_relevance_analysis=forward_relevance_analysis, backward_relevance_analysis=backward_relevance_analysis)

    @property
    def relaxed(self) -> GStripsProblem:
        """The relaxed version of the task."""
        return self._relaxed

    def compute_rpg_forward_diff(self, state: SState, relaxed_task: Optional[GStripsProblem] = None) -> Tuple[
        List[Set[SState]],
        Dict[SProposition, int],
        List[List[Tuple[int, int, Set[SProposition]]]],
        List[List[Tuple[int, int, Set[SProposition]]]],
        GSBoolForwardDiffReturn
    ]:
        """Compute the relaxed planning graph using forward differentiation.

        Args:
            state: the state to compute the relaxed planning graph.
            relaxed_task: the relaxed version of the task. If not provided, the task will be relaxed using the translator.

        Returns:
            the relaxed planning graph, which is represented as a tuple of the following elements:
            - a list of sets of states, where each set contains all the states that can be reached from the initial state in the given number of steps.
            - a dictionary that maps propositions to their level.
            - a list of lists of tuples, where each tuple contains the following elements:
                - the index of the action in the relaxed task operator list.
                - the index of the effect in the operator.
                - the propositions that contributes to the effect (computed using Forward Diff).
            - a list of lists of tuples, where each tuple contains the following elements:
                - the index of the action that achieves the proposition.
                - the level of the proposition.
                - the set of propositions that are achieved by the action.
            - the result of forward differentiation of the goal function.
        """

        if relaxed_task is None:
            relaxed_task = self._relaxed

        with GSBoolOutputExpression.enable_forward_diff_ctx():
            F_sets = [set(state)]
            A_sets = []
            D_sets = []

            used_operators = set()
            used_derived_predicates = set()
            # print('rpginit', F_sets[-1])

            goal_rv = self._goal_func(F_sets[-1])
            while not goal_rv.rv:
                # for op in relaxed_task.operators:
                #     print(' ', op.raw_operator, op.precondition, op.applicable(F_sets[-1]))

                new_ops: List[Tuple[int, int, Set[SProposition]]] = list()  # op_index, eff_index, op_precondition
                # print('Starting new step')
                for i, op in enumerate(relaxed_task.operators):
                    op_pred_rv = None
                    for j, e in enumerate(op.effects):
                        if (i, j) not in used_operators:
                            if op_pred_rv is None:
                                op_pred_rv = op.applicable(F_sets[-1])
                                # print('Evaluating op', op.raw_operator, op_pred_rv)
                            if op_pred_rv.rv:
                                if isinstance(e, GSSimpleBoolAssignExpression):
                                    # TODO:: check if it should be (i, j) or (i, 0, op_pred_rv.propositions)
                                    new_ops.append((i, j, op_pred_rv.propositions))
                                    used_operators.add((i, j))
                                elif isinstance(e, GSConditionalAssignExpression):
                                    eff_pred_rv = e.applicable(F_sets[-1])
                                    if eff_pred_rv.rv:
                                        new_ops.append((i, j, op_pred_rv.propositions | eff_pred_rv.propositions))
                                        # print('  Use operator', i, j)
                                        used_operators.add((i, j))
                            else:
                                break

                new_F = F_sets[-1].copy()
                for op_index, effect_index, _ in new_ops:
                    op = relaxed_task.operators[op_index]
                    eff = op.effects[effect_index]
                    new_F.update(eff.add_effects)

                new_dps: List[Tuple[int, int, Set[SProposition]]] = list()  # dp_index, eff_index, dp_precondition
                for i, dp in enumerate(relaxed_task.derived_predicates):
                    for j, e in enumerate(dp.effects):
                        if (i, j) not in used_derived_predicates:
                            dp_pred_rv = e.applicable(new_F)
                            # print((i, j), dp_pred_rv)
                            if dp_pred_rv.rv:
                                used_derived_predicates.add((i, j))
                                new_dps.append((i, j, dp_pred_rv.propositions))

                for dp_index, effect_index, _ in new_dps:
                    dp = relaxed_task.derived_predicates[dp_index]
                    eff = dp.effects[effect_index]
                    new_F.update(eff.add_effects)

                # print('depth', len(F_sets), new_F)
                if len(new_F) == len(F_sets[-1]):
                    break

                A_sets.append(new_ops)
                D_sets.append(new_dps)
                F_sets.append(new_F)
                goal_rv = self._goal_func(F_sets[-1])

            F_levels = {}
            for i in range(0, len(F_sets)):
                for f in F_sets[i]:
                    if f not in F_levels:
                        F_levels[f] = i

            return F_sets, F_levels, A_sets, D_sets, goal_rv

    def compute(self, state: SState) -> int:
        raise NotImplementedError()


class StripsHFFHeuristic(StripsRPGHeuristic):
    def compute(self, state: SState) -> int:
        """Compute the hFF heuristic value for the given state.

        Args:
            state: the state to compute the heuristic value for.

        Returns:
            the heuristic value.
        """

        # NB(Jiayuan Mao @ 12/04): We don't need to do relevance analysis because the actual computation for hFF will do similar things anyway...
        relaxed = self._translator.recompile_task_new_state(self._relaxed, state, forward_relevance_analysis=False, backward_relevance_analysis=False)
        state = relaxed.state

        F_sets, F_levels, A_sets, D_sets, goal_rv = self.compute_rpg_forward_diff(state, relaxed_task=relaxed)

        if not goal_rv.rv:
            return int(1e9)

        selected: Set[Tuple[int, int]] = set()
        selected_facts = [set() for _ in F_sets]

        goal_propositions = goal_rv.propositions
        for pred in goal_propositions:
            if pred not in F_levels:
                return int(1e9)
            selected_facts[F_levels[pred]].add(pred)

        for t in reversed(list(range(len(A_sets)))):
            for dp_index, effect_index, propositions in D_sets[t]:
                dp = self._relaxed.derived_predicates[dp_index]
                eff = dp.effects[effect_index]
                if eff.add_effects.intersection(selected_facts[t+1]):
                    # print('Selecting Derived Predicate', dp_index, effect_index, propositions, eff.add_effects)
                    selected_facts[t + 1] -= eff.add_effects
                    for pred in propositions:
                        # print('  Selecting predicate', pred)
                        selected_facts[F_levels[pred]].add(pred)
            for op_index, effect_index, propositions in A_sets[t]:
                op = self._relaxed.operators[op_index]
                eff = op.effects[effect_index]
                if eff.add_effects.intersection(selected_facts[t+1]):
                    # print('Selecting Action', op_index, effect_index, op.raw_operator, propositions, eff.add_effects)
                    selected.add((op_index, effect_index))  # only add the operator or both? Maybe at each level, just add the operator? or add a flag?
                    selected_facts[t + 1] -= eff.add_effects
                    for pred in propositions:
                        # print('  Selecting predicate', pred)
                        selected_facts[F_levels[pred]].add(pred)
        return len(selected)
