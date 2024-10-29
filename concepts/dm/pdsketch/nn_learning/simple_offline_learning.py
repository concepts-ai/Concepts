#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : simple_offline_learning.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/21/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import torch.nn as nn
import jactorch.nn.functional as jacf
from jactorch.graph.context import ForwardContext

from concepts.dsl.tensor_state import concat_states
from concepts.dm.pdsketch.executor import PDSketchExecutor
from concepts.dm.pdsketch.execution_utils import recompute_state_variable_predicates_, recompute_all_cacheable_predicates_


class SimpleOfflineLearningModel(nn.Module):
    DEFAULT_OPTIONS = {
        'bptt': False
    }

    def __init__(self, executor: PDSketchExecutor, goal_loss_weight: float = 1.0, action_loss_weight: float = 1.0, **options):
        super().__init__()

        self.executor = executor
        self.functions = nn.ModuleDict()
        self.bce = nn.BCELoss()
        self.xent = nn.CrossEntropyLoss()
        # self.mse = nn.MSELoss(reduction='sum')
        self.mse = nn.SmoothL1Loss(reduction='sum')

        self.goal_loss_weight = goal_loss_weight
        self.action_loss_weight = action_loss_weight
        self.options = options

        for key, value in type(self).DEFAULT_OPTIONS.items():
            self.options.setdefault(key, value)

        self.init_networks(executor)

    training: bool

    @property
    def domain(self):
        return self.executor.domain

    def init_networks(self, executor: PDSketchExecutor):
        raise NotImplementedError()

    def forward_train(self, feed_dict):
        forward_ctx = ForwardContext(self.training)
        with forward_ctx.as_default():
            goal_expr = feed_dict['goal_expr']
            states, actions, goal_succ = feed_dict['state'], feed_dict['action'], feed_dict['goal_succ']

            batch_state = concat_states(*states)
            recompute_state_variable_predicates_(self.executor, batch_state)
            recompute_all_cacheable_predicates_(self.executor, batch_state)

            if goal_expr is not None:
                pred = self.executor.execute(goal_expr, state=batch_state).tensor
                target = goal_succ
                if self.training:
                    loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target.float())
                    # loss = self.bce(pred, target.float())
                    forward_ctx.add_loss(loss, 'goal', accumulate=self.goal_loss_weight)
                forward_ctx.add_accuracy(((pred > 0.5) == target).float().mean(), 'goal')

            if 'subgoals' in feed_dict:
                subgoals = feed_dict['subgoals']
                subgoals_done = feed_dict['subgoals_done']

                for i, (subgoal, subgoal_done) in enumerate(zip(subgoals, subgoals_done)):
                    pred = self.executor.execute(subgoal, batch_state).tensor
                    target = subgoal_done
                    if self.training:
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target.float())
                        # loss = self.bce(pred, target.float())
                        forward_ctx.add_loss(loss, f'subgoal/{i}', accumulate=self.goal_loss_weight)
                    forward_ctx.add_accuracy(((pred > 0.5) == target).float().mean(), f'subgoal/{i}')

            if self.action_loss_weight > 0:
                for i, action in enumerate(actions):
                    state = states[i]
                    next_state_pred = self.executor.apply_effect(action, state)
                    next_state_target = states[i + 1]

                    has_learnable_parameters = False
                    for eff in action.operator.effects:
                        predicate_def = eff.unwrapped_assign_expr.predicate.function
                        if not predicate_def.is_state_variable:
                            continue

                        has_learnable_parameters = True
                        feature_name = predicate_def.name

                        # if action.operator.name == 'pickup':
                        #     print('prev', state[feature_name].tensor[..., -2:])
                        #     print('pred', next_state_pred[feature_name].tensor[..., -2:])
                        #     print('trgt', next_state_target[feature_name].tensor[..., -2:])

                        this_loss = self.mse(
                            input=next_state_pred[feature_name].tensor.float(),
                            target=next_state_target[feature_name].tensor.float()
                        )

                        forward_ctx.add_loss(this_loss, f'a', accumulate=False)
                        forward_ctx.add_loss(this_loss, f'a/{action.operator.name}/{feature_name}', accumulate=self.action_loss_weight)

                        # if action.operator.name.__contains__('pickup') and this_loss.item() > 0.1:
                        #     print('\n' + '-' * 80)
                        #     print(action)
                        #     print('prev', state[feature_name].tensor[..., -10:])
                        #     print('pred', next_state_pred[feature_name].tensor[..., -10:])
                        #     print('trgt', (next_state_target[feature_name].tensor[..., -10:] - next_state_pred[feature_name].tensor[..., -10:]).abs())
                        #     print(this_loss, self.action_loss_weight, 'loss/a/' + action.operator.name + '/' + feature_name, forward_ctx.monitors.raw()['loss/a/' + action.operator.name + '/' + feature_name])

                    if has_learnable_parameters and self.options['bptt']:
                        recompute_all_cacheable_predicates_(self.executor, next_state_pred)
                        if goal_expr is not None:
                            pred = self.executor.execute(next_state_pred, goal_expr).tensor
                            target = goal_succ[i + 1]
                            loss = self.bce(pred, target.float())
                            forward_ctx.add_loss(loss, 'goal_bptt', accumulate=self.goal_loss_weight * 0.1)
                            forward_ctx.add_accuracy(((pred > 0.5) == target).float().mean(), 'goal_bptt')

                        if 'subgoals' in feed_dict:
                            subgoals = feed_dict['subgoals']
                            subgoals_done = feed_dict['subgoals_done']

                            for j, (subgoal, subgoal_done) in enumerate(zip(subgoals, subgoals_done)):
                                pred = self.executor.execute(subgoal, next_state_pred).tensor
                                target = subgoal_done[i + 1]
                                if self.training:
                                    loss = self.bce(pred, target.float())
                                    # loss = self.bce(pred, target.float())
                                    forward_ctx.add_loss(loss, f'subgoal_bptt/{j}', accumulate=self.goal_loss_weight * 0.1)
                                forward_ctx.add_accuracy(((pred > 0.5) == target).float().mean(), f'subgoal_bptt/{j}')

        return forward_ctx.finalize()
