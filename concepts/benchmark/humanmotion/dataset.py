#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : dataset.py
# Author : Joy Hsu
# Email  : joycj@stanford.edu
# Date   : 03/23/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os.path as osp

import jacinle.io as io

from jacinle.logging import get_logger
from jacinle.utils.container import GView
from jactorch.data.dataset import FilterableDatasetUnwrapped, FilterableDatasetView

from concepts.benchmark.humanmotion.utils import nsclseq_to_nscltree, nsclseq_to_nsclqsseq, nscltree_to_nsclqstree, program_to_nsclseq

import numpy as np
import math



logger = get_logger(__file__)

__all__ = ['NSTrajDataset', 'MotionClassificationDataset']


class NSTrajDatasetUnwrapped(FilterableDatasetUnwrapped):
    def __init__(self, data_dir, data_split_file, split, data_source, no_gt_segments, filter_supervision, max_frames=150):
        super().__init__()

        self.labels_json = osp.join(data_dir, 'motion_concepts.json')
        self.questions_json = osp.join(data_dir, 'questions.json')
        self.joints_root = osp.join(data_dir, 'motion_sequences')

        self.labels = io.load_json(self.labels_json)
        self.questions = io.load_json(self.questions_json)
        self.split_question_ids = io.load_json(data_split_file)[split]

        self.data_source = data_source

        self.max_frames = max_frames
        self.no_gt_segments = no_gt_segments
        self.filter_supervision = filter_supervision

    def _get_questions(self):
        return self.questions

    def _get_metainfo(self, index):
        question = self.questions[self.split_question_ids[index]]

        # program section
        has_program = False
        if 'program_nsclseq' in question:
            question['program_raw'] = question['program_nsclseq']
            question['program_seq'] = question['program_nsclseq']
            has_program = True
        elif 'program' in question:
            question['program_raw'] = question['program']
            question['program_seq'] = program_to_nsclseq(question['program'])
            has_program = True

        if has_program:
            question['program_tree'] = nsclseq_to_nscltree(question['program_seq'])
            question['program_qsseq'] = nsclseq_to_nsclqsseq(question['program_seq'])
            question['program_qstree'] = nscltree_to_nsclqstree(question['program_tree'])

        return question

    def __getitem__(self, index):
        metainfo = GView(self.get_metainfo(index))
        feed_dict = GView()

        question = self.questions[self.split_question_ids[index]]

        if 'program_raw' in metainfo:
            feed_dict.program_raw = metainfo.program_raw
            feed_dict.program_seq = metainfo.program_seq
            feed_dict.program_tree = metainfo.program_tree
            feed_dict.program_qsseq = metainfo.program_qsseq
            feed_dict.program_qstree = metainfo.program_qstree

        if '/' in question['answer']:
            question['answer'] = question['answer'].split('/')[0]
        feed_dict.answer = question['answer']
        feed_dict.question_type = question['query_type']
        feed_dict.segment_boundaries = []
        feed_dict.question_text = question['question']

        # process joints
        if self.data_source == 'teach_synth':
            id_name = 'teach_synth_id'
            motion_id = question[id_name]
            num_segments = len(self.labels[motion_id]['labels'])
        else:
            id_name = 'babel_id'
            motion_id = question[id_name]
            num_segments = len(self.labels[motion_id])
        feed_dict.babel_id = motion_id

        joints = np.load(osp.join(self.joints_root, motion_id, 'joints.npy')) # T, V, C
        # change shape of joints to match model
        joints = joints[:, :, :, np.newaxis] # T, V, C, M
        joints = joints.transpose(2, 0, 1, 3) # C, T, V, M

        # label info
        if self.data_source == 'teach_synth':
            labels_frame_info = self.labels[motion_id]['labels']
        else:
            labels_frame_info = self.labels[motion_id]

        if 'filter_answer_0' in question:
            filter_segment = labels_frame_info[question['filter_answer_0']]
            if filter_segment['end_f'] > np.shape(joints)[1]: # right now end frame can be slightly off (dataset issue)
                filter_segment['end_f'] = np.shape(joints)[1]
            feed_dict.filter_boundaries = [(filter_segment['start_f'], filter_segment['end_f'])]
            if 'filter_answer_1' in question:
                filter_segment = labels_frame_info[question['filter_answer_1']]
                if filter_segment['end_f'] > np.shape(joints)[1]: # right now end frame can be slightly off (dataset issue)
                    filter_segment['end_f'] = np.shape(joints)[1]
                feed_dict.filter_boundaries.append((filter_segment['start_f'], filter_segment['end_f']))

        if not self.no_gt_segments:
            joints_combined = np.zeros((num_segments, 3, self.max_frames, 22, 1), dtype=np.float32) # num_segs, C, T, V, M

            for seg_i, seg in enumerate(labels_frame_info):
                if seg['end_f'] > np.shape(joints)[1]: # right now end frame can be slightly off (dataset issue)
                    seg['end_f'] = np.shape(joints)[1]
                num_frames = seg['end_f'] - seg['start_f']

                if num_frames > self.max_frames: # clip segments to max_frames
                    num_frames = self.max_frames

                joints_combined[seg_i, :, :num_frames, :, :] = joints[:, seg['start_f']: seg['start_f'] + num_frames, :, :]

                feed_dict.segment_boundaries.append((seg['start_f'], (seg['start_f'] + num_frames)))

            feed_dict.joints = joints_combined
            feed_dict.num_segs = num_segments
        else:
            total_num_frames = np.shape(joints)[1]
            num_frames_per_seg = 45
            overlap_frames = 15
            num_segments = math.ceil(total_num_frames / num_frames_per_seg)
            feed_dict['info'] = []
            joints_combined = np.zeros((num_segments, 3, num_frames_per_seg + overlap_frames*2, 22, 1), dtype=np.float32) # num_segs, C, T, V, M
            for i in range(num_segments):
                start_f = i * num_frames_per_seg
                end_f = (i + 1) * num_frames_per_seg
                if end_f > total_num_frames: end_f = total_num_frames

                missing_before_context = overlap_frames - start_f if start_f < overlap_frames else 0
                existing_after_context = total_num_frames - end_f
                if existing_after_context > overlap_frames: existing_after_context = overlap_frames

                joints_combined[i, :, missing_before_context:overlap_frames+(end_f - start_f)+existing_after_context, :, :] = joints[:, start_f - (overlap_frames - missing_before_context):end_f + existing_after_context, :, :]

                feed_dict.segment_boundaries.append((start_f - (overlap_frames - missing_before_context), end_f + existing_after_context))

            feed_dict.joints = joints_combined
            feed_dict.num_segs = num_segments

        return feed_dict.raw()

    def __len__(self):
        return len(self.split_question_ids)

class NSTrajDatasetFilterableView(FilterableDatasetView):
    def filter_questions(self, allowed):
        def filt(question):
            return question['query_type'] in allowed

        return self.filter(filt, 'filter-question-type[allowed={{{}}}]'.format(','.join(list(allowed))))


    def make_dataloader(self, batch_size, shuffle, drop_last, nr_workers):
        from jactorch.data.dataloader import JacDataLoader
        from jactorch.data.collate import VarLengthCollateV2

        collate_guide = {
            'joints': 'concat',
            'answer': 'skip',
            'segment_boundaries': 'skip',
            'filter_boundaries': 'skip',

            'program_raw': 'skip',
            'program_seq': 'skip',
            'program_tree': 'skip',
            'program_qsseq': 'skip',
            'program_qstree': 'skip',
        }

        return JacDataLoader(
            self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
            num_workers=nr_workers, pin_memory=True,
            collate_fn=VarLengthCollateV2(collate_guide))


def NSTrajDataset(*args, **kwargs):
    return NSTrajDatasetFilterableView(NSTrajDatasetUnwrapped(*args, **kwargs))

class MotionClassificationDatasetUnwrapped(FilterableDatasetUnwrapped):
    def __init__(self, labels_json, joints_root, symbolic, max_frames=150):
        super().__init__()
        self.labels_json = labels_json
        self.joints_root = joints_root
        self.act2idx_json = 'nscl/datasets/babel/action_label_2_idx.json'

        self.labels = io.load_json(self.labels_json)
        self.babel_ids = list(self.labels.keys())
        self.act2idx = io.load_json(self.act2idx_json)

        self.symbolic = symbolic
        self.max_frames = max_frames

    def __getitem__(self, index):
        feed_dict = GView()

        babel_id = self.babel_ids[index]
        actions = []

        num_segments = len(self.labels[babel_id])

        joints = np.load(osp.join(self.joints_root, f'{babel_id}.npy')) # T, V, C

        # change shape of joints to match model
        joints = joints[:, :, :, np.newaxis] # T, V, C, M
        joints = joints.transpose(2, 0, 1, 3) # C, T, V, M

        joints_combined = np.zeros((num_segments, 3, self.max_frames, 22, 1), dtype=np.float32) # num_segs, C, T, V, M

        # TODO: take out segments that only have one frame (to be consistent with unbatched version)
        for seg_i, seg in enumerate(self.labels[babel_id]):
            if seg['end_f'] > np.shape(joints)[1]: # right now end frame can be slightly off (dataset issue)
                seg['end_f'] = np.shape(joints)[1]
            num_frames = seg['end_f'] - seg['start_f']

            if num_frames > self.max_frames: # clip segments to max_frames
                num_frames = self.max_frames

            joints_combined[seg_i, :, :num_frames, :, :] = joints[:, seg['start_f']: seg['start_f'] + num_frames, :, :]

            action = seg['action']
            if self.symbolic: # option 1: add all actions
                for i in range(len(action)):
                    action[i] = self.act2idx[action[i]]
                actions.append(action)
            else: # option 2: pick randomly among actions
                random_action_idx = np.random.randint(len(action))
                action_idx = self.act2idx[action[random_action_idx].replace('.', '')] # remove periods from action names
                actions.append([action_idx])

        feed_dict.actions = actions
        feed_dict.joints = joints_combined
        feed_dict.id = babel_id
        feed_dict.num_segs = num_segments

        return feed_dict.raw()

    def __len__(self):
        return len(self.babel_ids)

class MotionClassificationDatsetFilterableView(FilterableDatasetView):
    def make_dataloader(self, batch_size, shuffle, drop_last, nr_workers):
        from jactorch.data.dataloader import JacDataLoader
        from jactorch.data.collate import VarLengthCollateV2

        collate_guide = {
            'joints': 'concat',
            'actions': 'skip',
        }

        gdef.update_collate_guide(collate_guide)

        return JacDataLoader(
            self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
            num_workers=nr_workers, pin_memory=True,
            collate_fn=VarLengthCollateV2(collate_guide))

def MotionClassificationDataset(*args, **kwargs):
    return MotionClassificationDatsetFilterableView(MotionClassificationDatasetUnwrapped(*args, **kwargs))



