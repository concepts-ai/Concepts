#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : dataset.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/07/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import tarfile
import os.path as osp
from typing import Optional

import numpy as np
import torch
from jactorch.data.dataset import FilterableDatasetUnwrapped, FilterableDatasetView
from jactorch.data.dataloader import JacDataLoader
from jactorch.data.collate import VarLengthCollateV2

__all__ = ['ShapesDatasetUnwrapped', 'ShapesDatasetFilterableView', 'ShapesDataset']


class ShapesDatasetUnwrapped(FilterableDatasetUnwrapped):
    def __init__(self, split: str, data_dir: Optional[str] = None, incl_pseudo_labels: bool = False):
        super().__init__()

        _init_shapes_dataset()
        self.data_dir = data_dir
        if self.data_dir is None:
            self.data_dir = osp.join(osp.dirname(__file__), 'shapes_dataset')
        self.split = split
        self.data = self._load_data()
        self.incl_pseudo_labels = incl_pseudo_labels

    def _load_data(self):
        images_filename = osp.join(self.data_dir, f'{self.split}.input.npy')
        questions_filename = osp.join(self.data_dir, f'{self.split}.query_str.txt')
        questions_program_filename = osp.join(self.data_dir, f'{self.split}.query')
        answer_filename = osp.join(self.data_dir, f'{self.split}.output')

        images = np.load(images_filename)[..., ::-1]  # BGR -> RGB
        questions = [line.strip() for line in open(questions_filename, 'r')]
        questions_program = [line.strip() for line in open(questions_program_filename, 'r')]
        answers = [line.strip() for line in open(answer_filename, 'r')]

        assert len(images) == len(questions) == len(questions_program) == len(answers)

        return {
            'images': images,
            'questions': questions,
            'questions_program': questions_program,
            'answers': answers
        }

    def _get_metainfo(self, index):
        return {
            'question': self.data['questions'][index],
            'question_program': self.data['questions_program'][index],
            'answer': self.data['answers'][index] == 'true',
            'question_length': len(self.data['questions'][index].split())
        }

    def __getitem__(self, index):
        return {
            'image': _to_image(self.data['images'][index]),
            'question': self.data['questions'][index],
            'question_program': self.data['questions_program'][index],
            'answer': self.data['answers'][index] == 'true'
        }

    def __len__(self):
        return len(self.data['images'])


def _to_image(image):
    image = image.transpose(2, 0, 1) / 255.0
    image = image.astype(np.float32)
    image = (image - 0.5) * 2
    return torch.tensor(image)


def _init_shapes_dataset():
    current_dir = osp.dirname(__file__)
    data_dir = osp.join(current_dir, 'shapes_dataset')
    if not osp.exists(data_dir):
        tar_filename = osp.join(current_dir, 'shapes_dataset.tar.gz')
        print('Extracting {} ...'.format(tar_filename))
        with tarfile.open(tar_filename, 'r:gz') as f:
            f.extractall(current_dir)


class ShapesDatasetFilterableView(FilterableDatasetView):
    def make_dataloader(self, batch_size: int, shuffle: bool, drop_last: bool, nr_workers: int) -> JacDataLoader:
        collate_guide = {
            'questions': 'skip',
            'questions_program': 'skip',
            'answer': 'skip'
        }
        return JacDataLoader(
            self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
            num_workers=nr_workers, pin_memory=True,
            collate_fn=VarLengthCollateV2(collate_guide)
        )

    def filter_question_length(self, length: int) -> 'ShapesDatasetFilterableView':
        def filt(meta):
            return meta['question_length'] <= length
        return self.filter(filt, f'filter-qlength[{length}]')


def ShapesDataset(split, data_dir=None) -> ShapesDatasetFilterableView:
    return ShapesDatasetFilterableView(ShapesDatasetUnwrapped(split, data_dir))

