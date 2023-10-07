#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : dataset.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/26/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os.path as osp

import nltk
import numpy as np
from PIL import Image

import jacinle.io as io
import jaclearn.vision.coco.mask_utils as mask_utils

from jacinle.logging import get_logger
from jacinle.utils.container import GView
from jactorch.data.dataset import FilterableDatasetUnwrapped, FilterableDatasetView

from concepts.benchmark.common.vocab import Vocab
from concepts.benchmark.clevr.clevr_constants import g_attribute_concepts, g_relational_concepts

logger = get_logger(__file__)

__all__ = ['CLEVRDatasetUnwrapped', 'CLEVRDatasetFilterableView', 'make_dataset', 'CLEVRCustomTransferDataset', 'make_custom_transfer_dataset']


class CLEVRDatasetUnwrapped(FilterableDatasetUnwrapped):
    def __init__(self, scenes_json, questions_json, image_root, image_transform, vocab_json, output_vocab_json, question_transform=None, incl_scene=True, incl_raw_scene=False):
        super().__init__()

        self.scenes_json = scenes_json
        self.questions_json = questions_json
        self.image_root = image_root
        self.image_transform = image_transform
        self.vocab_json = vocab_json
        self.output_vocab_json = output_vocab_json
        self.question_transform = question_transform

        self.incl_scene = incl_scene
        self.incl_raw_scene = incl_raw_scene

        logger.info('Loading scenes from: "{}".'.format(self.scenes_json))
        self.scenes = io.load_json(self.scenes_json)['scenes']
        if isinstance(self.questions_json, (tuple, list)):
            self.questions = list()
            for filename in self.questions_json:
                logger.info('Loading questions from: "{}".'.format(filename))
                self.questions.extend(io.load_json(filename)['questions'])
        else:
            logger.info('Loading questions from: "{}".'.format(self.questions_json))
            self.questions = io.load_json(self.questions_json)['questions']

        if self.vocab_json is not None:
            logger.info('Loading vocab from: "{}".'.format(self.vocab_json))
            self.vocab = Vocab.from_json(self.vocab_json)
        else:
            logger.info('Building the vocab.')
            self.vocab = Vocab.from_dataset(self, keys=['question_tokenized'])

        if output_vocab_json is not None:
            logger.info('Loading output vocab from: "{}".'.format(self.output_vocab_json))
            self.output_vocab = Vocab.from_json(self.output_vocab_json)
        else:
            logger.info('Building the output vocab.')
            self.output_vocab = Vocab.from_dataset(self, keys=['answer'], single_word=True)

    def _get_metainfo(self, index):
        question = self.questions[index]
        scene = self.scenes[question['image_index']]
        question['scene'] = scene
        question['program'] = question.pop('program', None)  # In CLEVR-Humans, there is no program.

        question['image_index'] = question['image_index']
        question['image_filename'] = self._get_image_filename(scene)
        question['question_index'] = index
        question['question_tokenized'] = nltk.word_tokenize(question['question'])
        question['question_type'] = get_question_type(question['program'])

        # question['scene_complexity'] = len(scene['objects'])
        # question['program_complexity'] = len(question['program'])

        return question

    def _get_image_filename(self, scene: dict) -> str:
        return scene['image_filename']

    def __getitem__(self, index):
        metainfo = GView(self.get_metainfo(index))
        feed_dict = GView()

        # metainfo annotations
        if self.incl_scene:
            feed_dict.update(annotate_objects(metainfo.scene))
            if 'objects' in feed_dict:
                # NB(Jiayuan Mao): in some datasets_v1, object information might be completely unavailable.
                feed_dict.objects_raw = feed_dict.objects.copy()
            feed_dict.update(annotate_scene(metainfo.scene))

        if self.incl_raw_scene:
            feed_dict.scene = metainfo.scene

        # image
        feed_dict.image_index = metainfo.image_index
        feed_dict.image_filename = metainfo.image_filename
        if self.image_root is not None and feed_dict.image_filename is not None:
            feed_dict.image = Image.open(osp.join(self.image_root, feed_dict.image_filename)).convert('RGB')
            feed_dict.image, feed_dict.objects = self.image_transform(feed_dict.image, feed_dict.objects)

        # question
        feed_dict.question_index = metainfo.question_index
        feed_dict.question_raw = metainfo.question
        feed_dict.question_raw_tokenized = metainfo.question_tokenized
        feed_dict.question = metainfo.question_tokenized
        feed_dict.question_type = metainfo.question_type
        feed_dict.answer = canonize_answer(metainfo.answer, None)

        if self.question_transform is not None:
            self.question_transform(feed_dict)
        feed_dict.question = np.array(self.vocab.map_sequence(feed_dict.question), dtype=np.int64)

        return feed_dict.raw()

    def __len__(self):
        return len(self.questions)


class CLEVRDatasetFilterableView(FilterableDatasetView):
    def filter_program_size_raw(self, max_length):
        def filt(question):
            return question['program'] is None or len(question['program']) <= max_length

        return self.filter(filt, 'filter-program-size-clevr[{}]'.format(max_length))

    def filter_scene_size(self, max_scene_size):
        def filt(question):
            return len(question['scene']['objects']) <= max_scene_size

        return self.filter(filt, 'filter-scene-size[{}]'.format(max_scene_size))

    def filter_question_type(self, *, allowed=None, disallowed=None):
        def filt(question):
            if allowed is not None:
                return question['question_type'] in allowed
            elif disallowed is not None:
                return question['question_type'] not in disallowed

        if allowed is not None:
            return self.filter(filt, 'filter-question-type[allowed={' + (','.join(list(allowed))) + '}]')
        elif disallowed is not None:
            return self.filter(filt, 'filter-question-type[disallowed={' + (','.join(list(disallowed))) + '}]')
        else:
            raise ValueError('Must provide either allowed={...} or disallowed={...}.')

    def make_dataloader(self, batch_size, shuffle, drop_last, nr_workers):
        from jactorch.data.dataloader import JacDataLoader
        from jactorch.data.collate import VarLengthCollateV2

        collate_guide = {
            'scene': 'skip',
            'objects_raw': 'skip',
            'objects': 'concat',

            'image_index': 'skip',
            'image_filename': 'skip',
            'question_index': 'skip',
            'question_metainfo': 'skip',

            'question_raw': 'skip',
            'question_raw_tokenized': 'skip',
            'question_type': 'skip',
            'question': 'pad',

            'program_raw': 'skip',
            'program_seq': 'skip',
            'program_tree': 'skip',
            'program_qsseq': 'skip',
            'program_qstree': 'skip',

            'question_type': 'skip',
            'answer': 'skip',
        }

        # Scene annotations.
        for attr_name in g_attribute_concepts:
            collate_guide['attribute_' + attr_name] = 'concat'
            collate_guide['attribute_relation_' + attr_name] = 'concat'
        for attr_name in g_relational_concepts:
            collate_guide['relation_' + attr_name] = 'concat'

        return JacDataLoader(
            self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
            num_workers=nr_workers, pin_memory=True,
            collate_fn=VarLengthCollateV2(collate_guide)
        )


class CLEVRCustomTransferDataset(FilterableDatasetUnwrapped):
    def __init__(self, scenes_json, questions_json, image_root, image_transform, query_list_key, custom_fields, output_vocab_json=None, incl_scene=True, incl_raw_scene=False):
        super().__init__()

        self.scenes_json = scenes_json
        self.questions_json = questions_json
        self.image_root = image_root
        self.image_transform = image_transform
        self.query_list_key = query_list_key
        self.custom_fields = custom_fields
        self.output_vocab_json = output_vocab_json

        self.incl_scene = incl_scene
        self.incl_raw_scene = incl_raw_scene

        logger.info('Loading scenes from: "{}".'.format(self.scenes_json))
        self.scenes = io.load_json(self.scenes_json)['scenes']
        if isinstance(self.questions_json, (tuple, list)):
            self.questions = list()
            for filename in self.questions_json:
                logger.info('Loading questions from: "{}".'.format(filename))
                self.questions.extend(io.load_json(filename)[query_list_key])
        else:
            logger.info('Loading questions from: "{}".'.format(self.questions_json))
            self.questions = io.load_json(self.questions_json)[query_list_key]

        if output_vocab_json is not None:
            logger.info('Loading output vocab from: "{}".'.format(self.output_vocab_json))
            self.output_vocab = Vocab.from_json(self.output_vocab_json)
        else:
            logger.info('Building the output vocab.')
            self.output_vocab = Vocab.from_dataset(self, keys=['answer'], single_word=True)

    def _get_scene_index_from_question(self, question):
        if 'image_index' in question:
            return question['image_index']
        if 'scene_index' in question:
            return question['scene_index']
        raise KeyError('Cannot find scene index from question.')

    def _get_metainfo(self, index):
        question = self.questions[index]
        scene = self.scenes[self._get_scene_index_from_question(question)]
        question['scene'] = scene
        question['program'] = question.pop('program', None)  # In CLEVR-Humans, there is no program.

        question['image_index'] = self._get_scene_index_from_question(question)
        question['image_filename'] = self._get_image_filename(scene)
        question['question_index'] = index
        question['question'] = question['question']
        question['answer'] = question['answer']
        question['question_type'] = question.get('question_type', self.query_list_key)

        for field_name in self.custom_fields:
            question[field_name] = scene[field_name]

        return question

    def _get_image_filename(self, scene: dict) -> str:
        return scene['image_filename']

    def __getitem__(self, index):
        metainfo = GView(self.get_metainfo(index))
        feed_dict = GView()

        # metainfo annotations
        feed_dict.update(annotate_objects(metainfo.scene))
        if 'objects' in feed_dict:
            # NB(Jiayuan Mao): in some datasets_v1, object information might be completely unavailable.
            feed_dict.objects_raw = feed_dict.objects.copy()

        if self.incl_scene:
            feed_dict.update(annotate_scene(metainfo.scene))
        if self.incl_raw_scene:
            feed_dict.scene = metainfo.scene

        # image
        feed_dict.image_index = metainfo.image_index
        feed_dict.image_filename = metainfo.image_filename
        if self.image_root is not None and feed_dict.image_filename is not None:
            feed_dict.image = Image.open(osp.join(self.image_root, feed_dict.image_filename)).convert('RGB')
            feed_dict.image, feed_dict.objects = self.image_transform(feed_dict.image, feed_dict.objects)

        # question
        feed_dict.question_raw = metainfo.question
        feed_dict.question_type = metainfo.question_type
        feed_dict.answer = metainfo.answer
        return feed_dict.raw()

    def __len__(self):
        return len(self.questions)

    def make_dataloader(self, batch_size, shuffle, drop_last, nr_workers):
        from jactorch.data.dataloader import JacDataLoader
        from jactorch.data.collate import VarLengthCollateV2

        collate_guide = {
            'scene': 'skip',
            'objects_raw': 'skip',
            'objects': 'concat',

            'image_index': 'skip',
            'image_filename': 'skip',
            'question_index': 'skip',
            'question_metainfo': 'skip',

            'question_raw': 'skip',
            'question_type': 'skip',
            'answer': 'skip',
        }

        for field in self.custom_fields:
            collate_guide[field] = 'skip'

        # Scene annotations.
        for attr_name in g_attribute_concepts:
            collate_guide['attribute_' + attr_name] = 'concat'
            collate_guide['attribute_relation_' + attr_name] = 'concat'
        for attr_name in g_relational_concepts:
            collate_guide['relation_' + attr_name] = 'concat'

        return JacDataLoader(
            self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
            num_workers=nr_workers, pin_memory=True,
            collate_fn=VarLengthCollateV2(collate_guide)
        )


def make_dataset(scenes_json, questions_json, image_root, *, image_transform=None, vocab_json=None, output_vocab_json=None, filterable_view_cls=None, **kwargs):
    if filterable_view_cls is None:
        filterable_view_cls = CLEVRDatasetFilterableView

    if image_transform is None:
        import jactorch.transforms.bbox as T
        image_transform = T.Compose([
            T.NormalizeBbox(),
            T.Resize(256),
            T.DenormalizeBbox(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    return filterable_view_cls(CLEVRDatasetUnwrapped(
        scenes_json, questions_json, image_root, image_transform, vocab_json, output_vocab_json, **kwargs
    ))


def make_custom_transfer_dataset(scenes_json, questions_json, image_root, query_list_key, custom_fields, *, image_transform=None, output_vocab_json=None, **kwargs):
    if image_transform is None:
        import jactorch.transforms.bbox as T
        image_transform = T.Compose([
            T.NormalizeBbox(),
            T.Resize(256),
            T.DenormalizeBbox(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    return CLEVRCustomTransferDataset(
        scenes_json, questions_json, image_root,
        query_list_key=query_list_key, custom_fields=custom_fields,
        image_transform=image_transform, output_vocab_json=output_vocab_json, **kwargs
    )


def annotate_scene(scene):
    feed_dict = dict()

    if not _is_object_annotation_available(scene):
        return feed_dict

    for attr_name, concepts in g_attribute_concepts.items():
        concepts2id = {v: i for i, v in enumerate(concepts)}
        values = list()
        for obj in scene['objects']:
            assert attr_name in obj
            values.append(concepts2id[obj[attr_name]])
        values = np.array(values, dtype='int64')
        feed_dict['attribute_' + attr_name] = values
        lhs, rhs = np.meshgrid(values, values)
        compare_label = (lhs == rhs).astype('float32')
        compare_label[np.diag_indices_from(compare_label)] = 0
        feed_dict['attribute_relation_' + attr_name] = compare_label.reshape(-1)

    nr_objects = len(scene['objects'])
    for attr_name, concepts in g_relational_concepts.items():
        concept_values = []
        for concept in concepts:
            values = np.zeros((nr_objects, nr_objects), dtype='float32')
            assert concept in scene['relationships']
            this_relation = scene['relationships'][concept]
            assert len(this_relation) == nr_objects
            for i, this_row in enumerate(this_relation):
                for j in this_row:
                    values[j, i] = 1
            concept_values.append(values)
        concept_values = np.stack(concept_values, -1)
        feed_dict['relation_' + attr_name] = concept_values.reshape(-1, concept_values.shape[-1])

    return feed_dict


def canonize_answer(answer, question_type):
    if answer in ('yes', 'no'):
        answer = (answer == 'yes')
    elif isinstance(answer, str) and answer.isdigit():
        answer = int(answer)
        assert 0 <= answer <= 10
    return answer


def annotate_objects(scene):
    if 'objects' not in scene and 'objects_detection' not in scene:
        return dict()

    boxes = [mask_utils.toBbox(i['mask']) for i in _get_object_masks(scene)]
    if len(boxes) == 0:
        return {'objects': np.zeros((0, 4), dtype='float32')}
    boxes = np.array(boxes)
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    return {'objects': boxes.astype('float32')}


def _is_object_annotation_available(scene):
    if len(scene['objects']) > 0 and 'mask' in scene['objects'][0]:
        return True
    return False


def _get_object_masks(scene):
    """Backward compatibility: in self-generated clevr scenes, the groundtruth masks are provided;
    while in the clevr test data, we use Mask R-CNN to detect all the masks, and stored in `objects_detection`."""
    if 'objects_detection' not in scene:
        return scene['objects']
    if _is_object_annotation_available(scene):
        return scene['objects']
    return scene['objects_detection']


g_last_op_to_question_type = {
    'query_color': 'query_attr',
    'query_shape': 'query_attr',
    'query_material': 'query_attr',
    'query_size': 'query_attr',
    'exist': 'exist',
    'count': 'count',
    'equal_integer': 'cmp_number',
    'greater_than': 'cmp_number',
    'less_than': 'cmp_number',
    'equal_color': 'cmp_attr',
    'equal_shape': 'cmp_attr',
    'equal_material': 'cmp_attr',
    'equal_size': 'cmp_attr',
}


def get_op_type(op):
    if 'type' in op:
        return op['type']
    return op['function']


def get_question_type(program):
    if program is None:
        return 'unk'
    return g_last_op_to_question_type[get_op_type(program[-1])]

