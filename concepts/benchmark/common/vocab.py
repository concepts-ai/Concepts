#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : vocab.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/26/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Optional, Union, Iterable, Sequence, List

import torch

import jacinle.io as io
import jaclearn.embedding.constant as const
from jacinle.utils.tqdm import tqdm

__all__ = ['Vocab', 'gen_vocab', 'gen_vocab_from_words']


class Vocab(object):
    """A simple vocabulary class."""

    def __init__(self, word2idx=None):
        """Initialize the vocabulary.

        Args:
            word2idx: a dictionary mapping words to indices. If not specified, the vocabulary will be empty.
        """
        self.word2idx = word2idx if word2idx is not None else dict()
        self._idx2word = None

    @classmethod
    def from_json(cls, json_file: str) -> 'Vocab':
        """Load a vocabulary from a json file."""
        return cls(io.load_json(json_file))

    @classmethod
    def from_dataset(cls, dataset, keys: Sequence[str], extra_words: Optional[Sequence[str]] = None, single_word: bool = False) -> 'Vocab':
        """Generate a vocabulary from a dataset.

        Args:
            dataset: the dataset to generate the vocabulary from.
            keys: the keys to retrieve from the dataset items.
            extra_words: additional words to add to the vocabulary.
            single_word: whether to treat the values of the keys as single words.
        """
        return gen_vocab(dataset, keys, extra_words=extra_words, cls=cls, single_word=single_word)

    @classmethod
    def from_list(cls, dataset: list, extra_words: Optional[Sequence[str]] = None, single_word: bool = False) -> 'Vocab':
        """Generate a vocabulary from a list of strings.

        Args:
            dataset: the list of strings to generate the vocabulary from.
            extra_words: additional words to add to the vocabulary.
            single_word: whether to treat the values of the keys as single words.
        """
        return gen_vocab(dataset, extra_words=extra_words, cls=cls, single_word=single_word)

    def dump_json(self, json_file: str):
        """Dump the vocabulary to a json file."""
        io.dump_json(json_file, self.word2idx)

    def check_json_consistency(self, json_file: str) -> bool:
        """Check whether the vocabulary is consistent with a json file."""
        rhs = io.load_json(json_file)
        for k, v in self.word2idx.items():
            if not (k in rhs and rhs[k] == v):
                return False
        return True

    def words(self) -> Iterable[str]:
        return self.word2idx.keys()

    @property
    def idx2word(self) -> dict:
        """A dictionary mapping indices to words. This is a lazy property. It will be automatically recomputed when the length of the vocabulary changes."""
        if self._idx2word is None or len(self.word2idx) != len(self._idx2word):
            self._idx2word = {v: k for k, v in self.word2idx.items()}
        return self._idx2word

    def __len__(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.word2idx)

    def __iter__(self) -> Iterable[str]:
        """Return an iterator over the words in the vocabulary."""
        return iter(self.word2idx.keys())

    def add(self, word: str):
        """Add a word to the vocabulary. Alias of :meth:`add_word`."""
        self.add_word(word)

    def add_word(self, word: str):
        """Add a word to the vocabulary."""
        self.word2idx[word] = len(self.word2idx)

    def map(self, word: str) -> int:
        """Map a word to its index. If the word is not in the vocabulary, return the index of the unknown token."""
        return self.word2idx.get(
            word,
            self.word2idx.get(const.EBD_UNKNOWN, -1)
        )

    def map_sequence(self, sequence: Sequence[str], add_be: bool = False) -> List[int]:
        """Map a sequence of words to a sequence of indices. If the argument `add_be` is True, the begin-of-sentence and end-of-sentence tokens will be added to the sequence.
        
        Args:
            sequence: the sequence of words to map.
            add_be: whether to add the begin-of-sentence and end-of-sentence tokens to the sequence.

        Returns:
            a list of indices.
        """
        if isinstance(sequence, str):
            sequence = sequence.split()
        sequence = [self.map(w) for w in sequence]
        if add_be:
            sequence.insert(0, self.word2idx[const.EBD_BOS])
            sequence.append(self.word2idx[const.EBD_EOS])
        return sequence

    def map_fields(self, feed_dict: dict, fields: Sequence[str]) -> dict:
        """Map the content in a specified set of fields in a dictionary to indices. The argument `fields` is a list of keys in the dictionary to map.
        This function will modify the dictionary in-place.

        Args:
            feed_dict: the dictionary of fields to map.
            fields: the list of keys to map.

        Returns:
            a dictionary of mapped fields.
        """
        feed_dict = feed_dict.copy()
        for k in fields:
            if k in feed_dict:
                feed_dict[k] = self.map(feed_dict[k])
        return feed_dict

    def invmap_sequence(self, sequence: Union[Sequence[int], torch.Tensor], proc_be: bool = False) -> List[str]:
        """Map a sequence of indices to a sequence of words. If the argument `proc_be` is True, the begin-of-sentence and end-of-sentence tokens will be removed from the sequence.

        Args:
            sequence: the sequence of indices to map.
            proc_be: whether to remove the begin-of-sentence and end-of-sentence tokens from the sequence.

        Returns:
            a list of words.
        """
        if torch.is_tensor(sequence):
            sequence = sequence.detach().cpu().tolist()
        str_sequence = [self.idx2word[int(x)] for x in sequence]
        if proc_be:
            if str_sequence[0] == const.EBD_BOS:
                str_sequence = str_sequence[1:]
            if str_sequence[-1] == const.EBD_EOS:
                str_sequence = str_sequence[:-1]
        return str_sequence


def gen_vocab(dataset: Sequence, keys: Optional[Iterable[str]] = None, extra_words: Optional[Iterable[str]] = None, cls: type = None, single_word: bool = False):
    """Generate a Vocabulary instance from a dataset.

    By default, this function will retrieve the data using the `get_metainfo` function,
    or it will fall back to `dataset[i]` if the function does not exist.

    The function should return a dictionary. Users can specify a list of keys that will
    be returned by the `get_metainfo` function. This function will split the string indexed
    by these keys and add tokens to the vocabulary.
    If the argument `keys` is not specified, this function assumes the return of `get_metainfo`
    to be a string.

    By default, this function will add four additional tokens:
    EBD_PAD, EBD_BOS, EBD_EOS, and EBD_UNK. Users can specify additional extra tokens using the
    extra_words argument.

    Args:
        dataset: the dataset to generate the vocabulary from. It can be a list of strings or a dataset instance.
        keys: the keys to retrieve from the dataset items. If not specified, the dataset is assumed to be a list of strings.
        extra_words: additional words to add to the vocabulary.
        cls: the class of the Vocabulary instance to generate.
        single_word: whether to treat the entries in the dataset as single words. Default to False. When set to False, the entries should either be a list of
            strings or a single string (in which case it will be split by spaces).
    """
    if cls is None:
        cls = Vocab

    all_words = set()
    for i in tqdm(len(dataset), desc='Building the vocab'):
        if hasattr(dataset, 'get_metainfo'):
            metainfo = dataset.get_metainfo(i)
        else:
            metainfo = dataset[i]

        if keys is None:
            for w in metainfo.split():
                all_words.add(w)
        else:
            for k in keys:
                if single_word:
                    all_words.add(str(metainfo[k]))
                elif isinstance(metainfo[k], str):
                    for w in metainfo[k].split():
                        all_words.add(w)
                else:
                    for w in metainfo[k]:
                        all_words.add(w)

    vocab = cls()
    vocab.add(const.EBD_ALL_ZEROS)
    for w in sorted(all_words):
        vocab.add(w)
    for w in [const.EBD_UNKNOWN, const.EBD_BOS, const.EBD_EOS]:
        vocab.add(w)

    if extra_words is not None:
        for w in extra_words:
            vocab.add(w)

    return vocab


def gen_vocab_from_words(words: Sequence[str], extra_words: Optional[Iterable[str]] = None, cls: type = None):
    """Generate a Vocabulary instance from a list of words."""
    if cls is None:
        cls = Vocab
    vocab = cls()
    vocab.add(const.EBD_ALL_ZEROS)
    for w in sorted(words):
        vocab.add(w)
    for w in [const.EBD_UNKNOWN, const.EBD_BOS, const.EBD_EOS]:
        vocab.add(w)
    if extra_words is not None:
        for w in extra_words:
            vocab.add(w)
    return vocab

