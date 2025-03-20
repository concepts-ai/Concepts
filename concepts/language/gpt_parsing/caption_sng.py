#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : caption_sng.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/23/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from dataclasses import dataclass
from typing import Any, Tuple, Dict
from concepts.language.openai_utils.default_client import get_default_client
from concepts.language.openai_utils.llm_prompting_utils import TagNotUniqueError, load_prompt, extract_tag


@dataclass
class CaptionSceneGraph(object):
    objects: Tuple[str]
    relations: Tuple[Tuple[str, str, str]]


class CaptionSNGParser(object):
    def __init__(self, max_tokens: int = 1024):
        self.prompt = load_prompt('gpt-35-turbo-chat-caption', __file__)
        self.max_tokens = max_tokens

    def parse(self, sentence: str) -> Dict[str, Any]:
        client = get_default_client()
        response = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                self.prompt[0],  # the system prmopt
                {'role': 'user', 'content': sentence}
            ],
            max_tokens=self.max_tokens
        )

        parsing = None
        exception = None
        try:
            parsing = self.extract(response.choices[0].message.content)
        except TagNotUniqueError as e:
            exception = e

        return {
            'sentence': sentence,
            'raw_response': response,
            'parsing': parsing,
            'exception': exception,
        }

    def parse_batch(self, sentences):
        # TODO(Jiayuan Mao @ 2023/04/23): support batchified parsing.
        raise NotImplementedError()

    def extract(self, raw_response):
        # extract all objects
        objects = extract_tag(raw_response, 'object', unique=False)

        # extract all relations
        raw_relations = extract_tag(raw_response, 'relation', unique=False)

        objects = set(objects)
        relations = list()
        for relation in raw_relations:
            this_subject = extract_tag(relation, 'subject', unique=True)
            this_predicate = extract_tag(relation, 'predicate', unique=True)
            this_object = extract_tag(relation, 'object', unique=True)

            if this_subject not in objects:
                objects.add(this_subject)
            if this_object not in objects:
                objects.add(this_object)
            relations.append((this_subject, this_predicate, this_object))

        objects = tuple(sorted(objects))
        relations = tuple(sorted(relations))
        return CaptionSceneGraph(objects, relations)


default_caption_sng_parser = CaptionSNGParser()


def parse_caption(sentence: str) -> Dict[str, Any]:
    return default_caption_sng_parser.parse(sentence)


if __name__ == '__main__':
    print(parse_caption('A little girl and a woman are having their picture taken in front of a desert.'))

