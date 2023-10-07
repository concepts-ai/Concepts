#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/23/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import re
import os.path as osp
from typing import Union, List, Dict


class ParsingFailedError(Exception):
    pass


def load_prompt(identifier) -> List[Dict[str, str]]:
    prompt_filename = osp.join(osp.dirname(__file__), 'prompts', identifier + '.txt')
    with open(prompt_filename, 'r') as f:
        content = f.read()
        return [
            {'role': 'system', 'content': content},
        ]


class TagNotUniqueError(ParsingFailedError):
    pass


def extract_tag(content: str, tag: str, unique: bool = False) -> Union[str, List[str]]:
    """Extract all matched content inside <tag></tag>.

    Args:
        content: the input string.
        tag: the tag name.
        unique: if True, only return the first matched content and raises an error.
    """

    pattern = r'<{}>(.*?)</{}>'.format(tag, tag)
    matches = re.findall(pattern, content, re.DOTALL)

    if unique:
        if len(matches) != 1:
            raise TagNotUniqueError('Tag "{}" is not unique. Content: {}'.format(tag, content))
        return matches[0]
    else:
        return matches

