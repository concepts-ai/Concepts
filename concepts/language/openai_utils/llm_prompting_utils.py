#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : llm_prompting_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/23/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import re
import os.path as osp
import functools
from typing import Optional, Union, List, Dict


class ParsingFailedError(Exception):
    pass


def load_prompt(identifier, filename: Optional[str] = None) -> List[Dict[str, str]]:
    if filename is None:
        filename = __file__
    prompt_filename = osp.join(osp.dirname(filename), 'prompts', identifier + '.txt')
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


def auto_retry(nr_retries: int):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(nr_retries):
                try:
                    return func(*args, **kwargs)
                except ParsingFailedError:
                    if i == nr_retries - 1:
                        raise
                except Exception as e:
                    if i == nr_retries - 1:
                        raise e
                    import traceback
                    traceback.print_exc()
        return wrapper

    return decorator
