#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : llm_prompting.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/21/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Optional, List
from concepts.language.openai_utils.default_client import get_default_client, get_default_chat_model
from concepts.language.openai_utils.llm_prompting_utils import auto_retry, extract_tag


def simple_llm_prompt(
    system_instruction: str,
    user_prompts: List[str],
    task_prompt: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 2048,
    tag: Optional[str] = None
):
    """
    Generate a simple LLM prompt for the given system instruction and user prompts.

    Args:
        system_instruction: the system instruction.
        user_prompts: a list of user prompts.
        task_prompt: the task prompt. If provided, it will be appended to the user prompts.
        model: the model name.
        max_tokens: the maximum number of tokens to generate.
        tag: the tag to extract from the generated text. For example, 'python' can be used to extract contents inside
            <python></python> tags.

    Returns:
        The generated prompt.
    """

    client = get_default_client()
    response = client.chat.completions.create(
        model=model if model is not None else get_default_chat_model(),
        messages=[
            {'role': 'system', 'content': system_instruction},
            *[{'role': 'user', 'content': p} for p in user_prompts],
            *([{'role': 'user', 'content': task_prompt}] if task_prompt is not None else [])
        ],
        max_tokens=max_tokens
    )

    if tag is None:
        return {'response': response}

    extracted, exception = None, None
    try:
        extracted = extract_tag(response.choices[0].message.content, tag, unique=False)
    except Exception as e:
        exception = e

    return {'response': response, 'tag': tag, 'extracted': extracted, 'extraction_exception': exception}


simple_llm_prompt_auto_retry = auto_retry(5)(simple_llm_prompt)
