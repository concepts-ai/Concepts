#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : default_client.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/21/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os

__all__ = ['get_default_client', 'get_default_chat_model', 'set_default_chat_model']

_DEFAULT_MODEL_FAMILY = 'openai'
_DEFAULT_CLIENT = None
_DEFAULT_CHAT_MODEL = "gpt-3.5-turbo"


def get_default_model_family():
    """Get the default model family."""
    return _DEFAULT_MODEL_FAMILY


def set_default_model_family(model_family):
    """Set the default model family."""
    global _DEFAULT_MODEL_FAMILY
    if model_family not in ['openai', 'anthropic']:
        raise ValueError(f"Unsupported model family: {model_family}")
    _DEFAULT_MODEL_FAMILY = model_family


def get_default_client():
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:

        if get_default_model_family() == 'openai':
            import openai
            if 'OPENAI_API_KEY' not in os.environ:
                raise ValueError('Please set the OPENAI_API_KEY environment variable')

            if 'OPENAI_ORGANIZATION_KEY' in os.environ:
                _DEFAULT_CLIENT = openai.OpenAI(
                    organization=os.environ['OPENAI_ORGANIZATION_KEY'],
                    project=os.environ.get('OPENAI_PROJECT_KEY', None),
                    api_key=os.environ['OPENAI_API_KEY']
                )
            else:
                _DEFAULT_CLIENT = openai.OpenAI()
        elif get_default_model_family() == 'anthropic':
            import anthropic
            _DEFAULT_CLIENT = anthropic.Anthropic()

    if _DEFAULT_CLIENT is None:
        raise ValueError('Default client is not set. This is probably due to an unrecognized model family.')

    return _DEFAULT_CLIENT


def get_default_chat_model():
    return _DEFAULT_CHAT_MODEL


def set_default_chat_model(model):
    global _DEFAULT_CHAT_MODEL
    _DEFAULT_CHAT_MODEL = model
