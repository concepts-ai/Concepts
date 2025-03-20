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
import openai

__all__ = ['get_default_client', 'get_default_chat_model', 'set_default_chat_model']


_DEFAULT_CLIENT = None


def get_default_client():
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
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
    return _DEFAULT_CLIENT


_DEFAULT_CHAT_MODEL = "gpt-3.5-turbo"


def get_default_chat_model():
    return _DEFAULT_CHAT_MODEL


def set_default_chat_model(model):
    global _DEFAULT_CHAT_MODEL
    _DEFAULT_CHAT_MODEL = model
