#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/04/2020
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Tools for parsing sentences with (purely symbolic) Combinatory Categorial Grammars."""

from . import learning
from .composition import CCGCompositionDirection, CCGCompositionType, CCGCompositionContext, get_ccg_composition_context, CCGCompositionResult, CCGComposable, CCGCompositionSystem
from .syntax import CCGSyntaxType, CCGPrimitiveSyntaxType, CCGConjSyntaxType, CCGComposedSyntaxType, CCGSyntaxSystem
from .semantics import CCGSemanticsConjFunction, CCGSemanticsSimpleConjFunction, CCGSemanticsLazyValue, CCGSemantics
from .grammar import Lexicon, LexiconUnion, CCGNode, compose_ccg_nodes, CCG

__all__ = [
    'CCGCompositionDirection', 'CCGCompositionType',
    'CCGCompositionContext', 'get_ccg_composition_context',
    'CCGCompositionResult', 'CCGComposable',
    'CCGCompositionSystem',
    'CCGSyntaxType', 'CCGPrimitiveSyntaxType', 'CCGConjSyntaxType', 'CCGComposedSyntaxType',
    'CCGSyntaxSystem',
    'CCGSemanticsConjFunction', 'CCGSemanticsSimpleConjFunction',
    'CCGSemanticsLazyValue', 'CCGSemantics',
    'Lexicon', 'LexiconUnion', 'CCGNode', 'compose_ccg_nodes', 'CCG',
    'learning'
]

