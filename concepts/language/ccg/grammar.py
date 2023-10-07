#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : grammar.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/05/2020
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Data structures and algorithms for lexicon, grammar and parsing."""

import copy
import inspect
import contextlib
import itertools
from collections import defaultdict
from typing import Any, Optional, Union, Iterable, Tuple, List, Dict, Callable
from jacinle.utils.printing import indent_text, print_to_string

from concepts.dsl.dsl_types import FormatContext
from concepts.dsl.dsl_functions import Function
from concepts.dsl.value import Value
from concepts.dsl.expression import ConstantExpression, FunctionApplicationExpression
from concepts.dsl.function_domain import FunctionDomain
from concepts.language.ccg.composition import (
    CCGCompositionType, CCGCompositionSystem,
    CCGCompositionError, get_ccg_composition_context,
    CCGCompositionResult, CCGCoordinationImmNode
)
from concepts.language.ccg.syntax import CCGSyntaxType, CCGSyntaxSystem
from concepts.language.ccg.semantics import CCGSemantics, CCGSemanticsSugar


__all__ = [
    'CCGParsingError',
    'Lexicon', 'LexiconUnion',
    'CCGNode', 'compose_ccg_nodes', 'CCG'
]

_profile = getattr(__builtins__, 'profile', lambda x: x)


class CCGParsingError(Exception):
    """An exception raised when parsing fails."""
    pass


class Lexicon(object):
    """The data structure for lexicon entries."""

    def __init__(self, syntax: Optional[CCGSyntaxType], semantics: Optional[CCGSemantics], weight: float = 0, extra: Optional[Any] = None):
        """Initialize the lexicon entry.

        Args:
            syntax: the syntax of the lexicon entry.
            semantics: the semantics of the lexicon entry.
            weight: the weight of the lexicon entry.
            extra: the extra information stored at the lexicon entry. If not None, this will be a tuple of (word, word_index, lexicon_entry_index).
        """
        self.syntax = syntax
        self.semantics = semantics
        self.weight = weight
        self.extra = extra

    syntax: Optional[CCGSyntaxType]
    """The syntax of the lexicon entry."""

    semantics: Optional[CCGSemantics]
    """The semantics of the lexicon entry."""

    weight: float
    """The weight of the lexicon entry."""

    extra: Optional[Any]
    """The extra information stored at the lexicon entry."""

    def __str__(self) -> str:
        fmt = type(self).__name__ + '['
        fmt += 'syntax=' + str(self.syntax) + ', '
        fmt += 'semantics=' + indent_text(
            str(self.semantics.value)
        ).lstrip() + ', '
        fmt += 'weight=' + str(self.weight) + ''
        fmt += ']'
        return fmt

    def __repr__(self):
        return str(self)


class _LexiconUnionType(object):
    def __init__(self, *annotations):
        self.annotations = annotations


class _LexiconUnionTyping(object):
    def __getitem__(self, item):
        if isinstance(item, tuple) and not isinstance(item[0], tuple):
            return _LexiconUnionType(item)
        return _LexiconUnionType(*item)


LexiconUnion = _LexiconUnionTyping()
"""A syntax sugar that allows users to define several lexicon entries for a single word.
This is useful when the word has multiple meanings (see :meth:`CCG.define` for details)."""


class CCGNode(object):
    """A node in the CCG parsing tree."""

    def __init__(
        self,
        composition_system: CCGCompositionSystem,
        syntax: Optional[CCGSyntaxType], semantics: Optional[CCGSemantics], composition_type: CCGCompositionType,
        lexicon: Optional[Lexicon] = None,
        lhs: Optional['CCGNode'] = None, rhs: Optional['CCGNode'] = None,
        weight: Optional[float] = None
    ):
        """Construct a CCG node.

        Args:
            composition_system: the composition system.
            syntax: the syntax type.
            semantics: the semantics.
            composition_type: the composition type.
            lexicon: the lexicon (if the composition type is LEXICON).
            lhs: the left child of the node (for other types of composition).
            rhs: the right child of the node (for other types of composition).
            weight: the weight. If not given, it will be computed automatically (lhs + rhs + composition weight).
        """

        self.composition_system = composition_system
        self.syntax = syntax
        self.semantics = semantics
        self.composition_type = composition_type

        self.lexicon = lexicon
        self.lhs = lhs
        self.rhs = rhs

        self.weight = weight
        if self.weight is None:
            self.weight = self._compute_weight()

    def _compute_weight(self):
        if self.composition_type is CCGCompositionType.LEXICON:
            return self.lexicon.weight

        return self.lhs.weight + self.rhs.weight + self.composition_system.weights[self.composition_type]

    composition_system: CCGCompositionSystem
    """The composition system."""

    syntax: Optional[CCGSyntaxType]
    """The syntax type of the node."""

    semantics: Optional[CCGSemantics]
    """The semantics of the node."""

    composition_type: CCGCompositionType
    """The composition type of the node."""

    lexicon: Optional[Lexicon]
    """The lexicon (if the composition type is LEXICON)."""

    lhs: Optional['CCGNode']
    """The left child of the node (for other types of composition)."""

    rhs: Optional['CCGNode']
    """The right child of the node (for other types of composition)."""

    weight: float
    """The weight of the node."""

    @_profile
    def compose(self, rhs: 'CCGNode', composition_type: Optional[CCGCompositionType] = None) -> Union[CCGCompositionResult, 'CCGNode']:
        """Compose the current node with another node. This function will automatically guess the composition type.

        Args:
            rhs: the right node.
            composition_type: the composition type. If not given, it will be guessed automatically.

        Returns:
            The composition result. When the composition type is not given, it will be a :class:`CCGCompositionResult` object.
        """
        if composition_type is not None:
            try:
                ctx = get_ccg_composition_context()
                self.compose_check(rhs, composition_type)  # throws CCGCompositionError
                new_syntax, new_semantics = None, None
                if ctx.syntax:
                    new_syntax = self.syntax.compose(rhs.syntax, composition_type)
                if ctx.semantics:
                    new_semantics = self.semantics.compose(rhs.semantics, composition_type)
                node = self.__class__(
                    self.composition_system, new_syntax, new_semantics, composition_type,
                    lhs=self, rhs=rhs
                )
                return node
            except CCGCompositionError as e:
                raise e
        else:
            results = list()
            exceptions = list()

            composition_types = self.compose_guess(rhs)
            if composition_types is None:
                composition_types = self.composition_system.allowed_composition_types

            for composition_type in composition_types:
                try:
                    results.append((composition_type, self.compose(rhs, composition_type)))
                except CCGCompositionError as e:
                    exceptions.append(e)

            if len(results) == 1:
                return CCGCompositionResult(*results[0])
            elif len(results) == 0:
                with get_ccg_composition_context().exc():
                    fmt = f'Failed to compose CCGNodes {self} and {rhs}.\n'
                    fmt += 'Detailed messages are:\n'
                    for t, e in zip(composition_types, exceptions):
                        fmt += indent_text('Trying CCGCompositionType.{}:\n{}'.format(t.name, str(e))) + '\n'
                    raise CCGCompositionError(fmt.rstrip())
            else:
                with get_ccg_composition_context().exc():
                    fmt = f'Got ambiguous composition for CCGNodes {self} and {rhs}.\n'
                    fmt += 'Candidates are:\n'
                    for r in results:
                        fmt += indent_text('CCGCompositionType.' + str(r[0].name)) + '\n'
                    raise CCGCompositionError(fmt.rstrip())

    def compose_check(self, rhs: 'CCGNode', composition_type: CCGCompositionType):
        """Check if the current node can be composed with another node. If the check fails, a CCGCompositionError will be
        raised.

        Args:
            rhs: the right node.
            composition_type: the composition type.

        Raises:
            CCGCompositionError: if the check fails.
        """
        if (
            isinstance(self.syntax, CCGCoordinationImmNode) or
            isinstance(self.semantics, CCGCoordinationImmNode)
        ):
            raise CCGCompositionError('Can not make non-coordination composition for CCGCoordinationImmResult.')
        if (
            isinstance(self.syntax, CCGCoordinationImmNode) or
            isinstance(self.semantics, CCGCoordinationImmNode) or
            isinstance(rhs.syntax, CCGCoordinationImmNode) or
            isinstance(rhs.semantics, CCGCoordinationImmNode)
        ):
            if composition_type is not CCGCompositionType.COORDINATION:
                raise CCGCompositionError('Can not make non-coordination composition for CCGCoordinationImmResult.')

    def compose_guess(self, rhs: 'CCGNode') -> Optional[Tuple[CCGCompositionType]]:
        """Guess the composition type of the current node with another node.

        Args:
            rhs: the right node.

        Returns:
            The guessed composition type. If None, all composition types are allowed.
        """
        return None

    def linearize_lexicons(self) -> List[Lexicon]:
        """Linearize the lexicons of the node. It return a list of lexicons corresponding to the leaves of parsing tree.

        Returns:
            The list of lexicons.
        """
        if self.lexicon is not None:
            return [self.lexicon]
        return self.lhs.linearize_lexicons() + self.rhs.linearize_lexicons()

    def as_nltk_str(self) -> str:
        """Convert the node to a string in nltk format."""
        if self.composition_type is CCGCompositionType.LEXICON:
            if self.lexicon.extra is not None:
                meaning = str(self.lexicon.extra[0])
            else:
                with FormatContext(function_format_lambda=True).as_default():
                    meaning = str(self.semantics.value).replace('(', '{').replace(')', '}')
            return f'({str(self.syntax)} {meaning})'

        if self.composition_type is CCGCompositionType.COORDINATION:
            return f'({str(self.syntax)} {self.lhs.as_nltk_str()} {self.rhs.lhs.as_nltk_str()} {self.rhs.rhs.as_nltk_str()})'
        return f'({str(self.syntax)} {self.lhs.as_nltk_str()} {self.rhs.as_nltk_str()})'

    def format_nltk_tree(self) -> str:
        """Format the node as a nltk tree."""
        with print_to_string() as fmt:
            self.print_nltk_tree()
        return fmt.get()

    def print_nltk_tree(self):
        """Print the node as a nltk tree."""
        from nltk.tree import Tree
        parsing_nltk = Tree.fromstring(self.as_nltk_str())
        parsing_nltk.pretty_print()

    def __str__(self) -> str:
        fmt = type(self).__name__ + '[\n'
        fmt += '  syntax   : ' + str(self.syntax) + '\n'
        with FormatContext(function_format_lambda=True).as_default():
            fmt += '  semantics: ' + indent_text(str(self.semantics), indent_format=' ' * 13).lstrip() + '\n'
        fmt += '  weight   : ' + str(self.weight) + '\n'
        fmt += ']'
        return fmt

    def __repr__(self) -> str:
        return str(self)


def compose_ccg_nodes(lhs: CCGNode, rhs: CCGNode, composition_type: Optional[CCGCompositionType] = None) -> CCGCompositionResult:
    """Compose two CCG nodes. If the composition type is not specified, it will be guessed.

    Args:
        lhs: the left node.
        rhs: the right node.
        composition_type: the composition type.

    Returns:
        The composition result.
    """
    return lhs.compose(rhs, composition_type)


class CCG(object):
    """The data structure of a CCG grammar and the implementation for parsing."""

    def __init__(self, domain: FunctionDomain, syntax_system: CCGSyntaxSystem, composition_system: Optional[CCGCompositionSystem] = None):
        """Initialize the CCG parser.

        Args:
            domain: the function domain.
            syntax_system: the syntax system, containing primitive and conjunctive syntax types.
            composition_system: the composition system, containing allowed composition rules.
        """
        self.function_domain = domain
        self.syntax_system = syntax_system
        self.semantics_sugar = CCGSemanticsSugar(self.function_domain)
        self.composition_system = composition_system
        self.lexicon_entries = defaultdict(list)

        if self.composition_system is None:
            self.composition_system = CCGCompositionSystem.make_default()

    function_domain: FunctionDomain
    """The function domain."""

    syntax_system: CCGSyntaxSystem
    """The syntax system, containing primitive and conjunctive syntax types."""

    semantics_sugar: CCGSemanticsSugar
    """The semantics sugar, which is a helper function to convert lambda functions to a :class:`concepts.language.ccg.semantics.CCGSemantics` instance."""

    composition_system: CCGCompositionSystem
    """The composition system, containing allowed composition rules."""

    lexicons: Dict[str, List[Lexicon]]
    """The lexicons entries, which is a dictionary from word to a list of :class:`Lexicon` instances."""

    def clone(self, deep: bool = False) -> 'CCG':
        """Clone the CCG grammar.

        Args:
            deep: whether to make a deep copy of the class. If False, only the lexicons will be copied.

        Returns:
            The cloned CCG grammar.
        """
        if deep:
            return copy.deepcopy(self)
        new_obj = self.__class__(self.function_domain, self.syntax_system, self.composition_system)
        new_obj.lexicon_entries = defaultdict(list)
        for k, v in self.lexicon_entries.items():
            new_obj.lexicon_entries[k].extend(v)
        return new_obj

    def make_node(self, arg1: Union[str, Lexicon, CCGNode], arg2: Optional[CCGNode] = None, *, composition_type: Optional[CCGCompositionType] = None) -> CCGNode:
        """Make a CCG node by trying to compose two nodes or retriving an lexicon entry. There are three use cases:

        - ``make_node(lexicon)``: make a node from a :class:`Lexicon` instance.
        - ``make_node(word)``: make a node from a word, which will be looked up in the lexicon. When there are multiple lexicon entries, an error will be raised.
        - ``make_node(lhs, rhs)``: make a node by composing two nodes.

        Args:
            arg1: the first argument.
            arg2: the second argument.
            composition_type: the composition type. If not specified, it will be guessed.

        Returns:
            The composed CCG node.
        """
        if isinstance(arg1, str):
            if arg1 in self.lexicon_entries:
                tot_entries = len(self.lexicon_entries[arg1])
                if tot_entries == 1:
                    return self.make_node(self.lexicon_entries[arg1][0])
                raise CCGParsingError('Ambiguous lexicon entry for word: "{}" (n = {}).'.format(arg1, tot_entries))
            raise CCGParsingError('Out-of-vocab word: {}.'.format(arg1))

        if isinstance(arg1, Lexicon):
            return CCGNode(self.composition_system, arg1.syntax, arg1.semantics, CCGCompositionType.LEXICON, lexicon=arg1)
        else:
            assert arg2 is not None
            return compose_ccg_nodes(arg1, arg2, composition_type=composition_type).result

    @property
    def Syntax(self):
        """A syntax sugar for defining syntax types. For example, use ``self.Syntax['S/NP']`` to define a syntax type."""
        return self.syntax_system

    @property
    def Semantics(self):
        """A syntax sugar for defining semantics. For example, use ``self.Semantics[lambda x: f(x)]`` to define a semantics."""
        return self.semantics_sugar

    def add_entry(self, word: str, lexicon: Lexicon):
        """Add a lexicon entry.

        Args:
            word: the word.
            lexicon: the lexicon entry.
        """
        self.lexicon_entries[word].append(lexicon)

    def add_entry_simple(
        self, word: str,
        syntax: Union[None, str, CCGSyntaxType],
        semantics: Union[None, Value, Function, ConstantExpression, FunctionApplicationExpression, Callable],
        weight: float = 0
    ):
        """Add a lexicon entry with a syntax type and a semantic form.

        Args:
            word: the word.
            syntax: the syntax type. It can be a string, a :class:`CCGSyntaxType` instance, or None.
            semantics: the semantics. It can be a value, a function, a constant expression, a function application expression, or a callable Python function.
            weight: the weight of the lexicon entry.
        """
        self.lexicon_entries[word].append(Lexicon(self.Syntax[syntax], self.Semantics[semantics], weight=weight))

    def clear_entries(self, word: str):
        """Clear all lexicon entries for a word.

        Args:
            word: the word.
        """
        self.lexicon_entries[word].clear()

    def update_entries(self, entries_dict: Dict[str, Iterable[Lexicon]]):
        """Update the lexicon entries.

        Args:
            entries_dict: the lexicon entries, which is a dictionary from word to a list of :class:`Lexicon` instances.
        """
        for word, entries in entries_dict.items():
            for entry in entries:
                self.add_entry(word, entry)

    @contextlib.contextmanager
    def define(self):
        """A helper function to define a CCG grammar. For example:

        .. code-block:: python

            with ccg.define():
                red: 'N/N', lambda x: filter(x, 'red')
                blue: 'N/N', lambda x: filter(x, 'blue'), 0.5  # weight = 0.5
        """
        locals_before = inspect.stack()[2][0].f_locals.copy()
        annotations_before = locals_before.get('__annotations__', dict()).copy()
        yield self
        locals_after = inspect.stack()[2][0].f_locals.copy()
        annotations_after = locals_after.get('__annotations__', dict()).copy()

        new_annotations = {
            k: v for k, v in annotations_after.items()
            if k not in annotations_before or annotations_after[k] != annotations_before[k]
        }

        if len(new_annotations) == 0:
            raise ValueError('ccg.define() is only allowed at the global scope.')

        def add_entry(word, annotation):
            assert isinstance(annotation, tuple) and len(annotation) in (2, 3)
            assert isinstance(annotation[0], CCGSyntaxType)
            assert isinstance(annotation[1], CCGSemantics)

            weight = 0
            if len(annotation) == 2:
                syntax, semantics = annotation
            else:
                syntax, semantics, weight = annotation

            self.add_entry(word, Lexicon(syntax, semantics, weight))

        for const_name, raw_annotation in new_annotations.items():
            if isinstance(raw_annotation, _LexiconUnionType):
                for a in raw_annotation.annotations:
                    add_entry(const_name, a)
            else:
                add_entry(const_name, raw_annotation)

    def parse(self, sentence: Union[str, Iterable[str]], beam: Optional[int] = None, preserve_syntax_types: bool = True) -> List[CCGNode]:
        """Parse a sentence.

        Args:
            sentence: the sentence to parse. It can be a string or a list of words.
            beam: the beam size. If not specified, it will perform the full chart parsing (beam size = Infinity).
            preserve_syntax_types: whether to preserve the syntax types in beam searcm.
                When it is False, the algorithm will sort all nodes by their weights and return the top-k nodes.
                When it is True, the algorithm will sort all nodes by their weights and return the top-k nodes for each syntax type.

        Returns:
            A list of :class:`CCGNode` instances.
        """
        if isinstance(sentence, str):
            sentence = sentence.split()

        length = len(sentence)
        dp = [[list() for _ in range(length + 1)] for _ in range(length)]
        for i, word in enumerate(sentence):
            if word not in self.lexicon_entries:
                raise CCGParsingError('Out-of-vocab word: {}.'.format(word))

            dp[i][i + 1] = [self.make_node(length) for length in self.lexicon_entries[word]]

        def merge(list1, list2):
            output_list = list()
            for node1, node2 in itertools.product(list1, list2):
                try:
                    node = compose_ccg_nodes(node1, node2).result
                    output_list.append(node)
                except CCGCompositionError:
                    pass
            return output_list

        for length in range(2, length + 1):
            for i in range(0, length + 1 - length):
                j = i + length
                for k in range(i + 1, j):
                    dp[i][j].extend(merge(dp[i][k], dp[k][j]))
                if beam is not None:
                    if preserve_syntax_types:
                        dp[i][j] = _filter_beam_per_type(dp[i][j], beam)
                    else:
                        dp[i][j] = sorted(dp[i][j], key=lambda x: x.weight, reverse=True)[:beam]

        return sorted(dp[0][length], key=lambda x: x.weight)

    def _format_lexicon_entries(self) -> str:
        fmt = 'Lexicon Entries:\n'
        # max_words_len = max([len(x) for x in self.lexicons])
        for word, lexicons in self.lexicon_entries.items():
            for lexicon in lexicons:
                this_fmt = f'{word}: ' + str(lexicon)
                fmt += indent_text(this_fmt) + '\n'
        return fmt

    def format_summary(self) -> str:
        """Format the summary of the CCG grammar."""
        fmt = 'Combinatory Categorial Grammar\n'
        fmt += indent_text(str(self.function_domain)) + '\n'
        fmt += indent_text(str(self.syntax_system)) + '\n'
        fmt += indent_text(str(self.composition_system)) + '\n'
        fmt += indent_text(self._format_lexicon_entries())
        return fmt

    def print_summary(self):
        """Print the summary of the CCG grammar."""
        print(self.format_summary())


def _filter_beam_per_type(nodes: List[CCGNode], beam: int) -> List[CCGNode]:
    all_nodes_by_type = defaultdict(list)
    for node in nodes:
        if isinstance(node.syntax, CCGCoordinationImmNode):
            typename = f'COORDIMM({node.syntax.conj.typename}, {node.syntax.rhs.typename})'
        else:
            typename = node.syntax.typename
        all_nodes_by_type[typename].append(node)
    for typename, nodes in all_nodes_by_type.items():
        all_nodes_by_type[typename] = sorted(nodes, key=lambda x: x.weight, reverse=True)[:beam]
    return list(itertools.chain.from_iterable(all_nodes_by_type.values()))

