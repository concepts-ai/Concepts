#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : learning.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/06/2020
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Algorithms for learning CCG entries from exmaples."""

import itertools
import collections
from typing import Any, Optional, Union, Iterable, Tuple, List, Callable
from dataclasses import dataclass
from jacinle.utils.printing import indent_text

from concepts.dsl.dsl_types import ConstantType
from concepts.dsl.dsl_functions import Function
from concepts.dsl.value import Value
from concepts.dsl.expression import ConstantExpression, FunctionApplicationExpression
from concepts.dsl.executors.executor_base import DSLExecutionError
from concepts.language.ccg.composition import CCGCompositionContext, CCGCompositionError
from concepts.language.ccg.semantics import CCGSemantics
from concepts.language.ccg.grammar import CCG, Lexicon, CCGNode
from concepts.language.ccg.search import CCGSyntaxSearcherBase, CCGSyntaxEnumerativeSearcher, CCGSemanticsSearcherBase, CCGSemanticsEnumerativeSearcher
from concepts.language.ccg.search import CCGSyntaxSearchResult, CCGSemanticsSearchResult

__all__ = ['CCGLearningResult', 'by_parsing', 'by_parsing_with_lexicon_gen', 'by_grounding']


@dataclass
class CCGLearningResult(object):
    """The result of CCG learning."""

    words: Tuple[str, ...]
    """The list of words learned by the algorithm."""

    lexicons: Tuple[Lexicon, ...]
    """The list of lexicons learned by the algorithm. It is a tuple of lexicons that has the same length as `words`."""

    parsing_results: Tuple[CCGNode, ...]
    """The parsing results of the entire sentence based on the learned lexicons."""

    def format_summary(self):
        fmt = 'Learned lexicons:\n'
        for w, l in zip(self.words, self.lexicons):
            fmt += indent_text(w + ': ' + str(l)) + '\n'
        fmt += 'Parsing result:\n'
        for i, r in enumerate(self.parsing_results):
            fmt += f'{i}: ' + indent_text(str(r), indent_format='    ').lstrip() + '\n'
        return fmt

    def print_summary(self):
        print(self.format_summary())


def by_parsing(
    ccg: CCG, sentence: Union[str, Iterable[str]], *,
    novel_words: Optional[tuple] = None,
    candidate_syntax_types: Optional[List[CCGSyntaxSearchResult]] = None,
    syntax_searcher: Optional[CCGSyntaxSearcherBase] = None,
    syntax_searcher_kwargs: Optional[dict] = None,
    candidate_semantics: Optional[List[CCGSemanticsSearchResult]] = None,
    semantics_searcher: Optional[CCGSemanticsSearcherBase] = None,
    semantics_searcher_kwargs: Optional[dict] = None,
    bind_concepts: bool = True
) -> List[CCGLearningResult]:
    """Learn CCG lexicon entries from a sentence by trying to parse the sentence.

    Args:
        ccg: the CCG grammar.
        sentence: the sentence to be parsed.
        novel_words: the list of novel words to be learned. If not specified, the algorithm will detect all novel words in the sentence.
        candidate_syntax_types: the list of candidate syntax types to be used for parsing.
            If not specified, the algorithm will use the syntax searcher to generate the candidate syntax types.
        syntax_searcher: the syntax searcher to be used for generating candidate syntax types.
            If not specified, the algorithm will use the enumerative searcher.
        syntax_searcher_kwargs: the keyword arguments for the syntax searcher.
        candidate_semantics: the list of candidate semantics to be used for parsing.
            If not specified, the algorithm will use the semantics searcher to generate the candidate semantics.
        semantics_searcher: the semantics searcher to be used for generating candidate semantics.
            If not specified, the algorithm will use the enumerative searcher.
        semantics_searcher_kwargs: the keyword arguments for the semantics searcher.
        bind_concepts: whether to bind concepts in the semantics. This will allow algorithm to invent novel concepts while learning.

    Returns:
        The result of the learning process, as a list of :class:`CCGLearningResult`.
    """

    if isinstance(sentence, str):
        sentence = sentence.split()

    if candidate_syntax_types is None:
        if syntax_searcher is None:
            syntax_searcher = CCGSyntaxEnumerativeSearcher(ccg.syntax_system)
        if syntax_searcher_kwargs is None:
            syntax_searcher_kwargs = dict()
        candidate_syntax_types = syntax_searcher.gen(**syntax_searcher_kwargs)

    if candidate_semantics is None:
        if semantics_searcher is None:
            semantics_searcher = CCGSemanticsEnumerativeSearcher(ccg.function_domain)
        if semantics_searcher_kwargs is None:
            semantics_searcher_kwargs = dict()
        candidate_semantics = semantics_searcher.gen(**semantics_searcher_kwargs)

    semantics_search_results_arity = collections.defaultdict(list)
    for r in candidate_semantics:
        if not bind_concepts and r.nr_constant_arguments > 0:
            continue  # if we do not allow binding new concepts, we skip those semantics with constant arguments.
        semantics_search_results_arity[r.nr_variable_arguments].append(r.semantics)

    if novel_words is None:
        novel_words = set()
        for word in sentence:
            if word not in ccg.lexicon_entries or len(ccg.lexicon_entries[word]) == 0:
                novel_words.add(word)
        novel_words = tuple(novel_words)

    new_ccg = ccg.clone()

    success_syntax_types = list()
    for syntax_types in itertools.product(candidate_syntax_types, repeat=len(novel_words)):
        for word, syntax in zip(novel_words, syntax_types):
            new_ccg.add_entry(word, Lexicon(syntax.syntax, None))

        try:
            with CCGCompositionContext(semantics=False, exc_verbose=False).as_default():
                parsing_results = new_ccg.parse(sentence)  # TODO: maybe cache the results.
            if len(parsing_results) > 0:
                # TODO: add a flag to control this check.
                found_success = False
                for r in parsing_results:
                    if r.syntax.typename in syntax_searcher._starting_symbols:
                        found_success = True
                        break
                if found_success:
                    success_syntax_types.append(syntax_types)
        except CCGCompositionError:
            pass

        for word in novel_words:
            new_ccg.clear_entries(word)

    success_results = list()
    for syntax_types in success_syntax_types:
        for semantics_list in itertools.product(*[
            semantics_search_results_arity[syntax.syntax.arity]
            for syntax in syntax_types
        ]):
            this_lexicons = list()
            for word, syntax, semantics in zip(novel_words, syntax_types, semantics_list):
                syntax, semantics = syntax.syntax, semantics
                if bind_concepts:
                    semantics = _bind_concepts(semantics, word)
                lexicon = Lexicon(syntax, semantics)
                new_ccg.add_entry(word, lexicon)
                this_lexicons.append(lexicon)

            try:
                with CCGCompositionContext(exc_verbose=True).as_default():
                    parsing_results = new_ccg.parse(sentence)
                if len(parsing_results) > 0:
                    success_results.append(CCGLearningResult(novel_words, tuple(this_lexicons), parsing_results))
            except CCGCompositionError:
                pass

            for word in novel_words:
                new_ccg.clear_entries(word)

    return success_results


def _bind_concepts(semantics: CCGSemantics, word: str):
    if semantics.is_function:
        mapping = dict()
        for i, argument in enumerate(semantics.value.arguments):
            if isinstance(argument.dtype, ConstantType):
                mapping[f'#{i}'] = Value(argument.dtype, word)
        if len(mapping) > 0:
            semantics = CCGSemantics(semantics.value.partial(**mapping, execute_fully_bound_functions=True), is_conj=semantics.is_conj)
    return semantics


def by_parsing_with_lexicon_gen(
    ccg: CCG, sentence: Union[str, list, tuple],
    lexicon_generator: Callable[[str], Iterable[Lexicon]],
    novel_words: Optional[tuple] = None
):
    """Learn CCG lexicon entries from a sentence by trying to parse the sentence.
    Unlike :func:`by_parsing`, this function takes a lexicon generator instead of a list of candidate syntax types and semantics.

    Args:
        ccg: the CCG grammar.
        sentence: the sentence to be parsed.
        lexicon_generator: the lexicon generator to be used for generating candidate lexicons. It takes a word as input and returns a list of candidate lexicons.
        novel_words: the list of novel words to be learned. If not specified, the algorithm will detect all novel words in the sentence.

    Returns:
        The result of the learning process, as a list of :class:`CCGLearningResult`.
    """

    if isinstance(sentence, str):
        sentence = sentence.split()

    if novel_words is None:
        novel_words = set()
        for word in sentence:
            if word not in ccg.lexicon_entries or len(ccg.lexicon_entries[word]) == 0:
                novel_words.add(word)
    novel_words = tuple(novel_words)

    new_ccg = ccg.clone()
    candidate_lexicons = [lexicon_generator(word) for word in novel_words]

    success_results = list()
    # TODO(Jiayuan Mao @ 07/11): optimize by first filtering syntax types.
    for new_lexicon in itertools.product(*candidate_lexicons):
        for word, entry in zip(novel_words, new_lexicon):
            new_ccg.add_entry(word, entry)

        try:
            with CCGCompositionContext(exc_verbose=False).as_default():
                parsing_results = new_ccg.parse(sentence)
            if len(parsing_results) > 0:
                success_results.append(CCGLearningResult(novel_words, tuple(new_lexicon), parsing_results))
        except CCGCompositionError:
            pass

        for word in novel_words:
            new_ccg.clear_entries(word)

    return success_results


def by_grounding(
    by_parsing_learning_func: Callable,
    ccg: CCG, sentence: Union[str, Iterable[str]],
    executor: Callable[[Union[Function, ConstantExpression, FunctionApplicationExpression]], Any],
    criterion: Callable[[Any], bool],
    **by_parsing_kwargs
) -> List[CCGLearningResult]:
    """Learn a CCG lexicon from a sentence by both parsing the sentence and grounding the parsing result with a given executor.

    Args:
        by_parsing_learning_func: the function to be used for learning a CCG lexicon from a sentence by parsing the sentence.
            Can be either :func:`by_parsing` or :func:`by_parsing_with_lexicon_gen`.
        ccg: the CCG grammar.
        sentence: the sentence to be parsed.
        executor: the executor used to ground the parsing result.
        criterion: the criterion used to classify whether the execution result is correct.
        **by_parsing_kwargs: the keyword arguments to be passed to the ``by_parsing_learning_func``.

    Returns:
        The result of the learning process, as a list of :class:`CCGLearningResult`.
    """
    by_parsing_results = by_parsing_learning_func(ccg, sentence, **by_parsing_kwargs)

    success_results = list()
    for r in by_parsing_results:
        ccg_node = r.parsing_results[0]
        try:
            result = executor(ccg_node.expression.value)
            if criterion(result):
                success_results.append(r)
        except DSLExecutionError:
            pass
    return success_results


def auto_research_novel_words(
    base_learning_func: Callable,
    ccg: CCG,
    sentence: Union[str, Iterable[str]],
    max_research: int,
    **kwargs
) -> List[CCGLearningResult]:
    """A helper function that automatically perform re-search for known words in a sentence.

    Args:
        base_learning_func: the base learning function to be used for learning a CCG lexicon from a sentence.
            Can be :func:`by_parsing`, or :func:`by_parsing_with_lexicon_gen`, or :func:`by_grounding`.
        ccg: the CCG grammar.
        sentence: the sentence to be parsed.
        max_research: the maximum number of words whose lexicon will be re-searched.
        **kwargs: additional keyword arguments to be passed to the ``base_learning_func``.
    """
    if isinstance(sentence, str):
        sentence = sentence.split()

    novel_words = set()
    known_words = set()
    for word in sentence:
        if word not in ccg.lexicon_entries or len(ccg.lexicon_entries[word]) == 0:
            novel_words.add(word)
        else:
            known_words.add(word)
    novel_words_tuple = tuple(novel_words)
    known_words_tuple = tuple(known_words)

    rv = base_learning_func(ccg, sentence, **kwargs)
    if len(rv) > 0:
        return rv

    kwargs = kwargs.copy()
    for nr_research in range(1, max_research + 1):
        rv_accumulate = list()
        for comb in itertools.combinations(known_words_tuple, nr_research):
            new_novel_words = novel_words_tuple + tuple(comb)
            kwargs['novel_words'] = new_novel_words
            rv = base_learning_func(ccg, sentence, **kwargs)
            rv_accumulate.extend(rv)
        if len(rv_accumulate) > 0:
            return rv_accumulate

    return list()

