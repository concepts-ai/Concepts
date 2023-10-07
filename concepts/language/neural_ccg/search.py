#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : search.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/13/2020
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Algorithms for enumerate possible lexicon entries in Neural CCG."""

from dataclasses import dataclass
from typing import Any, Optional, Union, Iterable, Tuple, List, Callable

import jacinle
import numpy as np
from scipy.optimize import linear_sum_assignment

from concepts.dsl.dsl_types import TypeBase
from concepts.dsl.dsl_functions import FunctionType, Function
from concepts.dsl.expression import ConstantExpression, FunctionApplicationExpression
from concepts.dsl.value import Value
from concepts.dsl.executors.function_domain_executor import FunctionDomainExecutor
from concepts.dsl.learning.function_domain_search import FunctionDomainExpressionSearchResult
from concepts.language.ccg.syntax import CCGSyntaxType, CCGPrimitiveSyntaxType, CCGComposedSyntaxType
from concepts.language.ccg.search import stat_function
from concepts.language.neural_ccg.grammar import parse_linearization_string, NeuralCCGSyntaxType
from concepts.language.neural_ccg.grammar import NeuralCCGGroundingFunction

__all__ = [
    'NeuralCCGLexiconSearchResult',
    'NeuralCCGLexiconSearcherBase',
    'NeuralCCGLexiconEnumerativeSearcher',
    'NeuralCCGLexiconEnumerativeSearcherWithSyntax',
    'gen_lexicon_search_results_from_functions',
    'gen_lexicon_search_results_from_syntax_and_semantics'
]

logger = jacinle.get_logger(__file__)


@dataclass
class NeuralCCGLexiconSearchResult(object):
    """A data structure representing the search result of a neural CCG lexicon searcher."""

    syntax: NeuralCCGSyntaxType
    """The syntax type of the lexicon."""

    semantics: Union[None, ConstantExpression, Function, FunctionApplicationExpression]
    """The semantic form of the lexicon."""

    executor: Optional[NeuralCCGGroundingFunction]
    """Optionally, the executor for the semantic form."""


class NeuralCCGLexiconSearcherBase(object):
    """The base class for neural CCG lexicon searchers."""

    def gen(self) -> List[NeuralCCGLexiconSearchResult]:
        """Generate a list of candidate lexicons.

        Returns:
            A list of candidate lexicons, as a list of :class:`NeuralCCGLexiconSearchResult`.
        """
        raise NotImplementedError()


class NeuralCCGLexiconEnumerativeSearcher(NeuralCCGLexiconSearcherBase):
    """An enumerative neural CCG lexicon searcher. This function takes a list of candidate function expressions, and generates a list of candidate lexicon entries."""

    def __init__(self, candidate_expressions: List[FunctionDomainExpressionSearchResult], executor: FunctionDomainExecutor):
        """Initialize the searcher.

        Args:
            candidate_expressions: a list of candidate function expressions (including constants, functions, and function applications).
            executor: the executor for the semantic forms.
        """
        self._candidate_expressions = candidate_expressions
        self._executor = executor

    def gen(
        self,
        init_executor: bool = True,
        allow_none_lexicon: bool = False,
        permute_arguments: bool = False,
        semantics_hash: Callable[[Union[Value, Function, FunctionApplicationExpression]], Any] = None
    ) -> List[NeuralCCGLexiconSearchResult]:
        def gen() -> Iterable[Tuple[NeuralCCGSyntaxType, Union[None, ConstantExpression, Function, FunctionApplicationExpression]]]:
            for result in self._candidate_expressions:
                expression = result.expression
                if isinstance(expression, ConstantExpression):
                    syntax_type = NeuralCCGSyntaxType(expression.return_type)
                    yield syntax_type, expression
                if isinstance(expression, FunctionApplicationExpression):
                    syntax_type = NeuralCCGSyntaxType(expression.return_type)
                    yield syntax_type, expression
                elif isinstance(expression, Function):
                    for syntax_type, sub_function in NeuralCCGSyntaxType.iter_from_function(expression, result.nr_variable_arguments):
                        yield syntax_type, sub_function
                else:
                    raise TypeError(f'Invalid semantics: {expression}.')

            if allow_none_lexicon:
                yield NeuralCCGSyntaxType(None), None

        results = gen_lexicon_search_results_from_syntax_and_semantics(
            gen(), self._executor, init_executor=init_executor
        )
        results = self._unique_semantics(results, semantics_hash)
        return results

    def _unique_semantics(self, input_list: List[NeuralCCGLexiconSearchResult], hash_func: Optional[Callable[[Union[Value, Function, FunctionApplicationExpression]], Any]] = None) -> List[NeuralCCGLexiconSearchResult]:
        if hash_func is None:
            hash_func = str

        output_list = list()
        hash_set = set()
        for r in input_list:
            try:
                lex_hash = (str(r.syntax), hash_func(r.semantics))
                if lex_hash not in hash_set:
                    hash_set.add(lex_hash)
                    output_list.append(r)
            except TypeError:
                output_list.append(r)
        return output_list


class NeuralCCGLexiconEnumerativeSearcherWithSyntax(NeuralCCGLexiconSearcherBase):
    """An enumerative neural CCG lexicon searcher. This function also searches for the linguistic syntax type of the semantic form."""

    def __init__(
        self,
        candidate_expressions: List[FunctionDomainExpressionSearchResult],
        allowed_syntax_types: List[CCGSyntaxType],
        executor: FunctionDomainExecutor,
        primitive_syntax_compatible=Callable[[TypeBase, CCGPrimitiveSyntaxType], bool],
    ):
        """Initialize the searcher.

        Args:
            candidate_semantics (Union[list, CCGSemanticsSearcherBase]): the semantics searcher.
            allowed_syntax_types (List[CCGSyntaxType]): a list of allowed syntax types.
            executor (QSImplementation): executor.
            primitive_syntax_compatible ((TypeBase, CCGBasicSyntaxType) -> bool): an optional function.
        """

        self._basic_searcher = NeuralCCGLexiconEnumerativeSearcher(candidate_expressions, executor=executor)
        self._allowed_syntax_types = allowed_syntax_types
        self._primitive_syntax_compatible = primitive_syntax_compatible

        if self._primitive_syntax_compatible is None:
            self._primitive_syntax_compatible = lambda x, y: True

    def is_primitive_compatible(self, dsl_type: TypeBase, syntax_type: CCGSyntaxType):
        if isinstance(dsl_type, FunctionType):
            syntax_type = syntax_type.flatten()
            if len(syntax_type) - 1 != dsl_type.nr_arguments:
                return False
            if not self.is_primitive_compatible(dsl_type.return_type, syntax_type[0]):
                return False

            align = np.zeros((len(dsl_type.argument_types), len(syntax_type[1:])), dtype=np.int32)
            for i, dsl_type_arg in enumerate(dsl_type.argument_types):
                for j, syntax_arg in enumerate(syntax_type[1:]):
                    align[i, j] = int(self.is_primitive_compatible(dsl_type_arg, syntax_arg[0]))
            if align[linear_sum_assignment(align, maximize=True)].sum() == len(dsl_type.argument_types):  # max matching exists
                return True
        else:
            if not isinstance(syntax_type, CCGPrimitiveSyntaxType):
                return False
            return self._primitive_syntax_compatible(dsl_type, syntax_type)

    def is_compatible(self, semantics_type: NeuralCCGSyntaxType, syntax_type: CCGSyntaxType):
        if isinstance(syntax_type, CCGPrimitiveSyntaxType):
            if semantics_type.is_function:
                return False
            return self.is_primitive_compatible(semantics_type.return_type, syntax_type)
        else:
            assert isinstance(syntax_type, CCGComposedSyntaxType)
            if not semantics_type.is_function:
                return False
            if syntax_type.direction != semantics_type.linearization[-1].direction:
                return False
            if not self.is_primitive_compatible(semantics_type.argument_types[-1], syntax_type.sub):
                return False
            new_semantics_type = NeuralCCGSyntaxType(
                semantics_type.return_type,
                argument_types=semantics_type.argument_types[:-1], linearization=semantics_type.linearization[:-1]
            )
            return self.is_compatible(new_semantics_type, syntax_type.main)

    def gen(
        self,
        init_executor: bool = True,
        allow_none_lexicon: bool = False,
        semantics_hash: Callable[[Union[Value, Function, FunctionApplicationExpression]], Any] = None
    ) -> List[NeuralCCGLexiconSearchResult]:
        without_syntax = self._basic_searcher.gen(init_executor=init_executor, allow_none_lexicon=allow_none_lexicon, semantics_hash=semantics_hash)
        with_syntax = list()
        for x in without_syntax:
            for y in self._allowed_syntax_types:
                if self.is_compatible(x.syntax, y):
                    with_syntax.append(NeuralCCGLexiconSearchResult(x.syntax.derive_lang_syntax_type(y), x.semantics, x.executor))
        return with_syntax


def gen_lexicon_search_results_from_functions(
    functions: Iterable[Tuple[Union[ConstantExpression, Function, FunctionApplicationExpression], str]],
    executor: FunctionDomainExecutor,
    init_executor: bool = True,
    allow_none_lexicon: bool = False,
) -> List[NeuralCCGLexiconSearchResult]:
    """Generate lexicon search results from a given list of functions and the corresponding linearization strings.

    Args:
        functions: a list of functions and the corresponding linearization strings.
        executor: the executor for the domain.
        init_executor: whether to initialize the executor.
        allow_none_lexicon: whether to allow None lexicon.

    Returns:
        a list of lexicon search results.
    """
    def gen() -> Iterable[Tuple[NeuralCCGSyntaxType, Union[None, ConstantExpression, Function, FunctionApplicationExpression]]]:
        for f, linearization_string in functions:
            if isinstance(f, ConstantExpression):
                yield f.return_type, f
            if isinstance(f, FunctionApplicationExpression):
                yield NeuralCCGSyntaxType(f.return_type), f
            elif isinstance(f, Function):
                f_stat = stat_function(f)
                linearization = parse_linearization_string(linearization_string)
                syntax_type = NeuralCCGSyntaxType(
                    f.ftype.return_type,
                    f.ftype.argument_types[:f_stat.nr_variable_arguments],
                    linearization
                )
                sub_f = f.remap_arguments(
                    [x.index for x in syntax_type.linearization[::-1]] +
                    list(range(f_stat.nr_variable_arguments, f_stat.nr_variable_arguments + f_stat.nr_constant_arguments))
                )
                yield syntax_type, sub_f
        if allow_none_lexicon:
            yield NeuralCCGSyntaxType(None), None

    return gen_lexicon_search_results_from_syntax_and_semantics(
        gen(), executor, init_executor=init_executor
    )


def gen_lexicon_search_results_from_syntax_and_semantics(
    results: Iterable[Tuple[NeuralCCGSyntaxType, Union[None, ConstantExpression, Function, FunctionApplicationExpression]]],
    executor: FunctionDomainExecutor,
    init_executor: bool = True,
) -> List[NeuralCCGLexiconSearchResult]:
    """Generate lexicon search results from a given list of syntax and semantic forms.

    Args:
        results: a list of syntax and semantic forms.
        executor: the executor for the domain.
        init_executor: whether to initialize the executor.

    Returns:
        A list of lexicon search results.
    """
    outputs = list()
    for syntax_type, expression in results:
        if expression is None:
            outputs.append(NeuralCCGLexiconSearchResult(syntax_type, None, None))
        elif isinstance(expression, ConstantExpression):
            exe = expression.constant if init_executor else None
            outputs.append(NeuralCCGLexiconSearchResult(syntax_type, expression, exe))
        if isinstance(expression, FunctionApplicationExpression):
            exe = NeuralCCGGroundingFunction(expression, executor, nr_arguments=0, constant_arg_types=tuple(), note=str(expression)) if init_executor else None
            outputs.append(NeuralCCGLexiconSearchResult(syntax_type, expression, exe))
        elif isinstance(expression, Function):
            exe = NeuralCCGGroundingFunction(
                expression, executor, syntax_type.arity,
                expression.ftype.argument_types[syntax_type.arity:],
                note=str(expression)
            ) if init_executor else None
            outputs.append(NeuralCCGLexiconSearchResult(syntax_type, expression, exe))
        else:
            raise TypeError(f'Unknown expression type: {type(expression)}.')

    return outputs
