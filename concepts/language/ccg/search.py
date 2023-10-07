#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : search.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/05/2020
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Algorithms for enumerate possible syntax types and semantic forms in a domain."""

import itertools
from typing import Any, Optional, Union, Iterable, Tuple, List, Callable
from dataclasses import dataclass
from jacinle.utils.defaults import default_args, ARGDEF

from concepts.dsl.dsl_functions import FunctionType, Function
from concepts.dsl.expression import ConstantExpression, FunctionApplicationExpression
from concepts.dsl.function_domain import FunctionDomain
from concepts.dsl.learning.function_domain_search import FunctionDomainExpressionEnumerativeSearcher, stat_function
from concepts.language.ccg.syntax import CCGSyntaxSystem, CCGPrimitiveSyntaxType, CCGSyntaxType
from concepts.language.ccg.semantics import CCGSemantics

__all__ = [
    'CCGSyntaxSearchResult', 'CCGSyntaxSearcherBase', 'CCGSyntaxEnumerativeSearcher', 'gen_syntax_search_result_from_syntax_types',
    'CCGSemanticsSearchResult', 'CCGSemanticsSearcherBase', 'CCGSemanticsEnumerativeSearcher', 'gen_semantics_search_result_from_functions',
]


@dataclass
class CCGSyntaxSearchResult(object):
    """Search result for :class:`CCGSyntaxSearcherBase`."""

    syntax: CCGSyntaxType
    """The candidate syntax type."""

    depth: int
    """The depth of the candidate syntax type."""


class CCGSyntaxSearcherBase(object):
    """Base class for syntax searchers."""

    def __init__(self, syntax_system: CCGSyntaxSystem):
        """Initialize the searcher.

        Args:
            syntax_system: the syntax system, containing all primitive and conjunction syntax types.
        """
        self._syntax_system = syntax_system

    @property
    def syntax_system(self):
        """The syntax system."""
        return self._syntax_system

    def gen(self) -> List[CCGSyntaxSearchResult]:
        """Generate a list of candidate syntax types.

        Returns:
            A list of candidate syntax types.
        """
        raise NotImplementedError()


class CCGSyntaxEnumerativeSearcher(CCGSyntaxSearcherBase):
    """Enumerative searcher for syntax types."""

    def __init__(self, syntax_system: CCGSyntaxSystem, starting_symbols: Iterable[str] = ('S', )):
        """Initialize the searcher.

        Args:
            syntax_system: the syntax system, containing all primitive and conjunction syntax types.
            starting_symbols: the root primitive symbol for candidate syntax types.
        """
        super().__init__(syntax_system)
        self._starting_symbols = tuple(starting_symbols)

        for s in self._starting_symbols:
            assert isinstance(self._syntax_system[s], CCGPrimitiveSyntaxType)

    def gen(self, max_depth: int = 3, allow_functor_type: bool = False) -> List[CCGSyntaxSearchResult]:
        """Generate a list of candidate syntax types.

        Args:
            max_depth: The maximum depth of the syntax tree.
            allow_functor_type: Whether to allow functor types during composition. When this is set to False, the
                function will not generate any syntax types that contains functor-typed arguments. For example S/(NP/NP).

        Returns:
            A list of candidate syntax types.
        """
        current = {i: list() for i in range(1, max_depth + 1)}
        current_typenames = set()

        def add(depth, syntax):
            if syntax.typename not in current_typenames:
                current[depth].append(syntax)
                current_typenames.add(syntax.typename)

        for symbol in self._syntax_system.types.values():
            add(1, symbol)

        for depth in range(2, max_depth + 1):
            if allow_functor_type:
                for depth1 in range(1, depth):
                    for depth2 in range(1, depth - depth1 + 1):
                        for syntax1, syntax2 in itertools.product(current[depth1], current[depth2]):
                            if syntax2.typename not in self._starting_symbols:
                                add(depth, syntax1 / syntax2)
                                add(depth, syntax1 // syntax2)
            else:
                depth1 = depth - 1
                depth2 = depth - depth1
                for syntax1, syntax2 in itertools.product(current[depth1], current[depth2]):
                    if syntax2.typename not in self._starting_symbols:
                        add(depth, syntax1 / syntax2)
                        add(depth, syntax1 // syntax2)

        results = list()
        for k, vs in current.items():
            results.extend([CCGSyntaxSearchResult(v, k) for v in vs])
        return results


def gen_syntax_search_result_from_syntax_types(syntax_system: CCGSyntaxSystem, syntax_types: Iterable[CCGSyntaxType]) -> List[CCGSyntaxSearchResult]:
    """Generate a list of syntax search results from a list of syntax types.

    Args:
        syntax_system: the syntax system, containing all primitive and conjunction syntax types.
        syntax_types: the syntax types.

    Returns:
        A list of syntax search results.
    """
    results = list()
    for syntax_type in syntax_types:
        results.append(CCGSyntaxSearchResult(syntax_type, 0))
    return results


@dataclass
class CCGSemanticsSearchResult(object):
    """Search result for :class:`CCGSemanticsSearcherBase`."""

    semantics: CCGSemantics
    """The candidate semantic form."""

    depth: int
    """The depth of the candidate semantics."""

    nr_constant_arguments: int
    """The number of constant arguments in the semantic form."""

    nr_variable_arguments: int
    """The number of variable arguments in the semantic form."""

    nr_function_arguments: int
    """The number of function arguments in the semantic form."""


class CCGSemanticsSearcherBase(object):
    """Base class for semantics searchers."""

    def gen(self) -> List[CCGSemanticsSearchResult]:
        """Generate a list of candidate semantic forms.

        Returns:
            A list of candidate semantic forms.
        """
        raise NotImplementedError()


_Types = FunctionDomain.AllowedTypes


class CCGSemanticsEnumerativeSearcher(CCGSemanticsSearcherBase):
    """Enumerative searcher for semantics."""

    def __init__(self, domain: FunctionDomain):
        """Initialize the searcher.

        Args:
            domain: the domain of the semantics.
        """
        self._domain = domain
        self._enumerator = FunctionDomainExpressionEnumerativeSearcher(domain)

    def gen(
        self,
        max_depth: int = ARGDEF,
        return_type: Optional[Union[_Types, Tuple[_Types, ...], List[_Types]]] = None,
        max_variable_arguments: int = ARGDEF,
        max_constant_arguments: int = ARGDEF,
        max_function_arguments: int = ARGDEF,
        search_constants: bool = ARGDEF,
        hash_function: Callable[[Union[Function, FunctionApplicationExpression]], Any] = None,
        verbose: bool = False
    ) -> List[CCGSemanticsSearchResult]:
        """Generate a list of candidate semantic forms.

        Args:
            max_depth: the maximum depth of the semantics tree.
            return_type: the return type of the semantics.
            max_variable_arguments: the maximum number of variables in the semantics.
            max_constant_arguments: the maximum number of constants in the semantics.
            max_function_arguments: the maximum number of functions in the semantics.
            search_constants: whether to search for constants when generating semantic forms.
            hash_function: an optional hash function that will be used to filter out duplicate functions.
            verbose: whether to print out the progress.

        Returns:
            A list of candidate semantic forms.
        """
        return (
            (self.gen_constant_semantics(return_type) if search_constants else list()) +
            self.gen_function_semantics(
                return_type,
                max_depth=max_depth,
                max_variable_arguments=max_variable_arguments,
                max_constant_arguments=max_constant_arguments,
                max_function_arguments=max_function_arguments,
                search_constants=search_constants,
                hash_function=hash_function,
                verbose=verbose
            )
        )

    def gen_constant_semantics(
        self,
        return_type: Optional[Union[_Types, Tuple[_Types, ...], List[_Types]]] = None
    ) -> List[CCGSemanticsSearchResult]:
        """Generate a list of candidate semantic forms that are simply constants.

        Args:
            return_type: the return type of the semantics.

        Returns:
            A list of candidate semantic forms.
        """
        return [CCGSemanticsSearchResult(
            CCGSemantics(c.expression),
            depth=0, nr_constant_arguments=1, nr_variable_arguments=0, nr_function_arguments=0
        ) for c in self._enumerator.gen_constant_expressions(return_type)]

    @default_args
    def gen_function_semantics(
        self,
        return_type: Optional[Union[_Types, Tuple[_Types, ...], List[_Types]]] = None, *,
        max_depth: int = 3,
        max_variable_arguments: int = 2,
        max_constant_arguments: int = 1,
        max_function_arguments: int = 0,
        search_constants: bool = False,
        hash_function: Callable[[Union[Function, FunctionApplicationExpression]], Any] = None,
        verbose: bool = False
    ) -> List[CCGSemanticsSearchResult]:
        return [CCGSemanticsSearchResult(
            CCGSemantics(f.expression),
            depth=f.depth,
            nr_constant_arguments=f.nr_constant_arguments,
            nr_variable_arguments=f.nr_variable_arguments,
            nr_function_arguments=f.nr_function_arguments
        ) for f in self._enumerator.gen_function_application_expressions(
            return_type=return_type,
            max_depth=max_depth,
            max_variable_arguments=max_variable_arguments,
            max_constant_arguments=max_constant_arguments,
            max_function_arguments=max_function_arguments,
            search_constants=search_constants,
            hash_function=hash_function,
            verbose=verbose,
        )]

    def _gen_function_primitives(self, ret_type, max_function_arguments):
        def function_call(func, *args):
            return func(*args)

        def gen():
            types = tuple(self._domain.types.values())
            for repeat in range(1, max_function_arguments + 1):
                for arg_types in itertools.product(types, repeat=repeat):
                    yield Function(
                        '__lambda__',
                        FunctionType(
                            [FunctionType(arg_types, ret_type), ] + list(arg_types),
                            ret_type
                        ),
                        overridden_call=function_call,
                    )

        return tuple(gen())

    @staticmethod
    def gen_lambda(f1: Function, arg_index=None, f2: Optional[Function] = None):
        if arg_index is None:
            return Function(
                '__lambda__',
                FunctionType(f1.ftype.argument_types, f1.ftype.return_type), overridden_call=f1
            )
        else:
            f1_arg_types = f1.ftype.argument_types
            f2_arg_types = f2.ftype.argument_types
            arg_types = f2_arg_types + f1_arg_types[:arg_index] + f1_arg_types[arg_index + 1:]

            def new_function_call(*args):
                f2_args = args[:f2.nr_arguments]
                f1_args = list(args[f2.nr_arguments:])

                f2_ret = f2(*f2_args)
                f1_args.insert(arg_index, f2_ret)
                return f1(*f1_args)

            return Function(
                '__lambda__',
                FunctionType(arg_types, f1.ftype.return_type),
                overridden_call=new_function_call
            )


def gen_semantics_search_result_from_functions(functions: Iterable[Union[Function, FunctionApplicationExpression, ConstantExpression]]) -> List[CCGSemanticsSearchResult]:
    """
    Generate a list of semantics search results from functions.

    Args:
        functions: the functions.

    Returns:
        A list of :class:`CCGSemanticsSearchResult` instances.
    """
    results = list()
    for f in functions:
        if isinstance(f, Function):
            stat = stat_function(f)
            results.append(CCGSemanticsSearchResult(CCGSemantics(f), 1, nr_constant_arguments=stat.nr_constant_arguments, nr_variable_arguments=stat.nr_variable_arguments, nr_function_arguments=stat.nr_function_arguments))
        else:
            results.append(CCGSemanticsSearchResult(CCGSemantics(f), 1, nr_constant_arguments=0, nr_variable_arguments=0, nr_function_arguments=0))
    return results

