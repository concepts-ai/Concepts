#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : cdl_parser.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/10/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os
import os.path as osp
import contextlib
import jacinle

from typing import Any, Optional, Union, Sequence, Tuple, Set, List, Dict
from dataclasses import dataclass, field
from lark import Lark, Tree
from lark.visitors import Transformer, Interpreter, v_args
from lark.indenter import PythonIndenter

import concepts.dsl.expression as E
from concepts.dsl.dsl_types import TypeBase, AutoType, ListType, BOOL, Variable, ObjectConstant, UnnamedPlaceholder, QINDEX
from concepts.dsl.dsl_functions import FunctionType
from concepts.dsl.tensor_value import TensorValue
from concepts.dsl.tensor_state import StateObjectList

from concepts.dm.crow.crow_function import CrowFunction
from concepts.dm.crow.controller import CrowControllerApplicationExpression
from concepts.dm.crow.behavior import CrowBehaviorBodyItem
from concepts.dm.crow.behavior import CrowBehaviorApplicationExpression
from concepts.dm.crow.behavior import CrowBehaviorOrderingSuite
from concepts.dm.crow.crow_generator import CrowGeneratorApplicationExpression
from concepts.dm.crow.crow_domain import CrowDomain, CrowProblem, CrowState
from concepts.dm.crow.behavior_utils import execute_effect_statements
from concepts.dm.crow.parsers.cdl_literal_parser import InTypedArgument, ArgumentsDef, CSList, CDLLiteralTransformer
from concepts.dm.crow.parsers.cdl_symbolic_execution import ArgumentsList, FunctionCall, Suite

logger = jacinle.get_logger(__name__)
inline_args = v_args(inline=True)


__all__ = [
    'CDLPathResolver', 'get_default_path_resolver',
    'CDLParser', 'get_default_parser',
    'load_domain_file', 'load_domain_string', 'load_domain_file_incremental', 'load_domain_string_incremental',
    'load_problem_file', 'load_problem_string',
    'parse_expression',
    'CDLDomainTransformer', 'CDLProblemTransformer', 'CDLExpressionInterpreter',
]


class CDLPathResolver(object):
    def __init__(self, search_paths: Sequence[str] = tuple()):
        self.search_paths = list(search_paths)

    def resolve(self, filename: str, relative_filename: Optional[str] = None) -> str:
        if osp.exists(filename):
            return filename
        # Try the relative filename first.
        if relative_filename is not None:
            relative_dir = osp.dirname(relative_filename)
            full_path = osp.join(relative_dir, filename)
            if osp.exists(full_path):
                return full_path
        # Try the current directory second.
        if osp.exists(osp.join(os.getcwd(), filename)):
            return osp.join(os.getcwd(), filename)
        # Then try the search paths.
        for path in self.search_paths:
            full_path = osp.join(path, filename)
            if osp.exists(full_path):
                return full_path
        raise FileNotFoundError(f'File not found: {filename}')

    def add_search_path(self, path: str):
        if path not in self.search_paths:
            self.search_paths.append(path)

    def remove_search_path(self, path: str):
        self.search_paths.remove(path)


class CDLParser(object):
    """The parser for PDSketch v3."""

    grammar_file = osp.join(osp.dirname(osp.abspath(__file__)), 'cdl.grammar')
    """The grammar definition v3 for PDSketch."""

    def __init__(self):
        """Initialize the parser."""
        with open(self.grammar_file, 'r') as f:
            self.grammar = f.read()
        self.parser = Lark(self.grammar, start='start', postlex=PythonIndenter(), parser='lalr')

    def parse(self, filename: str) -> Tree:
        """Parse a PDSketch v3 file.

        Args:
            filename: the filename to parse.

        Returns:
            the parse tree. It is a :class:`lark.Tree` object.
        """

        filename = _DEFAULT_PATH_RESOLVER.resolve(filename)
        with open(filename, 'r') as f:
            return self.parse_str(f.read())

    def parse_str(self, s: str) -> Tree:
        """Parse a PDSketch v3 string.

        Args:
            s: the string to parse.

        Returns:
            the parse tree. It is a :class:`lark.Tree` object.
        """
        # NB(Jiayuan Mao @ 2024/03/13): for reasons, the pdsketch-v3 grammar file requires that the string ends with a newline.
        # In particular, the suite definition requires that the file ends with a _DEDENT token, which seems to be only triggered by a newline.
        s = s.strip() + '\n'

        if s.startswith('#!') and not s.startswith('#!pragma'):
            s = s[s.find('\n') + 1:]

        return self.parser.parse(s)

    def parse_domain(self, filename: str, domain: Optional[CrowDomain] = None) -> CrowDomain:
        """Parse a PDSketch v3 domain file.

        Args:
            filename: the filename to parse.
            domain: the domain to use. If not provided, a new domain will be created.

        Returns:
            the parsed domain.
        """
        return self.transform_domain(self.parse(filename), domain=domain)

    def parse_domain_str(self, s: str, domain: Optional[CrowDomain] = None) -> Any:
        """Parse a PDSketch v3 domain string.

        Args:
            s: the string to parse.
            domain: the domain to use. If not provided, a new domain will be created.

        Returns:
            the parsed domain.
        """
        return self.transform_domain(self.parse_str(s), domain=domain)

    def parse_problem(self, filename: str, domain: Optional[CrowDomain] = None) -> CrowProblem:
        """Parse a PDSketch v3 problem file.

        Args:
            filename: the filename to parse.
            domain: the domain to use. If not provided, the domain will be parsed from the problem file.

        Returns:
            the parsed problem.
        """
        return self.transform_problem(self.parse(filename), domain=domain)

    def parse_problem_str(self, s: str, domain: Optional[CrowDomain] = None) -> CrowProblem:
        """Parse a PDSketch v3 problem string.

        Args:
            s: the string to parse.
            domain: the domain to use. If not provided, the domain will be parsed from the problem file.

        Returns:
            the parsed problem.
        """
        return self.transform_problem(self.parse_str(s), domain=domain)

    def parse_expression(self, s: str, domain: CrowDomain, state: Optional[CrowState] = None, variables: Optional[Sequence[Variable]] = None, auto_constant_guess: bool = True) -> E.Expression:
        """Parse a PDSketch v3 expression string.

        Args:
            s: the string to parse.
            domain: the domain to use.
            state: the current state, containing objects.
            variables: variables from the outer scope.
            auto_constant_guess: whether to automatically guess whether a variable is a constant.

        Returns:
            the parsed expression.
        """
        return self.transform_expression(self.parse_str(s), domain, state=state, variables=variables, auto_constant_guess=auto_constant_guess)

    @staticmethod
    def transform_domain(tree: Tree, domain: Optional[CrowDomain] = None) -> CrowDomain:
        """Transform a parse tree into a domain.

        Args:
            tree: the parse tree.
            domain: the domain to use. If not provided, a new domain will be created.

        Returns:
            the parsed domain.
        """
        transformer = CDLDomainTransformer(domain)
        transformer.transform(tree)
        return transformer.domain

    @staticmethod
    def transform_problem(tree: Tree, domain: Optional[CrowDomain] = None) -> CrowProblem:
        """Transform a parse tree into a problem.

        Args:
            tree: the parse tree.
            domain: the domain to use. If not provided, the domain will be parsed from the problem file.

        Returns:
            the parsed problem.
        """
        transformer = CDLProblemTransformer(domain)
        transformer.transform(tree)
        return transformer.problem

    @staticmethod
    def transform_expression(tree: Tree, domain: CrowDomain, state: Optional[CrowState] = None, variables: Optional[Sequence[Variable]] = None, auto_constant_guess: bool = True) -> E.Expression:
        """Transform a parse tree into an expression.

        Args:
            tree: the parse tree.
            domain: the domain to use.
            state: the current state, containing objects.
            variables: variables from the outer scope.
            auto_constant_guess: whether to automatically guess whether a variable is a constant.

        Returns:
            the parsed expression.
        """
        transformer = CDLProblemTransformer(domain, state, auto_constant_guess=auto_constant_guess)
        interpreter = transformer.expression_interpreter
        expression_def_ctx = transformer.expression_def_ctx

        # the root of the tree is the `start` rule.
        tree = transformer.transform(tree).children[0]

        if variables is None:
            variables = tuple()
        with expression_def_ctx.with_variables(*variables) as ctx:
            return interpreter.visit(tree)


_DEFAULT_PATH_RESOLVER = CDLPathResolver()
_DEFAULT_PARSER = None
_PARSER_VERBOSE = False


def get_default_path_resolver() -> CDLPathResolver:
    global _DEFAULT_PATH_RESOLVER
    return _DEFAULT_PATH_RESOLVER


def get_default_parser() -> CDLParser:
    global _DEFAULT_PARSER
    if _DEFAULT_PARSER is None:
        _DEFAULT_PARSER = CDLParser()
    return _DEFAULT_PARSER


def set_parser_verbose(verbose: bool = True):
    global _PARSER_VERBOSE
    _PARSER_VERBOSE = verbose


def load_domain_file(filename:str) -> CrowDomain:
    """Load a domain file.

    Args:
        filename: the filename of the domain file.

    Returns:
        the domain object.
    """

    return get_default_parser().parse_domain(filename)


def load_domain_string(string: str) -> CrowDomain:
    """Load a domain from a string.

    Args:
        string: the string containing the domain definition.

    Returns:
        the domain object.
    """

    return get_default_parser().parse_domain_str(string)


def load_domain_file_incremental(domain: CrowDomain, filename: str) -> CrowDomain:
    """Load a domain file incrementally.

    Args:
        domain: the domain object to be updated.
        filename: the filename of the domain file.

    Returns:
        the domain object.
    """

    return get_default_parser().parse_domain(filename, domain=domain)


def load_domain_string_incremental(domain: CrowDomain, string: str) -> CrowDomain:
    """Load a domain from a string incrementally.

    Args:
        domain: the domain object to be updated.
        string: the string containing the domain definition.

    Returns:
        the domain object.
    """

    return get_default_parser().parse_domain_str(string, domain=domain)


def load_problem_file(filename: str, domain: Optional[CrowDomain] = None) -> CrowProblem:
    """Load a problem file.

    Args:
        filename: the filename of the problem file.
        domain: the domain object. If not provided, the domain will be loaded from the domain file specified in the problem file.

    Returns:
        the problem object.
    """

    return get_default_parser().parse_problem(filename, domain=domain)


def load_problem_string(string: str, domain: Optional[CrowDomain] = None) -> CrowProblem:
    """Load a problem from a string.

    Args:
        string: the string containing the problem definition.
        domain: the domain object. If not provided, the domain will be loaded from the domain file specified in the problem file.

    Returns:
        the problem object.
    """

    return get_default_parser().parse_problem_str(string, domain=domain)


def parse_expression(domain: CrowDomain, string: str, state: Optional[CrowState] = None, variables: Optional[Sequence[Variable]] = None, auto_constant_guess: bool = True) -> E.Expression:
    """Parse an expression.

    Args:
        domain: the domain object.
        string: the string containing the expression.
        state: the current state, containing objects.
        variables: the variables.
        auto_constant_guess: whether to guess whether a variable is a constant.

    Returns:
        the parsed expression.
    """
    return get_default_parser().parse_expression(string, domain, state=state, variables=variables, auto_constant_guess=auto_constant_guess)


g_term_op_mapping = {
    '*': 'mul',
    '/': 'div',
    '+': 'add',
    '-': 'sub',
}


def _gen_term_expr_func(expr_typename: str):
    """Generate a term expression function. This function is used to generate the term expression functions for the transformer.

    It is used to generate the following functions:

    - mul_expr
    - arith_expr
    - shift_expr

    Args:
        expr_typename: the name of the expression type. This is only used for printing the debug information.

    Returns:
        the generated term expression function.
    """


    @inline_args
    def term(self, *values: Any):
        values = [self.visit(value) for value in values]
        if len(values) == 1:
            return values[0]

        assert len(values) % 2 == 1, f'[{expr_typename}] expressions expected an odd number of values, got {len(values)}. Values: {values}.'
        result = values[0]
        for i in range(1, len(values), 2):
            v1, v2 = _canonicalize_arguments_same_dtype([result, values[i + 1]])
            t = v1.return_type
            if t.is_uniform_sequence_type:
                t = t.element_type
            fname = f'type::{t.typename}::{g_term_op_mapping[values[i]]}'
            result = E.FunctionApplicationExpression(CrowFunction(fname, FunctionType([t, t], t)), [v1, v2])
        # print(f'[{expr_typename}] result: {result}')
        return result
    return term


def _gen_bitop_expr_func(expr_typename: str):
    """Generate a term expression function. This function is used to generate the term expression functions for the transformer.
    It is named `_noop` because the arguments to the function does not contain the operator being used. Therefore, we have to
    specify the operator name manually (`expr_typename`). This is used for the following functions:

    - bitand_expr
    - bitxor_expr
    - bitor_expr
    """
    op_mapping = {
        'bitand': E.BoolOpType.AND,
        'bitxor': E.BoolOpType.XOR,
        'bitor': E.BoolOpType.OR,
    }
    @inline_args
    def term(self, *values: Any):
        values = [self.visit(value) for value in values]
        if len(values) == 1:
            return values[0]
        result = E.BoolExpression(op_mapping[expr_typename], _canonicalize_arguments(values))
        # print(f'[{expr_typename}] result: {result}')
        return result
    return term


def _canonicalize_single_argument(arg: Any, dtype: Optional[TypeBase] = None) -> Union[E.ObjectOrValueOutputExpression, E.VariableExpression, type(Ellipsis)]:
    if isinstance(arg, E.ObjectOrValueOutputExpression):
        return arg
    if isinstance(arg, (CrowControllerApplicationExpression, CrowBehaviorApplicationExpression, CrowGeneratorApplicationExpression)):
        return arg
    if isinstance(arg, Variable):
        return E.VariableExpression(arg)
    if isinstance(arg, (bool, int, float, complex, str)):
        return E.ConstantExpression.from_value(arg, dtype=dtype)
    if isinstance(arg, E.ListCreationExpression):
        return arg
    if arg is QINDEX:
        return E.ObjectConstantExpression(ObjectConstant(StateObjectList(ListType(AutoType), QINDEX), ListType(AutoType)))
    if arg is Ellipsis:
        return Ellipsis
    raise ValueError(f'Invalid argument: {arg}. Type: {type(arg)}.')


def _canonicalize_arguments_same_dtype(args: Optional[Union[ArgumentsList, tuple, list]] = None, dtype: Optional[TypeBase] = None) -> Tuple[Union[E.ValueOutputExpression, E.VariableExpression], ...]:
    if args is None:
        return tuple()
    args = args.arguments if isinstance(args, ArgumentsList) else args

    if dtype is None:
        # Guess the dtype from the list.
        for arg in args:
            if isinstance(arg, E.ObjectOrValueOutputExpression):
                dtype = arg.return_type
                break

    canonicalized_args = list()
    for arg in args:
        canonicalized_args.append(_canonicalize_single_argument(arg, dtype=dtype))
    return tuple(canonicalized_args)


def _canonicalize_arguments(args: Optional[Union[ArgumentsList, tuple, list]] = None, dtypes: Optional[Union[Sequence[TypeBase], TypeBase]] = None) -> Tuple[Union[E.ValueOutputExpression, E.VariableExpression], ...]:
    if args is None:
        return tuple()
    if isinstance(dtypes, TypeBase):
        dtypes = [dtypes] * len(args)

    # TODO(Jiayuan Mao @ 2024/03/2): Strictly check the allowability of "Ellipsis" in the arguments.
    canonicalized_args = list()

    arguments = args.arguments if isinstance(args, ArgumentsList) else args
    if dtypes is not None:
        if len(arguments) != len(dtypes):
            raise ValueError(f'Number of arguments does not match the number of types: {len(arguments)} vs {len(dtypes)}. Args: {arguments}, Types: {dtypes}')

    for i, arg in enumerate(arguments):
        if arg is Ellipsis:
            canonicalized_args.append(Ellipsis)
        else:
            canonicalized_args.append(_canonicalize_single_argument(arg, dtype=dtypes[i] if dtypes is not None else None))
    return tuple(canonicalized_args)


def _safe_is_value_type(arg: Any) -> bool:
    if isinstance(arg, E.ObjectOrValueOutputExpression):
        return arg.return_type.is_value_type
    if isinstance(arg, (bool, int, float, complex, str)):
        return True
    raise ValueError(f'Invalid argument: {arg}. Type: {type(arg)}.')


def _has_list_arguments(args: Tuple[E.ObjectOrValueOutputExpression, ...]) -> bool:
    for arg in args:
        if arg.return_type.is_list_type:
            return True
    return False


class CDLExpressionInterpreter(Interpreter):
    """The transformer for expressions. Including:

    - typename
    - sized_vector_typename
    - unsized_vector_typename
    - typed_argument
    - is_typed_argument
    - in_typed_argument
    - arguments_def
    - atom_expr_funccall
    - atom_varname
    - atom
    - power
    - factor
    - unary_op_expr
    - mul_expr
    - arith_expr
    - shift_expr
    - bitand_expr
    - bitxor_expr
    - bitor_expr
    - comparison_expr
    - not_test
    - and_test
    - or_test
    - cond_test
    - test
    - test_nocond

    - tuple
    - list
    - cs_list
    - suite

    - expr_stmt
    - expr_list_expansion_stmt
    - assign_stmt
    - annotated_assign_stmt
    - local_assign_stmt
    - pass_stmt
    - return_stmt
    - achieve_stmt

    - if_stmt
    - foreach_stmt
    - foreach_in_stmt
    - while_stmt
    - forall_test
    - exists_test
    - findall_test
    - forall_in_test
    - exists_in_test
    """

    def __init__(self, domain: Optional[CrowDomain], state: Optional[CrowState], expression_def_ctx: E.ExpressionDefinitionContext, auto_constant_guess: bool = False):
        super().__init__()
        self.domain = domain
        self.state = state
        self.expression_def_ctx = expression_def_ctx
        self.auto_constant_guess = auto_constant_guess

        self.generator_impl_outputs = None
        self.local_variables = dict()

    def set_domain(self, domain: CrowDomain):
        self.domain = domain

    @contextlib.contextmanager
    def local_variable_guard(self):
        backup = self.local_variables.copy()
        yield
        self.local_variables = backup

    @contextlib.contextmanager
    def set_generator_impl_outputs(self, outputs: List[Variable]):
        backup = self.generator_impl_outputs
        self.generator_impl_outputs = outputs
        yield
        self.generator_impl_outputs = backup

    domain: CrowDomain
    expression_def_ctx: E.ExpressionDefinitionContext

    def visit(self, tree: Any) -> Any:
        if isinstance(tree, Tree):
            return super().visit(tree)
        return tree

    @inline_args
    def atom_colon(self):
        return QINDEX

    @inline_args
    def atom_varname(self, name: str) -> Union[E.VariableExpression, E.ObjectConstantExpression, E.ValueOutputExpression]:
        """Captures variable names such as `var_name`."""
        if name in self.local_variables:
            return self.local_variables[name]
        if self.state is not None and name in self.state.object_name2defaultindex:
            return E.ObjectConstantExpression(ObjectConstant(name, self.domain.types[self.state.get_default_typename(name)]))
        if name in self.domain.constants:
            constant = self.domain.constants[name]
            if isinstance(constant, ObjectConstant):
                return E.ObjectConstantExpression(constant)
            return E.ConstantExpression(constant)
        if name in self.domain.features and self.domain.features[name].nr_arguments == 0:
            return E.FunctionApplicationExpression(self.domain.features[name], tuple())
        # TODO(Jiayuan Mao @ 2024/03/12): smartly guess the type of the variable.
        if not self.expression_def_ctx.has_variable(name):
            if self.auto_constant_guess:
                return E.ObjectConstantExpression(ObjectConstant(name, AutoType))
        variable = self.expression_def_ctx.wrap_variable(name)
        return variable

    @inline_args
    def atom_expr_do_funccall(self, name: str, annotations: dict, args: Tree) -> CrowBehaviorBodyItem:
        if self.domain.has_controller(name):
            controller = self.domain.get_controller(name)
            args: Optional[ArgumentsList] = self.visit(args)
            args_c = _canonicalize_arguments(args, controller.argument_types)
            return CrowControllerApplicationExpression(controller, args_c, **annotations if annotations is not None else dict())
        else:
            raise KeyError(f'Controller {name} not found.')

    @inline_args
    def atom_expr_funccall(self, name: str, annotations: dict, args: Tree) -> Union[E.FunctionApplicationExpression, CrowBehaviorBodyItem, CrowBehaviorApplicationExpression, CrowGeneratorApplicationExpression]:
        """Captures function calls, such as `func_name(arg1, arg2, ...)`."""
        annotations: Optional[dict] = self.visit(annotations)
        args: Optional[ArgumentsList] = self.visit(args)

        if annotations is None:
            annotations = dict()
        if args is None:
            args = ArgumentsList(tuple())

        if self.domain.has_type(name):
            assert len(args.arguments) == 1, 'Type constructor expects exactly one argument.'
            if isinstance(args.arguments[0], E.ConstantExpression):
                if isinstance(args.arguments[0].constant, TensorValue) and args.arguments[0].constant.tensor.numel() == 0:
                    dtype = self.domain.get_type(name)
                    if dtype.is_uniform_sequence_type and dtype.element_type.is_object_type:
                        rv = E.ObjectConstantExpression(ObjectConstant(StateObjectList(dtype, []), dtype))
                    else:
                        rv = E.ConstantExpression(args.arguments[0].constant.clone(dtype=self.domain.get_type(name)))
                else:
                    rv = E.ConstantExpression(args.arguments[0].constant.clone(dtype=self.domain.get_type(name)))
            elif isinstance(args.arguments[0], E.ObjectConstantExpression):
                rv = E.ObjectConstantExpression(args.arguments[0].constant.clone(dtype=self.domain.get_type(name)))
            else:
                raise TypeError('Invalid type definition: {name}({args.arguments}).')
            return rv
        elif self.domain.has_feature(name):
            function = self.domain.get_feature(name)
            args_c = _canonicalize_arguments(args, function.ftype.argument_types)
            return E.FunctionApplicationExpression(function, args_c, **annotations)
        elif self.domain.has_function(name):
            function = self.domain.get_function(name)
            args_c = _canonicalize_arguments(args, function.ftype.argument_types)
            return E.FunctionApplicationExpression(function, args_c, **annotations)
        elif self.domain.has_controller(name):
            controller = self.domain.get_controller(name)
            args_c = _canonicalize_arguments(args, controller.argument_types)
            return CrowControllerApplicationExpression(controller, args_c)
        elif self.domain.has_behavior(name):
            behavior = self.domain.get_behavior(name)
            args_c = _canonicalize_arguments(args, behavior.argument_types)
            if len(args_c) > 0 and args_c[-1] is Ellipsis:
                args_c = args_c[:-1] + tuple([UnnamedPlaceholder(t) for t in behavior.argument_types[len(args_c) - 1:]])
            return CrowBehaviorApplicationExpression(behavior, args_c)
        elif self.domain.has_generator(name):
            generator = self.domain.get_generator(name)
            args_c = _canonicalize_arguments(args, generator.argument_types)
            return CrowGeneratorApplicationExpression(generator, args_c, list())
        else:
            if 'inplace_behavior_body' in annotations and annotations['inplace_behavior_body']:
                """Inplace definition of an function, used for define a __totally_ordered_plan__ function."""
                args_c = _canonicalize_arguments(args)
                argument_types = [arg.return_type for arg in args_c]
                # logger.warning(f'Behavior {name} not found, creating a new one with argument types {argument_types}.')
                predicate = self.domain.define_crow_function(name, argument_types, self.domain.get_type('__behavior_body__'))
                return E.FunctionApplicationExpression(predicate, args_c)
            elif 'inplace_generator' in annotations and annotations['inplace_generator']:
                """Inplace definition of an generator function. Typically this function is used together with the generator_placeholder annotation."""
                args_c = _canonicalize_arguments(args)
                argument_types = [arg.return_type for arg in args_c]
                # logger.warning(f'Generator placeholder function {name} not found, creating a new one with argument types {argument_types}.')

                predicate = self.domain.define_crow_function(
                    name, argument_types, BOOL,
                    generator_placeholder=annotations.get('generator_placeholder', True)
                )

                assert 'inplace_generator_targets' in annotations, f'Inplace generator {name} requires inplace generator targets to be set.'
                inplace_generator_targets = annotations['inplace_generator_targets']
                generator_name = 'gen_' + name
                generator_arguments = predicate.arguments
                generator_goal = E.FunctionApplicationExpression(predicate, [E.VariableExpression(arg) for arg in generator_arguments])

                output_argument_names = [x.value for x in inplace_generator_targets.items]
                output_indices = list()
                for i, arg in enumerate(args_c):
                    if isinstance(arg, E.VariableExpression) and arg.name in output_argument_names:
                        output_argument_names.remove(arg.name)
                        output_indices.append(i)
                assert len(output_argument_names) == 0, f'Mismatched output arguments for inplace generator {name}: {output_argument_names}'
                inputs = [arg for i, arg in enumerate(generator_arguments) if i not in output_indices]
                outputs = [arg for i, arg in enumerate(generator_arguments) if i in output_indices]
                self.domain.define_generator(generator_name, generator_arguments, generator_goal, inputs, outputs)
                return E.FunctionApplicationExpression(predicate, args_c)
            else:
                raise KeyError(f'Function {name} not found. Note that recursive function calls are not supported in the current version.')

    @inline_args
    def atom_subscript(self, name: str, annotations: dict, index: Tree) -> Union[E.FunctionApplicationExpression]:
        """Captures subscript expressions such as `name[index1, index2, ...]`."""
        feature = self.domain.get_feature(name, allow_function=True)
        index: CSList = self.visit(index)
        annotations: Optional[dict] = self.visit(annotations)
        if not feature.is_cacheable:
            raise ValueError(f'Invalid subscript expression: {name} is not a cacheable feature. Expression: {name}[{index.items}]')
        if annotations is None:
            annotations = dict()
        items = index.items
        if len(items) == 1 and items[0] is Ellipsis:
            return E.FunctionApplicationExpression(feature, tuple())
        arguments = _canonicalize_arguments(index.items, dtypes=feature.ftype.argument_types)
        return E.FunctionApplicationExpression(feature, arguments, **annotations)

    @inline_args
    def atom(self, value: Union[FunctionCall, Variable]) -> Union[FunctionCall, Variable]:
        """Captures the atom. This is used in the base case of the expression, including literal constants, variables, and subscript expressions."""
        return value

    def arguments(self, args: Tree) -> ArgumentsList:
        """Captures the argument list. This is used in function calls."""
        args = self.visit_children(args)
        return ArgumentsList(tuple(args))

    @inline_args
    def power(self, base: Union[FunctionCall, Variable], exp: Optional[Union[FunctionCall, Variable]] = None) -> Union[FunctionCall, Variable]:
        """The highest-priority expression. This is used to capture the power expression, such as `base ** exp`. If `exp` is None, it is treated as `base ** 1`."""
        if exp is None:
            return base
        raise NotImplementedError('Power expression is not supported in the current version.')

    @inline_args
    def factor(self, value: Union[FunctionCall, Variable]) -> Union[FunctionCall, Variable]:
        return value

    @inline_args
    def unary_op_expr(self, op: str, value: Union[FunctionCall, Variable]):
        value = self.visit(value)
        if op == '+':
            return value
        if op == '-':
            if isinstance(value, (int, float)):
                return -value

            t = value.return_type
            if t.is_uniform_sequence_type:
                t = t.element_type
            fname = f'type::{t.typename}::neg'
            return E.FunctionApplicationExpression(CrowFunction(fname, FunctionType([t], t)), [value])
        raise NotImplementedError(f'Unary operator {op} is not supported in the current version.')

    mul_expr = _gen_term_expr_func('mul')
    arith_expr = _gen_term_expr_func('add')
    shift_expr = _gen_term_expr_func('shift')
    bitand_expr = _gen_bitop_expr_func('bitand')
    bitxor_expr = _gen_bitop_expr_func('bitxor')
    bitor_expr = _gen_bitop_expr_func('bitor')

    @inline_args
    def comparison_expr(self, *values: Union[E.ValueOutputExpression, E.VariableExpression]) -> E.ValueOutputExpression:
        if len(values) == 1:
            return self.visit(values[0])
        assert len(values) % 2 == 1, f'[compare] expressions expected an odd number of values, got {len(values)}. Values: {values}.'
        values = [self.visit(value) for value in values]
        results = list()
        for i in range(1, len(values), 2):
            if not _safe_is_value_type(values[i - 1]) and not _safe_is_value_type(values[i + 1]):
                results.append(E.ObjectCompareExpression(E.CompareOpType.from_string(values[i][0].value), values[i - 1], values[i + 1]))
            elif _safe_is_value_type(values[i - 1]) and _safe_is_value_type(values[i + 1]):
                v1, v2 = _canonicalize_arguments_same_dtype([values[i - 1], values[i + 1]])
                results.append(E.ValueCompareExpression(E.CompareOpType.from_string(values[i][0].value), v1, v2))
            else:
                raise ValueError(f'Invalid comparison: {values[i - 1]} vs {values[i + 1]}')
        if len(results) == 1:
            return results[0]
        result = E.AndExpression(*results)
        return result

    @inline_args
    def not_test(self, value: Any) -> E.NotExpression:
        return E.NotExpression(*_canonicalize_arguments([self.visit(value)], BOOL))

    @inline_args
    def and_test(self, *values: Any) -> E.AndExpression:
        values = [self.visit(value) for value in values]
        if len(values) == 1:
            return values[0]
        result = E.AndExpression(*_canonicalize_arguments(values, BOOL))
        return result

    @inline_args
    def or_test(self, *values: Any) -> E.OrExpression:
        values = [self.visit(value) for value in values]
        if len(values) == 1:
            return values[0]
        result = E.OrExpression(*_canonicalize_arguments(values, BOOL))
        return result

    @inline_args
    def cond_test(self, value1: Any, cond: Any, value2: Any) -> E.ConditionExpression:
        x, y = _canonicalize_arguments_same_dtype([self.visit(value1), self.visit(value2)])
        return E.ConditionExpression(_canonicalize_single_argument(self.visit(cond), BOOL), x, y)

    @inline_args
    def test(self, value: Any):
        return self.visit(value)

    @inline_args
    def test_nocond(self, value: Any):
        return self.visit(value)

    @inline_args
    def tuple(self, *values: Any):
        return tuple(self.visit(v) for v in values)

    @inline_args
    def list(self, *values: Any):
        values = [self.visit(v) for v in values]

        # TODO(Jiayuan Mao @ 2024/08/10): make this more general, like nested list etc.
        if all(isinstance(v, (int, float)) for v in values):
            return E.ConstantExpression(TensorValue.from_values(*values))
        else:
            return E.ListCreationExpression(_canonicalize_arguments(values))

    @inline_args
    def cs_list(self, *values: Any):
        return CSList(tuple(self.visit(v) for v in values))

    @inline_args
    def suite(self, *values: Tree, activate_variable_guard: bool = True) -> Suite:
        if activate_variable_guard:
            with self.local_variable_guard():
                values = [self.visit(value) for value in values]
                local_variables = self.local_variables.copy()
        else:
            values = [self.visit(value) for value in values]
            local_variables = self.local_variables.copy()
        return Suite(tuple(v for v in values if v is not None), local_variables)

    @inline_args
    def expr_stmt(self, value: Tree):
        value = self.visit(value)
        if value is Ellipsis:
            return None
        # NB(Jiayuan Mao @ 2024/06/21): for handling string literals as docs.
        if isinstance(value, str):
            return None
        return FunctionCall('expr', ArgumentsList((_canonicalize_single_argument(value),)))

    @inline_args
    def expr_list_expansion_stmt(self, value: Any):
        value = _canonicalize_single_argument(self.visit(value))
        return FunctionCall('expr', ArgumentsList((E.ListExpansionExpression(value), )))

    @inline_args
    def compound_expr_stmt(self, value: Tree):
        value = self.visit(value)
        if value is Ellipsis:
            return None
        if isinstance(value, str):
            return None
        return FunctionCall('expr', ArgumentsList((_canonicalize_single_argument(value),)))

    def _make_additive_assign_stmt(self, lv, rv, op: str, annotations: dict):
        if op == '=':
            return FunctionCall('assign', ArgumentsList((lv, rv)), annotations)
        if op in ('+=', '-=', '*=', '/='):
            t = lv.return_type
            if t.is_uniform_sequence_type:
                t = t.element_type
            fname = f'type::{t.typename}::{g_term_op_mapping[op[0]]}'
            result = E.FunctionApplicationExpression(CrowFunction(fname, FunctionType([t, t], t)), [lv, rv])
            return FunctionCall('assign', ArgumentsList((lv, result)), annotations)
        if op == '%=':
            t = lv.return_type
            if t.is_uniform_sequence_type:
                t = t.element_type
            fname = f'type::{t.typename}::mod'
            result = E.FunctionApplicationExpression(CrowFunction(fname, FunctionType([t, t], t)), [lv, rv])
            return FunctionCall('assign', ArgumentsList((lv, result)), annotations)
        if op in ('&=', '|=', '^='):
            mapping = {
                '&': E.BoolOpType.AND,
                '|': E.BoolOpType.OR,
                '^': E.BoolOpType.XOR,
            }
            result = E.BoolExpression(mapping[op[0]], (lv, rv))
            return FunctionCall('assign', ArgumentsList((lv, result)), annotations)
        raise ValueError(f'Invalid assignment operator: {op}')

    def assign_stmt_inner(self, op: str, target: Any, value: Any, annotations: dict):
        if target.data == 'atom_varname':
            target_lv = target.children[0]
        else:
            target_lv = _canonicalize_single_argument(self.visit(target))  # left value

        if isinstance(target_lv, str):
            if target_lv in self.local_variables:
                annotations.setdefault('local', True)
                target_lv = self.local_variables[target_lv]
                target_rv = _canonicalize_single_argument(self.visit(value))
                # return FunctionCall('assign', ArgumentsList((target_lv, target_rv)), annotations)
                return self._make_additive_assign_stmt(target_lv, target_rv, op, annotations)
            else:
                if target_lv in self.domain.features and self.domain.get_feature(target_lv).nr_arguments == 0:
                    target_lv = E.FunctionApplicationExpression(self.domain.get_feature(target_lv), tuple())
                    # return FunctionCall('assign', ArgumentsList((target_lv, _canonicalize_single_argument(self.visit(value)))), annotations)
                    return self._make_additive_assign_stmt(target_lv, _canonicalize_single_argument(self.visit(value)), op, annotations)
                else:
                    raise NameError(f'Invalid assignment target: it is not a local variable and not a feature with 0 arguments: {target_lv}')
        # return FunctionCall('assign', ArgumentsList((target_lv, _canonicalize_single_argument(self.visit(value)))), annotations)
        return self._make_additive_assign_stmt(target_lv, _canonicalize_single_argument(self.visit(value)), op, annotations)

    @inline_args
    def assign_stmt(self, target: Any, op: Any, value: Any):
        return self.assign_stmt_inner(op.value, target, value, dict())

    @inline_args
    def annotated_assign_stmt(self, annotations: dict, target: Any, op: Any, value: Any):
        return self.assign_stmt_inner(op.value, target, value, annotations)

    @inline_args
    def let_assign_stmt(self, target: str, dtype: Any = None, value: Any = None):
        assert isinstance(target, str), f'Invalid local variable name: {target}'
        dtype = self.visit(dtype)
        value = _canonicalize_single_argument(self.visit(value))
        if dtype is not None:
            dtype = self.domain.get_type(dtype)
            if value is not None:
                assert value.return_type.downcast_compatible(dtype), f'Invalid assignment: variable {target} has dtype {dtype} but the value has dtype {value.return_type}.'
        if dtype is None and value is not None:
            dtype = value.return_type
        if dtype is None:
            dtype = AutoType
        self.local_variables[target] = E.VariableExpression(Variable(target, dtype))
        if value is not None:
            return FunctionCall('assign', ArgumentsList((self.local_variables[target], value)), {'local': True})
        return None

    @inline_args
    def symbol_assign_stmt(self, target: str, value: Any):
        assert isinstance(target, str), f'Invalid symbol variable name: {target}'
        value = _canonicalize_single_argument(self.visit(value))
        if target in self.local_variables:
            raise RuntimeError(f'Local symbol variable {target} has been assigned before.')
        self.local_variables[target] = E.VariableExpression(Variable(target, value.return_type))
        if value is not None:
            return FunctionCall('assign', ArgumentsList((self.local_variables[target], value)), {'symbol': True})
        return None

    @inline_args
    def pass_stmt(self):
        return FunctionCall('pass', ArgumentsList(tuple()))

    @inline_args
    def commit_stmt(self, kwargs: dict):
        return FunctionCall('commit', ArgumentsList(tuple()), kwargs)

    @inline_args
    def achieve_once_stmt(self, value: CSList):
        return FunctionCall('achieve', ArgumentsList(_canonicalize_arguments(self.visit(value).items, BOOL)), {'once': True})

    @inline_args
    def achieve_hold_stmt(self, value: CSList):
        return FunctionCall('achieve', ArgumentsList(_canonicalize_arguments(self.visit(value).items, BOOL)), {'once': False})

    @inline_args
    def pachieve_once_stmt(self, value: CSList):
        return FunctionCall('pachieve', ArgumentsList(_canonicalize_arguments(self.visit(value).items, BOOL)), {'once': True})

    @inline_args
    def pachieve_hold_stmt(self, value: CSList):
        return FunctionCall('pachieve', ArgumentsList(_canonicalize_arguments(self.visit(value).items, BOOL)), {'once': False})

    @inline_args
    def untrack_stmt(self, value: CSList):
        if value is None:
            return FunctionCall('untrack', ArgumentsList(tuple()))
        return FunctionCall('untrack', ArgumentsList(_canonicalize_arguments(self.visit(value).items, BOOL)))

    @inline_args
    def assert_once_stmt(self, value: Any):
        return FunctionCall('assert', ArgumentsList((_canonicalize_single_argument(self.visit(value), BOOL), )), {'once': True})

    @inline_args
    def assert_hold_stmt(self, value: Any):
        return FunctionCall('assert', ArgumentsList((_canonicalize_single_argument(self.visit(value), BOOL), )), {'once': False})

    @inline_args
    def return_stmt(self, value: Any):
        return FunctionCall('return', ArgumentsList((_canonicalize_single_argument(self.visit(value)), )))

    @inline_args
    def ordered_suite(self, ordering_op: Any, body: Any):
        ordering_op = self.visit(ordering_op)
        assert body.data.value == 'suite', f'Invalid body type: {body}'
        # We need a special handling for unordered and promotable sections because their execution order is unknown and therefore we can not rely the order of the statements.
        if ordering_op in ('promotable', 'unordered', 'promotable unordered', 'promotable sequential', 'alternative'):
            with self.local_variable_guard():
                body = self.visit(body)
            return FunctionCall('ordering', ArgumentsList((ordering_op, body)))
        else:
            body = self.visit_children(body)
            body = Suite(tuple(body), self.local_variables.copy())
            return FunctionCall('ordering', ArgumentsList((ordering_op, body)))

    @inline_args
    def ordering_op(self, *ordering_op: Any):
        return ' '.join([x.value for x in ordering_op])

    @inline_args
    def if_stmt(self, cond: Any, suite: Any, else_suite: Optional[Any] = None):
        cond = _canonicalize_single_argument(self.visit(cond))
        with self.local_variable_guard():
            suite = self.visit(suite)
        if else_suite is None:
            else_suite = Suite((FunctionCall('pass', ArgumentsList(tuple())), ))
        else:
            with self.local_variable_guard():
                else_suite = self.visit(else_suite)
        return FunctionCall('if', ArgumentsList((cond, suite, else_suite)))

    @inline_args
    def foreach_stmt(self, cs_list: Any, suite: Any):
        cs_list = self.visit(cs_list)
        with self.expression_def_ctx.new_variables(*cs_list.items), self.local_variable_guard():
            suite = self.visit(suite)
        return FunctionCall('foreach', ArgumentsList((cs_list, suite)))

    @inline_args
    def foreach_in_stmt(self, variables_cs_list: Any, values_cs_list: Any, suite: Any) -> object:
        values_cs_list = self.visit(values_cs_list)
        variables_cs_list = self.visit(variables_cs_list)

        if len(variables_cs_list.items) != len(values_cs_list.items):
            raise ValueError(f'Number of variables does not match the number of values: {len(variables_cs_list.items)} vs {len(values_cs_list.items)}. Variables: {variables_cs_list.items}, Values: {values_cs_list.items}')

        variable_items = list()
        for i in range(len(variables_cs_list.items)):
            # Variables are just names, not typed Variables. So we need to wrap them.
            return_type = values_cs_list.items[i].return_type
            if return_type.is_list_type:
                variable_items.append(Variable(variables_cs_list.items[i], return_type.element_type))
            elif return_type.is_batched_list_type:
                variable_items.append(Variable(variables_cs_list.items[i], return_type.iter_element_type()))
            else:
                raise ValueError(f'Invalid foreach_in statement: {values_cs_list.items[i]} is not a list.')
        variables_cs_list = CSList(tuple(variable_items))
        with self.expression_def_ctx.new_variables(*variables_cs_list.items), self.local_variable_guard():
            suite = self.visit(suite)
        return FunctionCall('foreach_in', ArgumentsList((variables_cs_list, values_cs_list, suite, )))

    @inline_args
    def while_stmt(self, cond: Any, suite: Any):
        cond = _canonicalize_single_argument(self.visit(cond))
        with self.local_variable_guard():
            suite = self.visit(suite)
        return FunctionCall('while', ArgumentsList((cond, suite)))

    def _quantification_expression(self, cs_list: Any, suite: Any, quantification_cls):
        cs_list = self.visit(cs_list)
        with self.expression_def_ctx.new_variables(*cs_list.items):
            body = self.visit(suite)
        for item in reversed(cs_list.items):
            body = quantification_cls(item, body)
        return body

    @inline_args
    def forall_test(self, cs_list: Any, suite: Any):
        return self._quantification_expression(cs_list, suite, E.ForallExpression)

    @inline_args
    def exists_test(self, cs_list: Any, suite: Any):
        return self._quantification_expression(cs_list, suite, E.ExistsExpression)

    @inline_args
    def batched_test(self, cs_list: Any, suite: Any):
        return self._quantification_expression(cs_list, suite, E.BatchedExpression)

    @inline_args
    def findall_test(self, variable: Variable, suite: Any):
        with self.expression_def_ctx.new_variables(variable), self.local_variable_guard():
            body = self.visit(suite)
        return E.FindAllExpression(variable, body)

    @inline_args
    def findone_test(self, variable: Variable, suite: Any):
        with self.expression_def_ctx.new_variables(variable), self.local_variable_guard():
            body = self.visit(suite)
        return E.FindOneExpression(variable, body)

    def _quantification_in_expression(self, cs_list: Any, suite: Any, quantification_cls):
        cs_list = self.visit(cs_list)
        with self.local_variable_guard():
            item: InTypedArgument
            for item in cs_list.items:
                self.local_variables[item.name] = self.visit(item.value)
            body = self.visit(suite)
        return quantification_cls(body)

    @inline_args
    def forall_in_test(self, cs_list: Any, suite: Any):
        return self._quantification_in_expression(cs_list, suite, E.AndExpression)

    @inline_args
    def exists_in_test(self, cs_list: Any, suite: Any):
        return self._quantification_in_expression(cs_list, suite, E.OrExpression)

    @inline_args
    def bind_stmt(self, cs_list: Any, suite: Any):
        cs_list = self.visit(cs_list)
        with self.expression_def_ctx.new_variables(*cs_list.items), self.local_variable_guard():
            suite = self.visit(suite)
        body = suite.get_derived_expression()
        if isinstance(body, list):
            body = E.AndExpression(*body)
        for item in cs_list.items:
            self.local_variables[item.name] = E.VariableExpression(item)
        return FunctionCall('bind', ArgumentsList((cs_list, body)))

    @inline_args
    def bind_stmt_no_where(self, cs_list: Any):
        """Captures bind statements without a body. For example:

        .. code-block:: python

           bind x: int, y: int
        """
        cs_list = self.visit(cs_list)
        for item in cs_list.items:
            self.local_variables[item.name] = E.VariableExpression(item)
        return FunctionCall('bind', ArgumentsList((cs_list, E.NullExpression(BOOL))))

    @inline_args
    def mem_query_stmt(self, expression: Any):
        return FunctionCall('mem_query', ArgumentsList((_canonicalize_single_argument(self.visit(expression)), )))

    @inline_args
    def annotated_compound_stmt(self, annotations: dict, stmt: Any):
        stmt = self.visit(stmt)
        if stmt.annotations is None:
            stmt.annotations = annotations
        else:
            stmt.annotations.update(annotations)
        return stmt


@dataclass
class GoalPart(object):
    suite: Tree


@dataclass
class MinimizePart(object):
    suite: Tree


@dataclass
class BodyPart(object):
    suite: Tree


@dataclass
class EffectPart(object):
    suite: Tree


@dataclass
class HeuristicPart(object):
    suite: Tree


@dataclass
class InPart(object):
    suite: Tree


@dataclass
class OutPart(object):
    suite: Tree


class CDLDomainTransformer(CDLLiteralTransformer):
    def __init__(self, domain: Optional[CrowDomain] = None, auto_init_domain: bool = True):
        super().__init__()

        if auto_init_domain or domain is not None:
            self._domain = CrowDomain() if domain is None else domain
            self._expression_def_ctx = E.ExpressionDefinitionContext(domain=self.domain)
            self._expression_interpreter = CDLExpressionInterpreter(domain=self.domain, state=None, expression_def_ctx=self.expression_def_ctx, auto_constant_guess=False)
        else:
            self._domain = None
            self._expression_def_ctx = None
            self._expression_interpreter = None

    @property
    def domain(self) -> CrowDomain:
        if self._domain is None:
            raise ValueError('Domain is not initialized.')
        return self._domain

    @property
    def expression_def_ctx(self) -> E.ExpressionDefinitionContext:
        if self._expression_def_ctx is None:
            raise ValueError('Expression definition context is not initialized.')
        return self._expression_def_ctx

    @property
    def expression_interpreter(self) -> CDLExpressionInterpreter:
        if self._expression_interpreter is None:
            raise ValueError('Expression interpreter is not initialized.')
        return self._expression_interpreter

    @inline_args
    def include_definition(self, path: str):
        path = _DEFAULT_PATH_RESOLVER.resolve(path)
        tree = get_default_parser().parse(path)
        self.transform(tree)

    @inline_args
    def pragma_definition(self, pragma: Dict[str, Any]):
        if _PARSER_VERBOSE:
            print('pragma_definition', pragma)

        self._handle_pragma(pragma)

    @inline_args
    def pragma_definition_with_args(self, function, arguments):
        if _PARSER_VERBOSE:
            print('pragma_definition_with_args', function, arguments)

        if arguments is None:
            arguments = tuple()
        else:
            arguments = self.expression_interpreter.visit(arguments).arguments

        self._handle_pragma({function: arguments})

    def _handle_pragma(self, pragma: Dict[str, Any]):
        for k, v in pragma.items():
            if k == 'load_implementation':
                for lib in v:
                    lib = _DEFAULT_PATH_RESOLVER.resolve(lib)
                    self.domain.add_external_function_implementation_file(lib)

    @inline_args
    def type_definition(self, typename, basetype: Optional[Union[str, TypeBase]]):
        if _PARSER_VERBOSE:
            print(f'type_definition:: {typename=} {basetype=}')
        self.domain.define_type(typename, basetype)

    @inline_args
    def object_constant_definition(self, name: str, typename: str):
        if _PARSER_VERBOSE:
            print(f'object_constant_definition:: {name=} {typename=}')
        self.domain.define_object_constant(name, typename)

    @inline_args
    def feature_definition(self, annotations: Optional[dict], name: str, args: Optional[ArgumentsDef], ret: Optional[Union[str, TypeBase]], suite: Optional[Tree]):
        if annotations is None:
            annotations = dict()
        if args is None:
            args = ArgumentsDef(tuple())
        if ret is None:
            ret = BOOL
        elif isinstance(ret, str):
            ret = self.domain.get_type(ret)

        return_stmt = None
        if suite is not None:
            with self.expression_def_ctx.with_variables(*args.arguments):
                suite = self.expression_interpreter.visit(suite)
                return_stmt = suite.get_derived_expression()
        self.domain.define_feature(name, args.arguments, ret, derived_expression=return_stmt, **annotations)

        if _PARSER_VERBOSE:
            print(f'feature_definition:: {name=} {args.arguments=} {ret=} {annotations=} {suite=}')
            if return_stmt is not None:
                print(jacinle.indent_text(f'Return statement: {return_stmt}'))

    @inline_args
    def function_definition(self, annotations: Optional[dict], name: str, args: Optional[ArgumentsDef], ret: Optional[Union[str, TypeBase]], suite: Optional[Tree]):
        if annotations is None:
            annotations = dict()
        if args is None:
            args = ArgumentsDef(tuple())
        if ret is None:
            ret = BOOL
        elif isinstance(ret, str):
            ret = self.domain.get_type(ret)

        return_stmt = None
        if suite is not None:
            with self.expression_def_ctx.with_variables(*args.arguments):
                suite = self.expression_interpreter.visit(suite)
                return_stmt = suite.get_derived_expression()
        self.domain.define_crow_function(name, args.arguments, ret, derived_expression=return_stmt, **annotations)

        if _PARSER_VERBOSE:
            print(f'function_definition:: {name=} {args.arguments=} {ret=} {annotations=} {suite=}')
            if return_stmt is not None:
                print(jacinle.indent_text(f'Return statement: {return_stmt}'))

    @inline_args
    def controller_definition(self, annotations: Optional[dict], name: str, args: Optional[ArgumentsDef], effect: Optional[EffectPart]):
        if annotations is None:
            annotations = dict()
        if args is None:
            args = ArgumentsDef(tuple())

        if effect is not None:
            with self.expression_def_ctx.with_variables(*args.arguments):
                suite = self.expression_interpreter.visit(effect.suite)
            suite = suite.get_effect_statements()
            effect = CrowBehaviorOrderingSuite.make_sequential(*suite)

        self.domain.define_controller(name, args.arguments, effect, **annotations)

        if _PARSER_VERBOSE:
            print(f'controller_definition:: {name=} {args.arguments=}')

    @inline_args
    def behavior_goal_definition(self, suite: Tree) -> GoalPart:
        return GoalPart(suite)

    @inline_args
    def behavior_minimize_definition(self, suite: Tree) -> MinimizePart:
        return MinimizePart(suite)

    @inline_args
    def behavior_body_definition(self, suite: Tree) -> BodyPart:
        return BodyPart(suite)

    @inline_args
    def behavior_effect_definition(self, suite: Tree) -> EffectPart:
        return EffectPart(suite)

    @inline_args
    def behavior_heuristic_definition(self, suite: Tree) -> HeuristicPart:
        return HeuristicPart(suite)

    @inline_args
    def behavior_definition(self, annotations: Optional[dict], name: str, args: Optional[ArgumentsDef], *parts: Union[GoalPart, EffectPart, BodyPart, HeuristicPart]):
        if annotations is None:
            annotations = dict()
        if args is None:
            args = ArgumentsDef(tuple())

        if _PARSER_VERBOSE:
            print(f'behavior_definition:: {name=} {args.arguments=} {annotations=}')

        goals = list()
        minimize = None
        body = list()
        effect = None
        heuristic = None
        local_variables = None

        for part in parts:
            with self.expression_def_ctx.with_variables(*args.arguments):
                if isinstance(part, EffectPart):
                    self.expression_interpreter.local_variables = local_variables if local_variables is not None else dict()
                    suite = self.expression_interpreter.visit(part.suite)
                    self.expression_interpreter.local_variables = dict()
                else:
                    suite = self.expression_interpreter.visit(part.suite)

                if isinstance(part, GoalPart):
                    suite = suite.get_derived_expression()
                    goals.append(suite)
                    if _PARSER_VERBOSE:
                        print(jacinle.indent_text(f'Goal: {suite}'))
                elif isinstance(part, MinimizePart):
                    suite = suite.get_derived_expression()
                    minimize = suite
                    if _PARSER_VERBOSE:
                        print(jacinle.indent_text(f'Minimize: {suite}'))
                elif isinstance(part, BodyPart):
                    local_variables = suite.local_variables
                    body = suite.get_behavior_body_statements()
                    if len(body) == 0:
                        body = CrowBehaviorOrderingSuite('sequential', tuple())
                    elif len(body) == 1:
                        if not isinstance(body[0], CrowBehaviorOrderingSuite) or body[0].order.value != 'sequential':
                            body = CrowBehaviorOrderingSuite('sequential', (body[0],), _skip_simplify=True)
                    else:
                        body = CrowBehaviorOrderingSuite('sequential', tuple(body), _skip_simplify=True)

                    if _PARSER_VERBOSE:
                        print(jacinle.indent_text(f'Body:'))
                        print(jacinle.indent_text(str(body), level=2))
                elif isinstance(part, EffectPart):
                    suite = suite.get_effect_statements()
                    effect = CrowBehaviorOrderingSuite.make_sequential(*suite)

                    if _PARSER_VERBOSE:
                        print(jacinle.indent_text(f'Effect: {effect}'))
                elif isinstance(part, HeuristicPart):
                    heuristic = suite.get_heuristic_statements()
                    if len(heuristic) == 0:
                        heuristic = None
                    elif len(heuristic) == 1:
                        if not isinstance(heuristic[0], CrowBehaviorOrderingSuite) or heuristic[0].order.value != 'sequential':
                            heuristic = CrowBehaviorOrderingSuite('sequential', (heuristic[0],), _skip_simplify=True)
                    else:
                        heuristic = CrowBehaviorOrderingSuite('sequential', tuple(heuristic), _skip_simplify=True)
                else:
                    raise ValueError(f'Invalid part: {part}')

        if effect is None:
            effect = CrowBehaviorOrderingSuite.make_sequential()
        if len(goals) == 0:
            goal = E.NullExpression(BOOL)
            self.domain.define_behavior(name, args.arguments, goal, body, effect, heuristic=heuristic, minimize=minimize, **annotations)
        elif len(goals) == 1:
            self.domain.define_behavior(name, args.arguments, goals[0], body, effect, heuristic=heuristic, minimize=minimize, **annotations)
        else:
            goal = E.NullExpression(BOOL)
            self.domain.define_behavior(name, args.arguments, goal, body, effect, minimize=minimize, **annotations)
            for i, goal in enumerate(goals):
                self.domain.define_behavior(f'{name}_{i}', args.arguments, goal, body, effect, heuristic=heuristic, minimize=minimize, **annotations)

    @inline_args
    def generator_definition(self, annotations: Optional[dict], name: str, args: Optional[ArgumentsDef], *parts: Union[GoalPart, InPart, OutPart]):
        if annotations is None:
            annotations = dict()
        if args is None:
            args = ArgumentsDef(tuple())

        if _PARSER_VERBOSE:
            print(f'generator_definition:: {name=} {args.arguments=} {annotations=}')

        inputs = list()
        outputs = None
        goal = E.NullExpression(BOOL)

        with self.expression_def_ctx.with_variables(*args.arguments):
            for part in parts:
                suite = self.expression_interpreter.visit(part.suite)
                if isinstance(part, GoalPart):
                    goal = suite.get_derived_expression()
                    if _PARSER_VERBOSE:
                        print(jacinle.indent_text(f'Goal: {goal}'))
                elif isinstance(part, InPart):
                    inputs = [E.VariableExpression(x) for x in suite.items]
                    if _PARSER_VERBOSE:
                        print(jacinle.indent_text(f'Inputs: {inputs}'))
                elif isinstance(part, OutPart):
                    outputs = [E.VariableExpression(x) for x in suite.items]
                    if _PARSER_VERBOSE:
                        print(jacinle.indent_text(f'Outputs: {outputs}'))
                else:
                    raise ValueError(f'Invalid part: {part}')

        self.domain.define_generator(name, args.arguments, goal, inputs, outputs, **annotations)

    @inline_args
    def generator_goal_definition(self, suite: Tree) -> GoalPart:
        return GoalPart(suite)

    @inline_args
    def generator_in_definition(self, values: Tree) -> InPart:
        return InPart(values)

    @inline_args
    def generator_out_definition(self, values: Tree) -> OutPart:
        return OutPart(values)


class CDLProblemTransformer(CDLDomainTransformer):
    def __init__(self, domain: Optional[CrowDomain] = None, state: Optional[CrowState] = None, auto_constant_guess: bool = False):
        super().__init__(None, auto_init_domain=False)
        self._problem = None
        self.auto_constant_guess = auto_constant_guess

        self._domain_is_provided = False
        if domain is not None:
            self._domain_is_provided = True
            self._init_domain(domain, state)

    def _handle_pragma(self, pragma: Dict[str, Any]):
        super()._handle_pragma(pragma)
        for key, value in pragma.items():
            if key.startswith('planner_'):
                self.problem.set_planner_option(key[len('planner_'):], value)

    @property
    def problem(self) -> CrowProblem:
        return self._problem

    def _init_domain(self, domain: CrowDomain, state: Optional[CrowState] = None):
        if self._domain is not None:
            raise ValueError('Domain is already initialized. Cannot overwrite the domain.')

        self._domain = domain.clone(deep=False)
        self._problem = CrowProblem(domain=self.domain)
        self._expression_def_ctx = E.ExpressionDefinitionContext(domain=self.domain)
        self._expression_interpreter = CDLExpressionInterpreter(domain=self.domain, state=state, expression_def_ctx=self.expression_def_ctx, auto_constant_guess=self.auto_constant_guess)

        for o in self.domain.constants.values():
            if isinstance(o, ObjectConstant):
                self.problem.add_object(o.name, o.dtype.typename)
                self.expression_interpreter.local_variables[o.name] = E.ObjectConstantExpression(ObjectConstant(o.name, o.dtype))

    @inline_args
    def domain_def(self, filename: str):
        if self._domain is not None:
            if not self._domain_is_provided:
                logger.warning('Domain is already initialized. Skip the in-place domain loading.')
            return
        if filename == '__empty__':
            self._init_domain(CrowDomain())
        else:
            domain = get_default_parser().parse_domain(filename)
            self._init_domain(domain)

    @inline_args
    def problem_name(self, name: str):
        self._problem.name = name

    @inline_args
    def objects_definition(self, *objects):
        for o in objects:
            if isinstance(o, CSList):
                for oo in o.items:
                    self.problem.add_object(oo.name, oo.dtype.typename)
                    self.expression_interpreter.local_variables[oo.name] = E.ObjectConstantExpression(ObjectConstant(oo.name, oo.dtype))
            else:
                self.problem.add_object(o.name, o.dtype.typename)
                self.expression_interpreter.local_variables[o.name] = E.ObjectConstantExpression(ObjectConstant(o.name, o.dtype))

    @inline_args
    def init_definition(self, suite: Tree):
        self.problem.init_state()
        suite = self.expression_interpreter.visit(suite)
        executor = self.domain.make_executor()
        execute_effect_statements(executor, suite.get_effect_statements(), state=self.problem.state, scope=dict())

    @inline_args
    def goal_definition(self, suite: Tree):
        suite = self.expression_interpreter.visit(suite)
        suite = suite.get_derived_expression()
        self.problem.set_goal(suite)

