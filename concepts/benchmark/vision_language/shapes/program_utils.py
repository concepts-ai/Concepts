#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : program_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/07/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Union, Tuple

__all__ = ['find_matching_bracket', 'parse_lisp_expr', 'gen_fol_from_lisp_expr']


def find_matching_bracket(program: str, start: int):
    assert program[start] == '('
    depth = 0
    for i in range(start, len(program)):
        if program[i] == '(':
            depth += 1
        elif program[i] == ')':
            depth -= 1
            if depth == 0:
                return i
    raise ValueError('No matching bracket found.')


def parse_lisp_expr(program) -> Tuple[Union[str, Tuple[str, ...]], ...]:
    assert program[0] == '('
    assert program[-1] == ')'
    program = program[1:-1]

    ret = []
    cur = ''
    depth = 0
    for c in program:
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif c == ' ' and depth == 0:
            if len(cur) > 0:
                ret.append(cur if cur[0] != '(' else parse_lisp_expr(cur))
                cur = ''
            continue
        cur += c
    if len(cur) > 0:
        ret.append(cur if cur[0] != '(' else parse_lisp_expr(cur))
    return tuple(ret)


def gen_fol_from_lisp_expr(program: str) -> str:
    def _gen_fol_from_lisp(program: Union[str, Tuple[str, ...]], current_var: str = 'a') -> str:
        if isinstance(program, str):
            return f'{program}({current_var})'
        else:
            if program[0] in ('left_of', 'right_of', 'above', 'below'):
                next_var = chr(ord(current_var) + 1)
                inner = _gen_fol_from_lisp(program[1], next_var)
                return f'exists(Object, lambda {next_var}: {program[0]}({current_var}, {next_var}) and {inner})'
            elif program[0] == 'is':
                inner = _gen_fol_from_lisp(program[2], current_var)
                return f'exists(Object, lambda {current_var}: {program[1]}({current_var}) and {inner})'
            else:
                raise ValueError(f'Unknown function: {program[0]}.')

    return _gen_fol_from_lisp(parse_lisp_expr(program))

