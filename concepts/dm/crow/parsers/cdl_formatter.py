#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : cdl_formatter.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/11/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from concepts.dm.crow.parsers.cdl_parser import get_default_parser


def lex_cdl_file(filename: str):
    with open(filename, 'r') as f:
        return list(get_default_parser().parser.lex(f.read()))


def lex_cdl_string(string: str):
    return list(get_default_parser().parser.lex(string))


def _color(r, g, b):
    return '#' + str(hex(r))[2:].zfill(2) + str(hex(g))[2:].zfill(2) + str(hex(b)[2:].zfill(2))


def get_token_styles():
    styles = {
        0: '#c24f1a',  # orange
        1: '#c24f1a',  # orange
        2: '#308822',  # dark green
        3: '#538135',  # green
        4: _color(230, 219, 116),  # light yellow
        5: '#2e75b5',             # blue
        6: _color(60, 60, 60),
    }


    # Token style is a number.
    # style % 10 === its color
    # style // 10 === its weight (1 for bold)
    token_styles = {
        "COLON": 5,
        "SEMICOLON": 5,
        "COMMA": 2,
        "LBRACE": 5,
        "RBRACE": 5,
        "LSQB": 5,
        "RSQB": 5,
        '_LLSQB': 5,  # [[
        '_RRSQB': 5,  # ]]
        "_RIGHT_ARROW": 5,  # ->

        "TRUE": 0,
        "FALSE": 0,
        "NULL": 0,

        'ELLIPSIS': 5,

        "STRING": 1,
        "LONG_STRING": 1,
        "NUMBER": 1,
        "DEC_NUMBER": 1,
        "BIN_NUMBER": 1,
        "OCT_NUMBER": 1,
        "HEX_NUMBER": 1,
        'ESCAPED_STRING': 1,
        'SIGNED_NUMBER': 1,

        'LPAR': 3,
        'RPAR': 3,

        'PLUS': 3,
        'MINUS': 3,
        'STAR': 3,
        'FLOORDIV': 3,
        'TILDE': 3,
        'AT': 3,
        'SLASH': 3,
        'PERCENT': 3,
        'EQUAL': 3,
        'DOUBLE_EQUAL': 3,  # ==
        'NOT_EQUAL': 3,  # !=
        'GREATER_EQ': 3,  # >=
        'LESS_EQ': 3,  # <=
        "RSHIFT": 3,  # >>
        "LSHIFT": 3,  # <<
        'CIRCUMFLEX': 3,

        'EXISTS': 1,
        'FORALL': 1,
        'FOREACH': 1,
        'FINDALL': 1,
        'FOR': 1,
        'IF': 1,
        'ELSE': 1,
        'AND': 1,
        'OR': 1,
        'NOT': 1,
    }

    for keyword in {
        'LOCAL', 'LET', 'EXPR', 'RETURN', 'PASS',
        'PREAMBLE', 'PROMOTABLE',
        'DO', 'BIND', 'ACHIEVE', 'PACHIEVE', 'ASSERT', 'WHERE',
        'ACHIEVE_ONCE', 'ACHIEVE_HOLD', 'PACHIEVE_ONCE', 'PACHIEVE_HOLD', 'ASSERT_ONCE', 'ASSERT_HOLD', 'UNTRACK',
        'SEQUENTIAL', 'UNORDERED', 'CRITICAL',
        'VECTOR',
    }:
        token_styles[keyword] = 2

    for keyword in {
        '_PRAGMA_KEYWORD',  # #!pragma
        'DOMAIN', 'PROBLEM',
        'TYPEDEF', 'FEATURE', 'DEF', 'GENERATOR', 'UNDIRECTED_GENERATOR', 'CONTROLLER', 'ACTION', 'BEHAVIOR',
        'OBJECTS', 'GOAL', 'INIT',
        'EFFECT', 'EFF', 'PRE', 'BODY', 'IN', 'OUT', 'IS',
    }:
        token_styles[keyword] = 12

    return styles, token_styles


def check_cdl_token_styles(string):
    _, g_token_styles = get_token_styles()
    other_types = set()
    for token in lex_cdl_string(string):
        if token.type in ('_INDENT', '_DEDENT', '_NEWLINE', 'BASIC_TYPENAME', 'VARNAME'):
            pass
        elif token.type not in g_token_styles:
            other_types.add(token.type)
    if len(other_types) > 0:
        print('Unknown token types:', other_types)


def format_cdl_html(string, input_file: str = '<cdl>'):
    tokens = lex_cdl_string(string)
    g_styles, g_token_styles = get_token_styles()

    fmt = ''
    fmt += f'<title>{input_file}</title>'
    fmt += f'<body style="background: {_color(255,255,255)}; color: {_color(192, 192, 192)}; font-family: monospace; "><div>'
    last_index = 0
    for token in tokens:
        if token.start_pos is None:
            continue
        if token.start_pos > last_index:
            this_string = string[last_index:token.start_pos].replace(' ', '&nbsp;').replace("\n", "<br />")
            fmt += f'<span style="color: {_color(192,192,192)}">{this_string}</span>'

        if token.type == '_NEWLINE':
            fmt += token.value.replace(' ', '&nbsp;').replace("\n", "<br />")
        elif token.type in ('_INDENT', '_DEDENT'):
            continue
        elif token.type in g_token_styles:
            style = g_token_styles[token.type]
            this_color = g_styles[style % 10]
            weight_str = 'font-weight: 700' if style >= 10 else ''
            fmt += f'<span data-type="{token.type}" style="color: {this_color}; {weight_str}">{token.value}</span>'
        else:
            fmt += f'<span style="color:{_color(60,60,60)}">{token.value}</span>'
        last_index = token.start_pos + len(token.value)
    fmt += '</div>'

    return fmt


def format_cdl_latex(string):
    tokens = lex_cdl_string(string)
    g_styles, g_token_styles = get_token_styles()

    fmt = ''

    def colored_print(text, r, g, b, weight=False):
        nonlocal fmt
        text = text.replace('\\', '\\textbackslash').replace('{', '\\{').replace('}', '\\}')
        text = text.replace('_', '\\_').replace('^', '\\^').replace('%', '\\%')
        text = text.replace('&', '\\&').replace('$', '\\$').replace('#', '\\#')
        text = text.replace('<', '\\textless').replace('>', '\\textgreater')

        text = text.replace(' ', '~')
        if r == -1:
            text = text.replace('\n', '~\\\\%\n')
            if weight:
                fmt += f'{text}'
            else:
                fmt += f'{text}'
        else:
            parts = text.split('\n')
            for i, part in enumerate(parts):
                if len(part) == 0 and i < len(parts) - 1:
                    fmt += '~\\\\%\n'
                else:
                    if weight:
                        fmt += f'\\textcolor[RGB]{{{r},{g},{b}}}{{{part}}}'
                    else:
                        fmt += f'\\textcolor[RGB]{{{r},{g},{b}}}{{{part}}}'
                    if i < len(parts) - 1:
                        fmt += '~\\\\%\n'

    last_index = 0
    for token in tokens:
        if token.start_pos > last_index:
            this_string = string[last_index:token.start_pos]
            colored_print(this_string, 192, 192, 192)

        if token.type == '_NEWLINE':
            this_string = token.value
            colored_print(this_string, 192, 192, 192)
        elif token.type in ('_INDENT', '_DEDENT'):
            continue
        elif token.type in g_token_styles:
            style = g_token_styles[token.type]
            this_color = g_styles[style % 10]
            colored_print(token.value, int(this_color[1:3], 16), int(this_color[3:5], 16), int(this_color[5:7], 16), style > 10)
        else:
            colored_print(token.value, -1, -1, -1)
        last_index = token.start_pos + len(token.value)

    header = r'''\noindent\fbox{%
    \parbox{\textwidth}{\tt\small%
'''
    footer = r'''}%
}%'''
    header = r'''\begin{mdframed}\tt\small'''
    footer = r'''\end{mdframed}%'''
    fmt = header + fmt + footer

    return fmt


def print_cdl_terminal(string):
    try:
        import colored
    except ImportError:
        print('Please install colored package to enable terminal highlighting.')
        return

    def colored_print(text, r, g, b):
        print(colored.stylize(text, colored.fore_rgb(r, g, b)), end='')

    tokens = lex_cdl_string(string)
    g_styles, g_token_styles = get_token_styles()

    last_index = 0
    for token in tokens:
        if token.start_pos is None:
            continue
        if token.start_pos > last_index:
            this_string = string[last_index:token.start_pos]
            colored_print(this_string, 192, 192, 192)

        if token.type == '_NEWLINE':
            colored_print(token.value, 192, 192, 192)
        elif token.type in ('_INDENT', '_DEDENT'):
            continue
        elif token.type in g_token_styles:
            style = g_token_styles[token.type]
            this_color = g_styles[style % 10]
            colored_print(token.value, int(this_color[1:3], 16), int(this_color[3:5], 16), int(this_color[5:7], 16))
        else:
            print(token.value, end='')
        last_index = token.start_pos + len(token.value)
