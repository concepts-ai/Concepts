" Vim syntax file
" Language: CDL
" Author: Jiayuan Mao
" Description: Syntax highlighting for CDL
" Usage: Put this file in ~/.vim/syntax/ and add the following line to ~/.vimrc:
"  au BufRead,BufNewFile *.cdl set filetype=cdl | set tabstop=2 shiftwidth=2 expandtab

if exists("b:current_syntax")
  finish
endif

syn case match

" Keywords
syn keyword CDLKeyword #!pragma achieve achieve_hold achieve_once action alternative assert assert_hold assert_once behavior bind body certifies commit controller critical def do domain eff else exists expr feature findall for forall foreach generator goal heuristic include if in init let local minimize objects out pass pragma pre preamble problem promotable return sequential typedef undirected_generator unordered untrack vector where while

syn keyword CDLSpecialKeyword False None True and bool float32 int64 not object or pyobject string

" Numbers (integer and floating-point)
syn match CDLNumber "\v\d+\.\d+|\d+"

" Strings (double-quoted)
syn match CDLString "\"[^\"]*\""

" Comments (starting with #)
syn match CDLComment "#.*"

" Highlighting groups
hi def link CDLKeyword Keyword
hi def link CDLSpecialKeyword Type
hi def link CDLNumber Number
hi def link CDLString String
hi def link CDLComment Comment

let b:current_syntax = "cdl"


