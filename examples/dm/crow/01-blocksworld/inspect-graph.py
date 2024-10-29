#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : inspect-graph.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/19/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import jacinle

parser = jacinle.JacArgumentParser()
parser.add_argument('graph', type=str)
args = parser.parse_args()


def main():
    graph = jacinle.load(args.graph)

    nodes = graph['nodes']
    edges = graph['edges']

    node_index2key = {i: key for i, key in enumerate(nodes.keys())}
    node_key2index = {key: i for i, key in enumerate(nodes.keys())}

    while True:
        node_id = input('Node ID: ')
        if node_id == 'q':
            break

        try:
            node_id = int(node_id)
        except:
            print('Invalid node ID.')
            continue

        key = node_index2key[node_id]
        print('Node:', key)
        print('Program:')
        print(nodes[key].state.print())
        print('Edges:')
        for edge in edges:
            if edge[1] == key:
                print('  <-', node_key2index[edge[0]], edge[0])
            if edge[0] == key:
                print('  ->', node_key2index[edge[1]], edge[1], edge[2] if len(edge) > 2 else '')


if __name__ == '__main__':
    main()

