#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : regression_dependency.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/05/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Dependencies for the regression planning.

The CROW Regression planner provides a way to visualize the dependencies between the regression rules and statements.

Recall that a CROW-generated plan is roughly a tree where leaf nodes corresponds to "primitive" statements such as calling a controller function or setting a variable or making an assertion,
whereas the internal nodes correspond to "achieve" statements that is like a function call in other programming languages.

Therefore, we can visualize the dependencies between the statements in the plan as a tree structure. We use the following rules to transform the plan into a dependency graph:

- `achieve x` has two cases:
    - If `x` is already achieved at the state where this statement is executed, then this statement will be translated into an "assert" statement.
    - If `x` has not been achieved, and the planner translates `x` into a "behavior function call" statement with `derived_from` pointing to the original `achieve x` statement,
        and the `additional_info` field containing the note that how the subsequences corresponding to `x` is serialized.
        All subsequent statements that are derived from this `achieve x` statement will be connected to this statement in the dependency graph.
        The `additional_info` field will contain the serialized subsequences: if no promotion is needed, it will state that the refinement is "sequential",
        that is, the planner executes the statements after other subgoals are achieved. If promotion is needed, it will state that the refinement is "promoted".
- `behavior()` statements will be translated into a "behavior function call" statement. The `additional_info` field will contain the serialized information.
- `assert x` statements will be left as is.
- `do x` statements will be left as is.
- `let x` statements will be left as is.
- `bind x where cond(x)` statements will be translated into an assignment statement. The `derived_from` field will point to the original `bind` statement.
- `if cond(x): y else: z` statements will be translated into an assertion statement followed by the body of the `if` statement. The `derived_from` field will point to the original `if` statement.
    Depending on whether the condition is satisfied, the planner will translate the condition into either an assert` or `assert not` statement. The `additional_info` field also contains the result of the condition evaluation.Q
- `while cond(x) do y` statements will be translated into a sequence of `assert cond(x); y; assert cond(x); y; ...; assert not cond(x)` statements. The `derived_from` field will point to the original `while` statement.
- `foreach x: T: y` statements will be translated into a sequence of `let x = ...; y; let x = ...; y; ...` statements. The `derived_from` field will point to the original `foreach` statement.

Note that it is not recommended to visualize this tree structure as a typical "tree" because of the promotion mechanism. Instead, we should roughly consider it as a layered graph.
The leaf nodes correspond to the primitive statements, and they are ordered left-to-right in the same order as they are executed. Then, we build the internal nodes layer by layer,
while edge-crossing is allowed, to indicate that some subgoals have been "promoted" to an earlier time.

To use this module, you need to first generate the trace of the regression planner, which is a sequence of `RegressionTraceStatement` objects.
This can be turned on by setting `include_dependency_trace=True` in the `CrowRegressionPlanner` constructor.

Then, given the result of the plan, you can use the `recover_dependency_graph_from_trace` function to recover the dependency graph from the trace.

.. code-block:: python

    import concepts.dm.crow as crow

    planner = crow.crow_regression(problem.domain, problem, include_dependency_trace=True, return_planner=True)
    planner.main()
    for r in planner.results:
        graph = crow.recover_dependency_graph_from_trace(r.trace, r.scopes)
        graph.render_graphviz('graph.pdf')

The graph is rendered using `graphviz`. You can either specify a filename to the `render_graphviz` function to save the graph to a file (in PNG or PDF or DOT format).
If the filename is not specified, the graph will be saved to a temporary PDF file and opened in the default PDF viewer.

Note that in order to use this feature, you need to have `graphviz` installed. You can install it by running `pip install graphviz`. The `graphviz` package also
requires the `graphviz` binary to be installed on your system. You can install it by running `brew install graphviz` on macOS or `sudo apt-get install graphviz` on Ubuntu.

The `cdl-plan` tool also provides a command `--visualize-dependency-graph`. If this flag is set, the tool will generate a PDF visualization of the dependency graph and open it in the default PDF viewer.

.. code-block:: bash

    cdl-plan blocksworld-problem-sussman-with-pragma.cdl --visualize-dependency-graph

"""

import tempfile
import os
from dataclasses import dataclass
from typing import Optional, Sequence, List, Dict, TYPE_CHECKING

from jacinle.utils.printing import indent_text
from concepts.dm.crow.behavior_utils import format_behavior_statement

if TYPE_CHECKING:
    from concepts.dm.crow.planners.regression_planning import SupportedCrowExpressionType


@dataclass(unsafe_hash=True)
class RegressionTraceStatement(object):
    stmt: 'SupportedCrowExpressionType'
    scope_id: int = None
    new_scope_id: Optional[int] = None
    additional_info: Optional[str] = None

    scope: Optional[dict] = None
    new_scope: Optional[dict] = None
    derived_from: Optional['SupportedCrowExpressionType'] = None

    def node_string(self, scopes: Dict[int, dict]) -> str:
        stmt_scope = self.new_scope if self.new_scope is not None else self.scope
        basic_fmt = format_behavior_statement(self.stmt, scope=stmt_scope)

        if self.derived_from is not None:
            basic_fmt += '\n  derived from: ' + indent_text(format_behavior_statement(self.derived_from, scope=self.scope), 1, indent_first=False)
        if self.additional_info is not None:
            basic_fmt += '\n  note: ' + self.additional_info
        return basic_fmt


class RegressionDependencyGraph(object):
    def __init__(self, scopes: Dict[int, dict]):
        self.scopes = scopes
        self.nodes = list()
        self.node2index = dict()
        self.edges = dict()

    nodes: List[RegressionTraceStatement]
    node2index: Dict[int, int]
    edges: Dict[int, List[int]]

    def add_node(self, node: RegressionTraceStatement) -> 'RegressionDependencyGraph':
        self.nodes.append(node)
        self.node2index[id(node)] = len(self.nodes) - 1
        return self

    def connect(self, x: RegressionTraceStatement, y: RegressionTraceStatement) -> 'RegressionDependencyGraph':
        """Connect two nodes in the dependency graph. x is the "parent" of y.

        Args:
            x: the parent node.
            y: the child node.
        """
        self.edges.setdefault(id(x), list()).append(self.node2index[id(y)])
        return self

    def print(self, i: int = 0, indent_level: int = 0) -> None:
        print(indent_text(f'{i}::' + self.nodes[i].node_string(self.scopes), indent_level))
        for child in self.edges.get(id(self.nodes[i]), []):
            self.print(child, indent_level + 1)

    def get_node_ranks(self):
        ranks = dict()

        def dfs(i):
            max_rank = -1
            for child in self.edges.get(id(self.nodes[i]), []):
                max_rank = max(dfs(child), max_rank)
            ranks[i] = max(i, max_rank + 0.1)
            return ranks[i]

        dfs(0)
        return ranks

    def sort_nodes_into_levels(self):
        levels = dict()
        def dfs(i):
            max_level = -1
            for child in self.edges.get(id(self.nodes[i]), []):
                max_level = max(dfs(child), max_level)
            levels[i] = max_level + 1
            return max_level + 1
        max_level = dfs(0)

        output_levels = list()
        for i in range(max_level + 1):
            output_levels.append([j for j in range(len(self.nodes)) if levels[j] == i])
        return output_levels

    def render_graphviz(self, filename: Optional[str] = None) -> None:
        """Render the dependency graph using graphviz."""
        try:
            import graphviz
        except ImportError:
            raise ImportError('Please install graphviz first by running "pip install graphviz".')

        levels = self.sort_nodes_into_levels()
        dot = graphviz.Digraph(comment='Regression Dependency Graph')

        ranks = self.get_node_ranks()
        ranked_nodes = sorted(range(len(self.nodes)), key=lambda i: ranks[i])
        for i in ranked_nodes:
            node = self.nodes[i]
            dot.node(str(i), node.node_string(self.scopes).replace('\n', '\l') + '\l', shape='rectangle', ordering='out')

        node2level = dict()
        for i, level_nodes in enumerate(levels):
            for j in level_nodes:
                node2level[j] = i

        for i in range(len(levels)):
            dot.node(f'level_{i}', '', ordering='out', style='invis')
            levels[i] = sorted(levels[i], key=lambda j: ranks[j])
            for j in levels[i]:
                dot.edge(f'level_{i}', str(j), style='invis')

        for i in reversed(range(len(levels))):
            if i > 0:
                dot.edge(f'level_{i}', f'level_{i - 1}', style='invis')

        for x, ys in self.edges.items():
            for y in ys:
                dot.edge(str(self.node2index[x]), str(y))

        if filename is not None:
            if filename.endswith('.png'):
                actual_filename = filename[:-4]
                dot.render(actual_filename, format='png', cleanup=True)
                print(f'Graphviz file saved to "{filename}".')
            elif filename.endswith('.pdf'):
                actual_filename = filename[:-4]
                dot.render(actual_filename, format='pdf', cleanup=True)
                print(f'Graphviz file saved to "{filename}".')
            elif filename.endswith('.dot'):
                dot.render(filename)
                print(f'Graphviz file saved to "{filename}".')
            else:
                raise ValueError(f'Unsupported file format: {filename}. Only PNG, PDF, and DOT are supported.')
        else:
            with tempfile.NamedTemporaryFile(suffix='.pdf') as f:
                dot.render(f.name[:-4], format='pdf', cleanup=True)
                print(f'Graphviz file saved to "{f.name}". Now opening it in the default PDF viewer...')
                os.system(f'open "{f.name}"')
                import time; time.sleep(3)  # We need to sleep for a while to prevent the file from being deleted too early.


def recover_dependency_graph_from_trace(trace: Sequence[RegressionTraceStatement], scopes: Dict[int, dict]) -> RegressionDependencyGraph:
    graph = RegressionDependencyGraph(scopes)
    scope_to_node = dict()

    graph.add_node(trace[0])
    scope_to_node[trace[0].new_scope_id] = trace[0]

    for stmt in trace[1:]:
        graph.add_node(stmt)
        if stmt.scope_id in scope_to_node:
            graph.connect(scope_to_node[stmt.scope_id], stmt)
        if stmt.new_scope_id is not None:
            scope_to_node[stmt.new_scope_id] = stmt

    return graph
