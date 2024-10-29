#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : regression_rule.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/29/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import itertools
from typing import TYPE_CHECKING, Any, Optional, Union, Iterable, Sequence, Tuple, List, Mapping, Callable

from jacinle.utils.enum import JacEnum
from jacinle.utils.printing import indent_text
from jacinle.utils.cache import cached_property
from jacinle.utils.meta import repr_from_str

from concepts.dsl.dsl_types import ObjectType, NamedTensorValueType, Variable, ObjectConstant, UnnamedPlaceholder
from concepts.dsl.expression import ConstantExpression, ValueOutputExpression, AndExpression, FunctionApplicationExpression, AssignExpression, VariableExpression, ListExpansionExpression
from concepts.dsl.expression_utils import find_free_variables
from concepts.dsl.tensor_value import TensorValue
from concepts.dm.pdsketch.operator import Precondition, Effect, OperatorApplicationExpression, filter_static_grounding
from concepts.dm.pdsketch.generator import GeneratorApplicationExpression

if TYPE_CHECKING:
    pass

__all__ = [
    'SubgoalSerializability', 'SubgoalCSPSerializability', 'RegressionCommitFlag',
    'AchieveExpression', 'BindExpression', 'RuntimeAssignExpression',
    'ConditionalRegressionRuleBodyExpression', 'LoopRegressionRuleBodyExpression',
    'RegressionRuleBodyItemType',
    'RegressionRule', 'RegressionRuleApplier', 'RegressionRuleApplicationExpression',
    'gen_all_grounded_regression_rules'
]


class SubgoalSerializability(JacEnum):
    """The subgoal serializability of a regression rule. By definition, a regression rule is called strongly serializable if
    all its subgoal prefix refinements can be extended to a new subgoal prefix refinement plan by adding a new subgoal at the end.

    This serializability can be used to weakened in two ways:
        - Not all subgoal prefix refinements can be extended. Therefore, the algorithm needs to keep track of a set of possible subgoal prefix refinements.
        - When the new subgoal is added to the end and being refined, some of the preconditions corresponding to that subgoal might need to be promoted to an earlier subgoal.

    If we use the first mechanism to weaken the serializability, we call the regression rule-weakly serializable (SubgoalSerializability.RULE).
    If we use the second mechanism to weaken the serializability, we call the regression order-weakly serializable (SubgoalSerializability.ORDER).
    If we use both mechanisms to weaken the serializability, we call the regression weakly serializable (SubgoalSerializability.WEAK).
    """

    STRONG = 'strong'
    RULE = 'rule'
    ORDER = 'order'
    WEAK = 'weak'


class SubgoalCSPSerializability(JacEnum):
    FORALL = 'forall'
    SOME = 'some'
    NONE = 'none'


class RegressionCommitFlag(object):
    def __init__(self, goal_serializability: str = 'STRONG', csp_serializability: str = 'FORALL', goal: bool = False):
        self.goal_serializability = SubgoalSerializability.from_string(goal_serializability)
        self.csp_serializability = SubgoalCSPSerializability.from_string(csp_serializability)
        self.include_goal = goal

    goal_serializability: SubgoalSerializability

    csp_serializability: SubgoalCSPSerializability

    include_goal: bool
    """Whether to include the current subgoal while committing the CSP."""

    def __str__(self):
        return f'Commit![goal_serializability={self.goal_serializability}, csp_serializability={self.csp_serializability}, include_goal={self.include_goal}]'

    def __repr__(self):
        return f'RegressionCommitFlag(goal_serializability={self.goal_serializability}, csp_serializability={self.csp_serializability}, include_goal={self.include_goal})'


class AchieveExpression(object):
    """An achieve expression. This is used in the definition of regression rules. Each achieve expression has two properties: serializability and csp_serializability.

        - serializability: 'strong', 'rule', or 'order'. Strong serializability means that the achieve expression is serializable for any state that satisfies the preconditions
          of the achieve expression, i.e., all achieve expressions prior to the current achieve expression. Rule serializability means that the achieve expression is serializable
          only for some refinement of the achieve expressions prior to the current achieve expression. However, there exists some refinement for previous expressions such that
          this new achieve expression can be planned based on the results of previous subgoals. Order serializability means that this subgoal is only serializable with respect
          to the order of the subgoals. Therefore, when plan for the sequence of subgoals, the order of the subgoals are preserved, but the preconditions for the subsequent
          rules might need to be promoted to a previous point in the plan. There is no absolute comparisons between the "rule" and "order" serializability, they are weaker
          than the "strong" serializability in different ways.

        - csp_serializability: 'some' or 'forall'. Some continuous serializability means that the achieve expression is serializable for some continuous values of the action parameters.
          Forall continuous serializability means that the achieve expression is serializable for all continuous values of the action parameters.
    """

    def __init__(self, goal: ValueOutputExpression, maintains: Sequence[ValueOutputExpression], serializability: Union[str, SubgoalSerializability] = 'strong', csp_serializability: Union[str, SubgoalCSPSerializability] = 'none', once: bool = False):
        self.goal = goal
        self.maintains = tuple(maintains)
        self.serializability = SubgoalSerializability.from_string(serializability)
        self.csp_serializability = SubgoalCSPSerializability.from_string(csp_serializability)
        self.once = once

    goal: ValueOutputExpression
    """The goal of the achieve expression."""

    maintains: Tuple[ValueOutputExpression, ...]
    """The list of maintain expressions."""

    serializability: SubgoalSerializability
    """The serializability of the achieve expression: 'strong', 'rule', or 'order'."""

    csp_serializability: SubgoalCSPSerializability
    """The continuous serializability of the achieve expression: 'some' or 'forall' or 'none'."""

    once: bool
    """Whether the goal to be achieved is a one-time achievement or should we keep "holding" the goal."""

    @property
    def sequential_decomposable(self) -> bool:
        return self.serializability in (SubgoalSerializability.STRONG, SubgoalSerializability.RULE)

    @property
    def refinement_compressible(self) -> bool:
        return self.serializability in (SubgoalSerializability.STRONG, SubgoalSerializability.ORDER)

    def __str__(self):
        return f'achieve({self.goal}, serializability={self.serializability.value}, csp_serializability={self.csp_serializability.value}, maintain={{{" ".join([str(m) for m in self.maintains])}}})'

    __repr__ = repr_from_str


class BindExpression(object):
    """A bind expression. This is used in the definition of regression rules."""

    def __init__(self, variables: Sequence[Variable], goal: Union[ValueOutputExpression, GeneratorApplicationExpression], serializability: Union[str, SubgoalSerializability] = 'strong', csp_serializability: Union[str, SubgoalCSPSerializability] = 'none', ordered: bool = True):
        self.variables = tuple(variables)
        self.goal = goal
        self.serializability = SubgoalSerializability.from_string(serializability)
        self.csp_serializability = SubgoalCSPSerializability.from_string(csp_serializability)
        self.ordered = ordered
        self.is_object_bind_expression = self._compute_is_object_bind()

    variables: Tuple[Variable, ...]
    """The variables to be found."""

    goal: Union[ValueOutputExpression, GeneratorApplicationExpression]
    """The goal of the bind expression."""

    serializability: SubgoalSerializability
    """The serializability of the bind expression: 'strong', 'rule', or 'order'."""

    csp_serializability: SubgoalCSPSerializability
    """The continuous serializability of the bind expression: 'some' or 'forall'."""

    is_object_bind_expression: bool
    """Whether the bind expression is an object bind expression (i.e., all variables are object variables)."""

    def _compute_is_object_bind(self):
        nr_object_variables = 0
        nr_value_variables = 0

        for variable in self.variables:
            if isinstance(variable.dtype, ObjectType):
                nr_object_variables += 1
            else:
                nr_value_variables += 1

        if nr_object_variables > 0 and nr_value_variables > 0:
            raise ValueError('A bind expression cannot have both object and value variables.')

        is_object_bind = nr_object_variables > 0
        return is_object_bind

    @property
    def sequential_decomposable(self) -> bool:
        return self.serializability in (SubgoalSerializability.STRONG, SubgoalSerializability.RULE)

    @property
    def refinement_compressible(self) -> bool:
        return self.serializability in (SubgoalSerializability.STRONG, SubgoalSerializability.ORDER)

    ordered: bool
    """Whether this expression participates in the ordering of the variable orderings."""

    def __str__(self):
        return f'bind({{{", ".join([str(v) for v in self.variables])}}}, {self.goal}, serializability={self.serializability.value}, csp_serializability={self.csp_serializability.value})'

    __repr__ = repr_from_str


class RuntimeAssignExpression(object):
    def __init__(self, variable: Variable, value: ValueOutputExpression):
        self.variable = variable
        self.value = value

    variable: Variable
    """The variable to be assigned."""

    value: ValueOutputExpression
    """The value to be assigned to the variable."""

    def __str__(self):
        return f'runtime_assign({self.variable}, {self.value})'

    __repr__ = repr_from_str


RegressionRuleBodyItemType = Union[
    AchieveExpression,        # for achieving a subgoal.
    BindExpression,           # for binding a set of variables that satisfy a condition.
    RuntimeAssignExpression,  # For assigning a value to a variable at runtime.
    ListExpansionExpression,  # for applying a function to generate a sequence of regression steps.
    RegressionCommitFlag,     # for committing the variables in the CSP.
    OperatorApplicationExpression,          # for directly applying an operator.
    'RegressionRuleApplicationExpression',  # for directly recursively applying a regression rule.
    'ConditionalRegressionRuleBodyExpression',  # for conditional regression rule body expression.
    'LoopRegressionRuleBodyExpression'          # for loop regression rule body expression.
]


class ConditionalRegressionRuleBodyExpression(object):
    """A conditional regression rule body. For example, in the definition of a regression rule, we can have a conditional body expression like this:

    .. code-block:: python

        ConditionalRegressionRuleBodyExpression(
            condition=condition,
            body=[
                OperatorApplicationExpression(...),
                AchieveExpression(...),
                ...
            ]
        )
    """
    def __init__(self, condition: ValueOutputExpression, body: Sequence[RegressionRuleBodyItemType]):
        self.condition = condition
        self.body = tuple(body)

    condition: ValueOutputExpression
    """The condition of the conditional regression rule body expression."""

    body: Tuple[RegressionRuleBodyItemType, ...]
    """The body of the conditional regression rule body expression."""

    def __str__(self):
        fmt = f'if {self.condition} then'
        for item in self.body:
            fmt += f'\n{indent_text(str(item))}'
        return fmt

    __repr__ = repr_from_str


class LoopRegressionRuleBodyExpression(object):
    """A while-loop regression rule body. For example, in the definition of a regression rule, we can have a loop body expression like this:

    .. code-block:: python

        LoopRegressionRuleBodyExpression(
            condition=condition,
            body=[
                OperatorApplicationExpression(...),
                AchieveExpression(...),
                ...
            ]
        )
    """
    def __init__(self, condition: ValueOutputExpression, body: Sequence[RegressionRuleBodyItemType]):
        self.condition = condition
        self.body = tuple(body)

    condition: ValueOutputExpression
    """The condition of the loop regression rule body expression."""

    body: Tuple[RegressionRuleBodyItemType, ...]
    """The body of the loop regression rule body expression."""

    def __str__(self):
        fmt = f'while {self.condition} do'
        for item in self.body:
            fmt += f'\n{indent_text(str(item))}'
        return fmt

    __repr__ = repr_from_str


class RegressionRule(object):
    BodyItemType = RegressionRuleBodyItemType

    def __init__(
        self,
        name: str,
        parameters: Sequence[Variable],
        preconditions: Sequence[Precondition],
        goal_expression: ValueOutputExpression,
        all_effects: Sequence[Effect],
        body: Sequence[BodyItemType],
        always: bool = False
    ):
        """Initialize a regression rule."""

        self.name = name
        self.arguments = tuple(parameters)
        self.preconditions = tuple(preconditions)
        self.preconditions_conjunction = AndExpression(*[p.bool_expr for p in self.preconditions])
        self.goal_expression = goal_expression
        self.all_effects = tuple(all_effects)
        self.body = tuple(body)
        self.always = always

        self.goal_arguments, self.binding_arguments = self._split_arguments()
        self.max_reorder_prefix_length = self._compute_max_reorder_prefix_length()
        self.max_rule_prefix_length = self._compute_max_rule_prefix_length()

        assert self.max_reorder_prefix_length <= self.max_rule_prefix_length or self.max_rule_prefix_length == 0

    name: str
    """The name of the regression rule."""

    arguments: Tuple[Variable, ...]
    """The arguments to the regression rule."""

    goal_arguments: Tuple[Variable, ...]
    """The arguments that appear in the goal of the regression rule."""

    binding_arguments: Tuple[Variable, ...]
    """The arguments that do not appear in the goal of the regression rule."""

    preconditions: Tuple[Precondition, ...]
    """The preconditions of the regression rule."""

    preconditions_conjunction: AndExpression
    """The conjunction of the preconditions of the regression rule."""

    goal_expression: ValueOutputExpression
    """The goal expression of the regression rule."""

    all_effects: Tuple[Effect, ...]
    """The side effects of the regression rule."""

    body: Tuple[RegressionRuleBodyItemType, ...]
    """The body of the regression rule, including operator applications and achieve expressions."""

    always: bool
    """Whether the regression rule is always applicable."""

    max_reorder_prefix_length: int
    """The maximum length of the prefix of the body that might need to be reordered."""

    max_rule_prefix_length: int
    """The maximum length of the prefix of the body that need tracking of all possible refinements."""

    @property
    def nr_arguments(self) -> int:
        """The number of arguments of the regression rule."""
        return len(self.arguments)

    @property
    def argument_names(self) -> Tuple[str, ...]:
        """The names of the arguments of the regression rule."""
        return tuple(arg.name for arg in self.arguments)

    @property
    def argument_types(self) -> Tuple[Union[ObjectType, NamedTensorValueType], ...]:
        """The types of the arguments of the regression rule."""
        return tuple(arg.dtype for arg in self.arguments)

    @property
    def serializability(self):
        return 'always' if self.always else 'sometimes'

    def iter_effects(self) -> Iterable[Effect]:
        if _is_cacheable_fluent(self.goal_expression):
            yield Effect(AssignExpression(self.goal_expression, ConstantExpression.TRUE))
        yield from self.all_effects

    def _split_arguments(self) -> Tuple[Tuple[Variable, ...], Tuple[Variable, ...]]:
        goal_variables = find_free_variables(self.goal_expression)
        goal_variable_names = set(v.name for v in goal_variables)
        binding_variables = [arg for arg in self.arguments if arg.name not in goal_variable_names]
        return tuple(goal_variables), tuple(binding_variables)

    def _compute_max_reorder_prefix_length(self) -> int:
        max_reorder_prefix_length = len(self.body)
        while max_reorder_prefix_length > 0:
            item = self.body[max_reorder_prefix_length - 1]
            if isinstance(item, AchieveExpression) and item.serializability is SubgoalSerializability.ORDER:
                break
            max_reorder_prefix_length -= 1
        return max_reorder_prefix_length

    def _compute_max_rule_prefix_length(self) -> int:
        max_rule_prefix_length = len(self.body)
        while max_rule_prefix_length > 0:
            item = self.body[max_rule_prefix_length - 1]
            if isinstance(item, AchieveExpression) and item.serializability is SubgoalSerializability.RULE:
                break
            max_rule_prefix_length -= 1
        return max_rule_prefix_length

    def __str__(self) -> str:
        return f'{self.name}({", ".join(str(param) for param in self.arguments)}, always={self.always})'

    __repr__ = repr_from_str

    def pddl_str(self) -> str:
        body_str = '\n'.join([indent_text(str(action), 1, tabsize=3) for action in self.body])
        precondition_string = f'\n :precondition (and {" ".join(str(precondition) for precondition in self.preconditions)})' if len(self.preconditions) > 0 else ''
        side_effect_string = f'\n :all-effects (and {" ".join(str(effect) for effect in self.all_effects)})' if len(self.all_effects) > 0 else ''
        return f'''(:regression {self.name} [always={self.always}]
 :parameters ({', '.join(str(param) for param in self.arguments)})
 :goal {str(self.goal_expression)}{precondition_string}{side_effect_string}
 :rule (then
   {body_str.lstrip()}
 )
)'''

    def __call__(self, *args) -> 'RegressionRuleApplier':
        """Ground the operator with a list of arguments."""
        output_args = list()
        if len(args) > 0 and args[-1] is Ellipsis:
            args = args[:-1] + ('??', ) * (self.nr_arguments - len(args) + 1)
        for i, arg in enumerate(args):
            if isinstance(arg, str) and arg == '??':
                output_args.append(UnnamedPlaceholder(self.arguments[i].dtype))
            else:
                output_args.append(arg)
        return RegressionRuleApplier(self, tuple(output_args))


class RegressionRuleApplier(object):
    def __init__(
        self, regression_rule: RegressionRule, arguments: Sequence[Union[Variable, TensorValue, Any]],
        maintains: Optional[Sequence[ValueOutputExpression]] = None,
        serializability: Union[SubgoalSerializability, str] = 'strong', csp_serializability: Union[SubgoalCSPSerializability, str] = 'none'
    ):
        self.regression_rule = regression_rule
        self.arguments = tuple(arguments)
        self.maintains = tuple(maintains) if maintains is not None else tuple()
        self.serializability = SubgoalSerializability.from_string(serializability)
        self.csp_serializability = SubgoalCSPSerializability.from_string(csp_serializability)

    regression_rule: RegressionRule
    """The regression rule that is applied."""

    arguments: Tuple[Union[Variable, TensorValue, Any], ...]
    """The arguments of the regression rule."""

    maintains: Tuple[ValueOutputExpression, ...]
    """The maintain expressions of the regression rule."""

    serializability: SubgoalSerializability
    """The serializability of the regression rule."""

    csp_serializability: SubgoalCSPSerializability
    """The continuous serializability of the regression rule."""

    @cached_property
    def goal_expression(self) -> ValueOutputExpression:
        """The goal of the regression rule."""
        from concepts.dm.pdsketch.regression_utils import ground_fol_expression_v2
        arguments = [ObjectConstant(arg, arg_def.dtype) if isinstance(arg, str) else arg for arg, arg_def in zip(self.arguments, self.regression_rule.arguments)]
        return ground_fol_expression_v2(self.regression_rule.goal_expression, {arg_def.name: arg for arg_def, arg in zip(self.regression_rule.arguments, arguments)})

    def __str__(self) -> str:
        return f'{self.regression_rule.name}({", ".join(str(arg) for arg in self.arguments)})'

    __repr__ = repr_from_str

    def pddl_str(self) -> str:
        arg_str = ' '.join([str(arg) for arg in self.arguments])
        return f'({self.regression_rule.name} {arg_str})'


class RegressionRuleApplicationExpression(object):
    """An abstract regression rule grounding. For example, in :code:`(regression-ontop ?x ?y)` where :code:`?x` and :code:`?y` are variables in the context."""

    def __init__(
        self, regression_rule: 'RegressionRule', arguments: Sequence[Union[VariableExpression, UnnamedPlaceholder, ValueOutputExpression]],
        maintains: Optional[Sequence[ValueOutputExpression]] = None,
        serializability: Union[SubgoalSerializability, str] = 'strong', csp_serializability: Union[SubgoalCSPSerializability, str] = 'none'
    ):
        self.regression_rule = regression_rule
        self.arguments = tuple(arguments)
        self.maintains = tuple(maintains) if maintains is not None else tuple()
        self.serializability = SubgoalSerializability.from_string(serializability)
        self.csp_serializability = SubgoalCSPSerializability.from_string(csp_serializability)

    regression_rule: 'RegressionRule'
    """The regression rule that is applied."""

    arguments: Tuple[Union[VariableExpression, UnnamedPlaceholder, ValueOutputExpression], ...]
    """The arguments of the regression rule."""

    maintains: Tuple[ValueOutputExpression, ...]
    """The maintain expressions of the regression rule."""

    serializability: SubgoalSerializability
    """The serializability of the regression rule."""

    csp_serializability: SubgoalCSPSerializability
    """The continuous serializability of the regression rule."""

    def ground(self, executor: 'PDSketchExecutor') -> 'RegressionRuleApplier':
        """Ground the regression rule statement."""
        return RegressionRuleApplier(self.regression_rule, tuple(executor.execute(arg) for arg in self.arguments))

    @property
    def name(self) -> str:
        """The name of the regression rule."""
        return self.regression_rule.name

    def __str__(self) -> str:
        def_name = 'regression'
        arg_string = ', '.join([
            arg_def.name + '=' + str(arg)
            for arg_def, arg in zip(self.regression_rule.arguments, self.arguments)
        ])
        return f'{def_name}::{self.regression_rule.name}({arg_string})'

    __repr__ = repr_from_str

    def pddl_str(self) -> str:
        arg_str = ' '.join([str(arg) for arg in self.arguments])
        return f'({self.regression_rule.name} {arg_str})'


def gen_all_grounded_regression_rules(
    executor: 'PDSketchExecutor', state: 'State', continuous_values: Optional[Mapping[str, Iterable[TensorValue]]] = None,
    regression_rule_names: Optional[Sequence[str]] = None, regression_rule_filter: Optional[Callable[[RegressionRuleApplier], bool]] = None, filter_static: bool = True,
) -> List[RegressionRuleApplier]:
    """Generate all grounded regression rules applicable in an environment, given the initial state.
    Note that this function does not check if the action is applicable at the current state.

    Args:
        executor: a :class:`~concepts.dm.pdsketch.executor.PDSketchExecutor` object.
        state: the current state.
        continuous_values: a dictionary mapping the typename of continuous types to a list of possible values.
        regression_rule_names: a list of regression rule names to generate. If None, all regression rules will be generated.
        regression_rule_filter: a function that takes an :class:`RegressionRuleApplier` object and returns a boolean value indicating whether the action should be included in the result.
        filter_static: whether to use the :func:`filter_static_actions` function to filter out static regression rules. The function will check all static predicates in the precondition list.
    """
    if regression_rule_names is not None:
        lifted_regression_rules = [executor.domain.regression_rules[x] for x in regression_rule_names]
    else:
        lifted_regression_rules = list(executor.domain.regression_rules.values())

    regression_rules = list()
    for op in lifted_regression_rules:
        argument_candidates = list()
        for arg in op.arguments:
            if isinstance(arg.dtype, ObjectType):
                argument_candidates.append(state.object_type2name[arg.dtype.typename])
            else:
                assert isinstance(arg.dtype, NamedTensorValueType)
                argument_candidates.append(continuous_values[arg.dtype.typename])
        for comb in itertools.product(*argument_candidates):
            regression_rules.append(op(*comb))
    if filter_static:
        regression_rules = filter_static_grounding(executor, state, regression_rules)
    if regression_rule_filter is not None:
        regression_rules = list(filter(regression_rule_filter, regression_rules))
    return regression_rules


def _is_cacheable_fluent(expr: ValueOutputExpression):
    if isinstance(expr, FunctionApplicationExpression):
        return expr.function.ftype.is_cacheable
    return False

