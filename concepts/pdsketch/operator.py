#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : operator.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/04/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import itertools
from typing import TYPE_CHECKING, Any, Optional, Union, Iterable, Iterator, Sequence, Tuple, List, Mapping, Callable

from jacinle.logging import get_logger
from jacinle.utils.enum import JacEnum
from jacinle.utils.printing import indent_text
from jacinle.utils.meta import repr_from_str

from concepts.dsl.executors.tensor_value_executor import compose_bvdict_args
from concepts.dsl.dsl_types import ObjectType, NamedTensorValueType, PyObjValueType, Variable, UnnamedPlaceholder
from concepts.dsl.expression import (
    AndExpression, AssignExpression, ConditionalAssignExpression, DeicticAssignExpression,
    ValueOutputExpression, VariableExpression, VariableAssignmentExpression,
    find_free_variables
)
from concepts.dsl.tensor_value import TensorValue
from concepts.pdsketch.predicate import is_simple_bool, get_simple_bool_predicate

if TYPE_CHECKING:
    from concepts.pdsketch.executor import PDSketchExecutor
    from concepts.pdsketch.domain import State

logger = get_logger(__file__)

__all__ = [
    'Precondition', 'Effect', 'Operator', 'MacroOperator', 'OperatorApplier', 'OperatorApplicationExpression',
    'SubgoalSerializability', 'SubgoalContSerializability', 'AchieveExpression',
    'RegressionRule', 'RegressionRuleApplier',
    'gen_all_grounded_actions', 'filter_static_grounding', 'gen_all_partially_grounded_actions',
    'gen_all_grounded_regression_rules'
]


class Precondition(object):
    """The precondition of an operator. It is basically a wrapper around :class:`~concepts.dsl.expression.ValueOutputExpression`."""

    def __init__(self, bool_expr: ValueOutputExpression):
        self.bool_expr = bool_expr
        self.ao_discretization = None

    bool_expr: ValueOutputExpression
    """The underlying Boolean expression."""

    ao_discretization: Optional[Any]
    """The And-Or discretization of the precondition."""

    def __str__(self) -> str:
        return str(self.bool_expr)

    __repr__ = repr_from_str


class Effect(object):
    """The effect of an operator. It is basically a wrapper around :class:`~concepts.dsl.expression.VariableAssignmentExpression`."""

    def __init__(self, assign_expr: VariableAssignmentExpression):
        self.assign_expr = assign_expr
        self.update_from_simulation = False
        self.ao_discretization = None

    assign_expr: VariableAssignmentExpression
    """The underlying assign expression."""

    update_from_simulation: bool
    """Whether the effect should be updated from simulation, instead of the evaluation of the expression."""

    ao_discretization: Optional[Any]
    """The And-Or discretization of the effect."""

    def set_update_from_simulation(self, update_from_simulation: bool = True):
        self.update_from_simulation = update_from_simulation

    @property
    def unwrapped_assign_expr(self) -> Union[AssignExpression, ConditionalAssignExpression]:
        """Unwrap the DeicticAssignExpression and return the innermost AssignExpression."""
        expr = self.assign_expr
        if isinstance(expr, DeicticAssignExpression):
            expr = expr.expression
        assert isinstance(expr, (AssignExpression, ConditionalAssignExpression))
        return expr

    def __str__(self) -> str:
        return str(self.assign_expr)

    __repr__ = repr_from_str


class Controller(object):
    def __init__(self, name: str, arguments: Sequence[ValueOutputExpression]):
        self.name = name
        self.arguments = tuple(arguments)

    name: str
    """The identifier of the controller."""

    arguments: Tuple[ValueOutputExpression, ...]
    """The argument expressions to the controller function."""

    def __str__(self) -> str:
        return '{}({})'.format(self.name, ', '.join(map(str, self.arguments)))

    __repr__ = repr_from_str

    def pddl_str(self) -> str:
        return f'({self.name} {" ".join(map(str, self.arguments))})'


class OperatorBase(object):
    @property
    def is_primitive(self) -> bool:
        """Whether this operator is a primitive operator (instead of a macro operator or a sub operator in a macro)."""
        raise True

    @property
    def is_macro(self):
        """Whether the operator is a macro operator."""
        return False

    @property
    def is_axiom(self) -> bool:
        """Whether the operator is an axiom."""
        return False


class Operator(OperatorBase):
    """The operator definition in a planning domain."""

    def __init__(
        self,
        name: str,
        arguments: Sequence[Variable],
        preconditions: Sequence[Precondition],
        effects: Sequence[Effect],
        controller: Optional[Controller] = None,
        is_primitive: bool = True,
        is_axiom: bool = False,
        is_template: bool = False,
        extends: Optional[str] = None,
    ):
        self.name = name
        self.arguments = tuple(arguments)
        self.preconditions = tuple(preconditions)
        self.effects = tuple(effects)
        self.controller = controller
        self.is_template = is_template
        self.extends = extends

        self._is_primitive = is_primitive
        self._is_axiom = is_axiom

    name: str
    """The name of the operator."""

    arguments: Tuple[Variable]
    """The list of arguments of the operator."""

    preconditions: Tuple[Precondition]
    """The list of preconditions of the operator."""

    effects: Tuple[Effect]
    """The list of effects of the operator."""

    controller: Optional[Controller]
    """The controller function of the operator."""

    is_axiom: bool
    """Whether this operator is an axiom."""
    is_template: bool
    """Whether this operator is a template."""

    extends: Optional[str]
    """The name of the operator that this operator extends."""

    @property
    def is_primitive(self) -> bool:
        return self._is_primitive

    @property
    def is_axiom(self) -> bool:
        return self._is_axiom

    @property
    def nr_arguments(self) -> int:
        """The number of arguments of the operator."""
        return len(self.arguments)

    def rename(self, new_name: str, is_primitive: Optional[bool] = None) -> 'Operator':
        """Rename the operator."""

        if is_primitive is None:
            is_primitive = self.is_primitive

        return Operator(
            new_name,
            self.arguments,
            self.preconditions,
            self.effects,
            controller=self.controller,
            is_primitive=is_primitive,
            is_axiom=self.is_axiom,
            is_template=self.is_template,
            extends=self.extends,
        )

    def __call__(self, *args) -> 'OperatorApplier':
        """Ground the operator with a list of arguments."""
        output_args = list()
        if args[-1] is Ellipsis:
            args = args[:-1] + ('??', ) * (self.nr_arguments - len(args) + 1)
        for i, arg in enumerate(args):
            if isinstance(arg, str) and arg == '??':
                output_args.append(UnnamedPlaceholder(self.arguments[i].dtype))
            else:
                output_args.append(arg)
        return OperatorApplier(self, tuple(output_args))

    def __str__(self) -> str:
        arg_string = ', '.join([str(arg) for arg in self.arguments])
        return f'{self.name}({arg_string})'

    __repr__ = repr_from_str

    def pddl_str(self) -> str:
        """Return the PDDL representation of the operator."""

        if not self.is_axiom:
            def_name, def_name_a, def_name_p, def_name_e = f'action {self.name}', 'parameters', 'precondition', 'effect'
        else:
            def_name, def_name_a, def_name_p, def_name_e = 'axiom', 'vars', 'context', 'implies'
        arg_string = ' '.join([str(arg) for arg in self.arguments])
        pre_string = '\n'.join([indent_text(str(pre), 1, tabsize=3) for pre in self.preconditions])
        eff_string = '\n'.join([indent_text(str(eff), 1, tabsize=3) for eff in self.effects])
        controller_string = ''
        if self.controller is not None:
            controller_string = f'\n  :controller {self.controller.pddl_str()}'
        return f'''(:{def_name}
 :{def_name_a} ({arg_string})
 :{def_name_p} (and
   {pre_string.lstrip()}
 )
 :{def_name_e} (and
   {eff_string.lstrip()}
 ){controller_string}
)'''


class MacroOperator(OperatorBase):
    def __init__(
        self,
        name: str,
        arguments: Sequence[Variable],
        sub_operators: Sequence['OperatorApplier']
    ):
        self.name = name
        self.arguments = tuple(arguments)
        self.sub_operators = tuple(sub_operators)
        self._check_sub_operator_arguments()

    name: str
    """The name of the macro operator."""

    arguments: Tuple[Variable]
    """The list of arguments of the macro operator."""

    sub_operators: Tuple['OperatorApplier']
    """The list of sub operators of the macro operator."""

    def _check_sub_operator_arguments(self):
        for sub_operator in self.sub_operators:
            for argument in sub_operator.arguments:
                if not isinstance(argument, Variable):
                    raise ValueError('The arguments of a sub operator must be variables.')

    @property
    def nr_arguments(self) -> int:
        """The number of arguments of the operator."""
        return len(self.arguments)

    @property
    def is_macro(self) -> bool:
        """Whether this operator is a macro operator."""
        return True

    @property
    def is_primitive(self) -> bool:
        return False

    def __call__(self, *args) -> 'OperatorApplier':
        """Ground the operator with a list of arguments."""
        output_args = list()
        if args[-1] is Ellipsis:
            args = args[:-1] + ('??', ) * (self.nr_arguments - len(args) + 1)
        for i, arg in enumerate(args):
            if isinstance(arg, str) and arg == '??':
                output_args.append(UnnamedPlaceholder(self.arguments[i].dtype))
            else:
                output_args.append(arg)
        return OperatorApplier(self, tuple(output_args))

    def __str__(self) -> str:
        def_name = 'macro'
        arg_string = ', '.join([str(arg) for arg in self.arguments])
        return f'{def_name}::{self.name}({arg_string})'

    __repr__ = repr_from_str

    def pddl_str(self) -> str:
        """Return the PDDL representation of the operator."""

        arg_string = ' '.join([str(arg) for arg in self.arguments])
        action_string = '\n'.join([indent_text(str(action), 1, tabsize=3) for action in self.sub_operators])
        return f'''(:macro
 ({self.name} {arg_string})
 (and
   {action_string.lstrip()}
 )
)'''


class OperatorApplier(object):
    """An operator applier is essentially a grounded operator, composed of an operator and its arguments."""

    def __init__(self, operator: Union[Operator, MacroOperator], arguments: Sequence[Union[str, Variable, UnnamedPlaceholder, TensorValue]]):
        """Initialize an operator applier."""
        self.operator = operator
        self.arguments = tuple(arguments)

        if len(self.arguments) != len(self.operator.arguments):
            raise ValueError(f'The number of arguments does not match the operator: {self.operator}, arguments: {self.arguments}.')

    operator: Union[Operator, MacroOperator]
    """The operator."""

    arguments: Tuple[Union[str, Variable, UnnamedPlaceholder, TensorValue], ...]
    """The arguments of the grounded operator."""

    @property
    def is_macro(self) -> bool:
        """Whether this operator is a macro operator."""
        return isinstance(self.operator, MacroOperator)

    def iter_sub_operator_appliers(self) -> Iterator['OperatorApplier']:
        if not self.is_macro:
            yield self
            return

        operator: MacroOperator = self.operator
        for sub_operator in operator.sub_operators:
            new_sub_operator = sub_operator.replace_arguments({arg_def.name: arg for arg_def, arg in zip(operator.arguments, self.arguments)})
            if new_sub_operator.is_macro:
                yield from new_sub_operator.iter_sub_operator_appliers()
            else:
                yield new_sub_operator

    def replace_arguments(self, argument_map: Mapping[str, Union[Variable, UnnamedPlaceholder, TensorValue]]) -> 'OperatorApplier':
        """Replace the arguments of the operator applier with a map from argument names to new arguments."""
        new_arguments = list()
        for arg in self.arguments:
            if isinstance(arg, Variable) and arg.name in argument_map:
                new_arguments.append(argument_map[arg.name])
            else:
                new_arguments.append(arg)
        return OperatorApplier(self.operator, new_arguments)

    @property
    def name(self) -> str:
        """The name of the operator."""
        return self.operator.name

    def __str__(self) -> str:
        if not self.operator.is_axiom:
            def_name = 'action'
        else:
            def_name = 'axiom'

        def arg_str(arg_def, arg):
            if isinstance(arg, TensorValue):
                return f'{arg_def.name}={arg.short_str()}'
            elif isinstance(arg, UnnamedPlaceholder):
                return f'{arg_def.name}=??'
            else:
                return f'{arg_def.name}={arg}'

        arg_string = ', '.join([
            arg_str(arg_def, arg)
            for arg_def, arg in zip(self.operator.arguments, self.arguments)
        ])
        return f'{def_name}::{self.operator.name}({arg_string})'

    __repr__ = repr_from_str

    def pddl_str(self) -> str:
        arg_str = ' '.join([str(arg) for arg in self.arguments])
        return f'({self.operator.name} {arg_str})'


class OperatorApplicationExpression(object):
    """An abstract operator grounding. For example :code:`(move ?x ?y)` where :code:`?x` and :code:`?y` are variables in the context."""

    def __init__(self, operator: Operator, arguments: Sequence[Union[VariableExpression, UnnamedPlaceholder, ValueOutputExpression]]):
        self.operator = operator
        self.arguments = tuple(arguments)

    operator: Operator
    """The operator that is applied."""

    arguments: Tuple[Union[VariableExpression, UnnamedPlaceholder, ValueOutputExpression], ...]
    """The arguments of the operator."""

    def ground(self, executor: 'PDSketchExecutor') -> 'OperatorApplier':
        """Ground the operator statement."""
        return OperatorApplier(self.operator, tuple(executor.execute(arg) for arg in self.arguments))

    @property
    def name(self) -> str:
        """The name of the operator."""
        return self.operator.name

    def __str__(self) -> str:
        def_name = 'action'
        arg_string = ', '.join([
            arg_def.name + '=' + str(arg)
            for arg_def, arg in zip(self.operator.arguments, self.arguments)
        ])
        return f'{def_name}::{self.operator.name}({arg_string})'

    __repr__ = repr_from_str

    def pddl_str(self) -> str:
        arg_str = ' '.join([str(arg) for arg in self.arguments])
        return f'({self.operator.name} {arg_str})'


class SubgoalSerializability(JacEnum):
    STRONG = 'strong'
    RULE = 'rule'
    ORDER = 'order'


class SubgoalContSerializability(JacEnum):
    FORALL = 'forall'
    SOME = 'some'


class AchieveExpression(object):
    """An achieve expression. This is used in the definition of regression rules. Each achieve expression has two properties: serializability and cont_serializability.

        - serializability: 'strong', 'rule', or 'order'. Strong serializability means that the achieve expression is serializable for any state that satisfies the preconditions
            of the achieve expression, i.e., all achieve expressions prior to the current achieve expression. Rule serializability means that the achieve expression is serializable
            only for some refinement of the achieve expressions prior to the current achieve expression. However, there exists some refinement for previous expressions such that
            this new achieve expression can be planned based on the results of previous subgoals. Order serializability means that this subgoal is only serializable with respect
            to the order of the subgoals. Therefore, when plan for the sequence of subgoals, the order of the subgoals are preserved, but the preconditions for the subsequent
            rules might need to be promoted to a previous point in the plan.
        - cont_serializability: 'some' or 'forall'. Some continuous serializability means that the achieve expression is serializable for some continuous values of the action parameters.
            Forall continuous serializability means that the achieve expression is serializable for all continuous values of the action parameters.
    """

    def __init__(self, goal: ValueOutputExpression, maintains: Sequence[ValueOutputExpression], serializability: str = 'strong', cont_serializability: str = 'some'):
        self.goal = goal
        self.maintains = tuple(maintains)
        self.serializability = SubgoalSerializability.from_string(serializability)
        self.cont_serializability = SubgoalContSerializability.from_string(cont_serializability)

    goal: ValueOutputExpression
    """The goal of the achieve expression."""

    maintains: Tuple[ValueOutputExpression]
    """The list of maintain expressions."""

    serializability: SubgoalSerializability
    """The serializability of the achieve expression: 'strong', 'rule', or 'order'."""

    cont_serializability: SubgoalContSerializability
    """The continuous serializability of the achieve expression: 'some' or 'forall'."""

    def __str__(self):
        return f'achieve({self.goal}, serializability={self.serializability}, maintain={{{" ".join([str(m) for m in self.maintains])}}})'

    __repr__ = repr_from_str


class RegressionRule(object):
    def __init__(
        self,
        name: str,
        parameters: Sequence[Variable],
        preconditions: Sequence[Precondition],
        preconstraints: Sequence[Precondition],
        goal: Sequence[Effect],
        goal_expression: ValueOutputExpression,
        body: Sequence[Union[OperatorApplicationExpression, AchieveExpression]],
        always: bool = False
    ):
        """Initialize a regression rule."""

        self.name = name
        self.arguments = tuple(parameters)
        self.preconditions = tuple(preconditions)
        self.preconditions_conjunction = AndExpression(*[p.bool_expr for p in self.preconditions])
        self.preconstraints = tuple(preconstraints)
        self.goal = tuple(goal)
        self.goal_expression = goal_expression
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

    preconstraints: Tuple[Precondition, ...]
    """The preconstraints of the regression rule (the constraints in the maintain set)."""

    goal: Tuple[Effect, ...]
    """The goal of the regression rule."""

    goal_expression: ValueOutputExpression
    """The goal expression of the regression rule."""

    body: Tuple[Union[OperatorApplicationExpression, AchieveExpression], ...]
    """The body of the regression rule, including operator applications and achieve expressions."""

    always: bool
    """Whether the regression rule is always applicable."""

    max_reorder_prefix_length: int
    """The maximum length of the prefix of the body that might need to be reordered."""

    max_rule_prefix_length: int
    """The maximum length of the prefix of the body that need tracking of all possible refinements."""

    @property
    def nr_arguments(self) -> int:
        """The number of arguments of the operator."""
        return len(self.arguments)

    @property
    def serializability(self):
        return 'always' if self.always else 'sometimes'

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
        return f'''(:regression {self.name} [always={self.always}]
 :parameters ({' '.join(str(param) for param in self.arguments)})
 :preconditions (and {' '.join(str(precondition) for precondition in self.preconditions)})
 :preconstraints (and {' '.join(str(preconstraint) for preconstraint in self.preconstraints)})
 :goal (and {' '.join(str(effect) for effect in self.goal)})
 :rule (then
   {body_str.lstrip()}
 )
)'''

    def __call__(self, *args) -> 'RegressionRuleApplier':
        """Ground the operator with a list of arguments."""
        output_args = list()
        if args[-1] is Ellipsis:
            args = args[:-1] + ('??', ) * (self.nr_arguments - len(args) + 1)
        for i, arg in enumerate(args):
            if isinstance(arg, str) and arg == '??':
                output_args.append(UnnamedPlaceholder(self.arguments[i].dtype))
            else:
                output_args.append(arg)
        return RegressionRuleApplier(self, tuple(output_args))


class RegressionRuleApplier(object):
    def __init__(self, regression_rule: RegressionRule, arguments: Sequence[Union[Variable, TensorValue, Any]]):
        self.regression_rule = regression_rule
        self.arguments = tuple(arguments)

    regression_rule: RegressionRule
    """The regression rule that is applied."""

    arguments: Tuple[Union[Variable, TensorValue, Any], ...]
    """The arguments of the regression rule."""

    def __str__(self) -> str:
        return f'{self.regression_rule.name}({", ".join(str(arg) for arg in self.arguments)})'

    __repr__ = repr_from_str

    def pddl_str(self) -> str:
        arg_str = ' '.join([str(arg) for arg in self.arguments])
        return f'({self.regression_rule.name} {arg_str})'


def gen_all_grounded_actions(
    executor: 'PDSketchExecutor', state: 'State', continuous_values: Optional[Mapping[str, Iterable[TensorValue]]] = None,
    action_names: Optional[Sequence[str]] = None, action_filter: Optional[Callable[[OperatorApplier], bool]] = None, filter_static: bool = True,
    allow_macro_operator: bool = False
) -> List[OperatorApplier]:
    """Generate all grounded actions in a state. Note that this function does not check if the action is applicable at the current state.

    Args:
        executor: a :class:`~concepts.pdsketch.executor.PDSketchExecutor` object.
        state: the current state.
        continuous_values: a dictionary mapping the typename of continuous types to a list of possible values.
        action_names: a list of action names to generate. If None, all actions will be generated.
        action_filter: a function that takes an :class:`OperatorApplier` object and returns a boolean value indicating whether the action should be included in the result.
        filter_static: whether to use the :func:`filter_static_actions` function to filter out static actions. The function will check all static predicates in the
            domain and remove actions that will never been applicable.
        allow_macro_operator: whether to allow macro operators.
    """
    if action_names is not None:
        action_ops = [executor.domain.operators[x] for x in action_names]
    else:
        action_ops = list(executor.domain.operators.values())

    if not allow_macro_operator:
        action_ops = [x for x in action_ops if not x.is_macro]
    else:
        raise NotImplementedError('Macro operators are not supported yet.')

    actions = list()
    for op in action_ops:
        argument_candidates = list()
        for arg in op.arguments:
            if isinstance(arg.dtype, ObjectType):
                argument_candidates.append(state.object_type2name[arg.dtype.typename])
            else:
                assert isinstance(arg.dtype, NamedTensorValueType)
                argument_candidates.append(continuous_values[arg.dtype.typename])
        for comb in itertools.product(*argument_candidates):
            actions.append(op(*comb))
    if filter_static:
        actions = filter_static_grounding(executor, state, actions)
    if action_filter is not None:
        actions = list(filter(action_filter, actions))
    return actions


def filter_static_grounding(executor: 'PDSketchExecutor', state: 'State', actions: Sequence[Union[OperatorApplier, RegressionRuleApplier]]) -> List[Union[OperatorApplier, RegressionRuleApplier]]:
    """Filter out grounded actions or regression rules that do not satisfy static preconditions.

    Args:
        executor: a :class:`~concepts.pdsketch.executor.PDSketchExecutor` object.
        state: the current state.
        actions: a list of grounded actions to be filtered.

    Returns:
        a list of grounded actions that satisfy static preconditions.
    """

    output_actions = list()
    for action in actions:
        if isinstance(action, OperatorApplier):
            bounded_variables = compose_bvdict_args(action.operator.arguments, action.arguments, state=state)
            preconditions = action.operator.preconditions
        elif isinstance(action, RegressionRuleApplier):
            bounded_variables = compose_bvdict_args(action.regression_rule.arguments, action.arguments, state=state)
            preconditions = action.regression_rule.preconditions
        else:
            raise TypeError(f'filter_static_grounding only accepts OperatorApplier or RegressionRuleApplier sequences； got {type(action)} in the input list.')
        flag = True
        for pre in preconditions:
            if is_simple_bool(pre.bool_expr) and get_simple_bool_predicate(pre.bool_expr).is_static:
                rv = executor.execute(pre.bool_expr, state=state, bounded_variables=bounded_variables).item()
                if rv < 0.5:
                    flag = False
                    break
        if flag:
            output_actions.append(action)
    return output_actions


def gen_all_partially_grounded_actions(
    executor: 'PDSketchExecutor', state: 'State',
    action_names: Optional[Sequence[str]] = None, action_filter: Optional[Callable[[OperatorApplier], bool]] = None, filter_static: bool = True,
    allow_macro_operator: bool = False
) -> List[OperatorApplier]:
    """Generate all partially grounded actions in a state. Partially grounded actions are actions with only object-typed arguments grounded.
    For Value-typed arguments, the resulting :class:`~concepts.pdsketch.operators.OperatorApplier` object will have arguments that are
    placeholders (i.e., :class:`~concepts.dsl.dsl_types.UnnamedPlaceholder` objects).
    Note that, as :func:`generate_all_grounded_actions`, this function does not check if the action is applicable at the current state.

    Args:
        executor: a :class:`~concepts.pdsketch.executor.PDSketchExecutor` object.
        state: the current state.
        action_names: a list of action names to generate. If None, all actions will be generated.
        action_filter: a function that takes an :class:`OperatorApplier` object and returns a boolean value indicating whether the action should be included in the result.
        filter_static: whether to use the :func:`filter_static_actions` function to filter out static actions. The function will check all static predicates in the
            domain and remove actions that will never been applicable.
        allow_macro_operator: whether to allow macro operators.

    Returns:
        a list of partially grounded actions.
    """

    if action_names is not None:
        action_ops = [executor.domain.operators[x] for x in action_names]
    else:
        action_ops = list(executor.domain.operators.values())

    if not allow_macro_operator:
        action_ops = [x for x in action_ops if not x.is_macro]
    else:
        raise NotImplementedError('Macro operators are not supported yet.')

    actions = list()
    for op in action_ops:
        argument_candidates = list()
        for arg in op.arguments:
            if isinstance(arg.dtype, ObjectType):
                argument_candidates.append(state.object_type2name[arg.dtype.typename])
            else:
                assert isinstance(arg.dtype, (NamedTensorValueType, PyObjValueType))
                argument_candidates.append([UnnamedPlaceholder(arg.dtype)])
        for comb in itertools.product(*argument_candidates):
            actions.append(op(*comb))

    if filter_static:
        actions = filter_static_grounding(executor, state, actions)
    if action_filter is not None:
        actions = list(filter(action_filter, actions))
    return actions


def gen_all_grounded_regression_rules(
    executor: 'PDSketchExecutor', state: 'State', continuous_values: Optional[Mapping[str, Iterable[TensorValue]]] = None,
    regression_rule_names: Optional[Sequence[str]] = None, regression_rule_filter: Optional[Callable[[RegressionRuleApplier], bool]] = None, filter_static: bool = True,
) -> List[RegressionRuleApplier]:
    """Generate all grounded regression rules applicable in an environment, given the initial state.
    Note that this function does not check if the action is applicable at the current state.

    Args:
        executor: a :class:`~concepts.pdsketch.executor.PDSketchExecutor` object.
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

