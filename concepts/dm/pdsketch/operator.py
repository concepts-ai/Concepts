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
from jacinle.utils.printing import indent_text
from jacinle.utils.meta import repr_from_str

from concepts.dsl.executors.tensor_value_executor import compose_bvdict_args
from concepts.dsl.dsl_types import ObjectType, ValueType, NamedTensorValueType, PyObjValueType, Variable, UnnamedPlaceholder
from concepts.dsl.expression import (
    AssignExpression, ConditionalAssignExpression, DeicticAssignExpression,
    ValueOutputExpression, VariableExpression, VariableAssignmentExpression,
)
from concepts.dsl.tensor_value import TensorValue
from concepts.dm.pdsketch.predicate import FunctionEvaluationDefinitionMode, is_simple_bool, get_simple_bool_predicate

if TYPE_CHECKING:
    from concepts.dm.pdsketch.regression_rule import RegressionRuleApplier

logger = get_logger(__file__)

__all__ = [
    'Precondition', 'Effect', 'Implementation',
    'Operator', 'MacroOperator', 'OperatorApplier', 'OperatorApplicationExpression',
    'gen_all_grounded_actions', 'filter_static_grounding', 'gen_all_partially_grounded_actions',
]


class Precondition(object):
    """The precondition of an operator. It is basically a wrapper around :class:`~concepts.dsl.expression.ValueOutputExpression`."""

    def __init__(self, bool_expr: ValueOutputExpression, simulation: bool = False, execution: bool = False):
        self.bool_expr = bool_expr
        self.mode = FunctionEvaluationDefinitionMode.from_bools(simulation, execution)
        self.ao_discretization = None

    bool_expr: ValueOutputExpression
    """The underlying Boolean expression."""

    mode: FunctionEvaluationDefinitionMode
    """The mode of the precondition. There are three possible modes: FUNCTIONAL (the description is a pure function),
    SIMULATION (the functions have to rely on an update-to-date simulator state for evaluation),
    and EXECUTION (the functions have to rely on the exact world state after execution to be evaluated)."""

    ao_discretization: Optional[Any]
    """The And-Or discretization of the precondition."""

    def __str__(self) -> str:
        if self.mode is FunctionEvaluationDefinitionMode.FUNCTIONAL:
            return self.bool_expr.cached_string(-1)
        return self.mode.get_prefix() + ' ' + self.bool_expr.cached_string(-1)

    __repr__ = repr_from_str


class Effect(object):
    """The effect of an operator. It is basically a wrapper around :class:`~concepts.dsl.expression.VariableAssignmentExpression`."""

    def __init__(self, assign_expr: VariableAssignmentExpression, simulation: bool = False, execution: bool = False):
        self.assign_expr = assign_expr
        self.mode = FunctionEvaluationDefinitionMode.from_bools(simulation, execution)
        self.ao_discretization = None

    assign_expr: VariableAssignmentExpression
    """The underlying assign expression."""

    mode: FunctionEvaluationDefinitionMode
    """The mode of the effect. There are three possible modes: FUNCTIONAL (the description is a pure function),
    SIMULATION (the actual effect computation will be computed by the simulator),
    and EXECUTION (the actual effect can only be observed by actually executing the operator in the environment)."""

    ao_discretization: Optional[Any]
    """The And-Or discretization of the effect."""

    @property
    def update_from_simulation(self) -> bool:
        """Whether the effect should be updated from simulation, instead of the evaluation of the expression."""
        return self.mode is FunctionEvaluationDefinitionMode.SIMULATION

    @property
    def update_from_execution(self) -> bool:
        return self.mode is FunctionEvaluationDefinitionMode.EXECUTION

    def set_update_from_simulation(self, update_from_simulation: bool = True):
        if update_from_simulation:
            self.mode = FunctionEvaluationDefinitionMode.SIMULATION
        else:
            self.mode = FunctionEvaluationDefinitionMode.FUNCTIONAL

    @property
    def unwrapped_assign_expr(self) -> Union[AssignExpression, ConditionalAssignExpression]:
        """Unwrap the DeicticAssignExpression and return the innermost AssignExpression."""
        expr = self.assign_expr
        if isinstance(expr, DeicticAssignExpression):
            expr = expr.expression
        assert isinstance(expr, (AssignExpression, ConditionalAssignExpression))
        return expr

    def __str__(self) -> str:
        if self.mode is FunctionEvaluationDefinitionMode.FUNCTIONAL:
            return self.assign_expr.cached_string(-1)
        return self.mode.get_prefix() + ' ' + self.assign_expr.cached_string(-1)

    __repr__ = repr_from_str


class Implementation(object):
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
        return f'{self.name} {" ".join(map(str, self.arguments))}'


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
        controller: Optional[Implementation] = None,
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

    controller: Optional[Implementation]
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

    @property
    def argument_names(self) -> Tuple[str, ...]:
        """The names of the arguments of the operator."""
        return tuple(arg.name for arg in self.arguments)

    @property
    def argument_types(self) -> Tuple[Union[ObjectType, ValueType], ...]:
        """The types of the arguments of the operator."""
        return tuple(arg.dtype for arg in self.arguments)

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
            extends=self.name,
        )

    def __call__(self, *args) -> 'OperatorApplier':
        """Ground the operator with a list of arguments."""
        output_args = list()
        if len(args) > 0 and args[-1] is Ellipsis:
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
        arg_string = ', '.join([str(arg) for arg in self.arguments])
        pre_string = '\n'.join([indent_text(str(pre), 1, tabsize=3) for pre in self.preconditions])
        eff_string = '\n'.join([indent_text(str(eff), 1, tabsize=3) for eff in self.effects])
        # pre_string = ' '.join([str(pre) for pre in self.preconditions])
        # eff_string = ' '.join([str(eff) for eff in self.effects])
        controller_string = ''
        if self.controller is not None:
            controller_string = f'\n :controller {self.controller.pddl_str()}'
        return f'''(:{def_name}
 :{def_name_a} ({arg_string})
 :{def_name_p} (and\n   {pre_string.lstrip()}\n )
 :{def_name_e} (and\n   {eff_string.lstrip()}\n ){controller_string}
)'''


class MacroOperator(OperatorBase):
    def __init__(
        self,
        name: str,
        arguments: Sequence[Variable],
        sub_operators: Sequence['OperatorApplier'],
        preconditions: Sequence[Precondition] = tuple(),
        effects: Sequence[Effect] = tuple()
    ):
        self.name = name
        self.arguments = tuple(arguments)
        self.preconditions = tuple(preconditions)
        self.effects = tuple(effects)
        self.sub_operators = tuple(sub_operators)
        self._check_sub_operator_arguments()

    name: str
    """The name of the macro operator."""

    arguments: Tuple[Variable, ...]
    """The list of arguments of the macro operator."""

    sub_operators: Tuple['OperatorApplier', ...]
    """The list of sub operators of the macro operator."""

    preconditions: Tuple[Precondition, ...]
    """The list of preconditions of the macro operator (only used for extended macros)."""

    effects: Tuple[Effect, ...]
    """The list of effects of the macro operator (only used for extended macros)."""

    def _check_sub_operator_arguments(self):
        pass
        # for sub_operator in self.sub_operators:
        #     for argument in sub_operator.arguments:
        #         if not isinstance(argument, Variable):
        #             raise ValueError('The arguments of a sub operator must be variables.')

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
        """Whether this operator is a primitive operator."""
        return False

    @property
    def is_extended_macro(self) -> bool:
        """Whether this macro operator is an extended macro."""
        return len(self.preconditions) > 0 or len(self.effects) > 0

    def __call__(self, *args) -> 'OperatorApplier':
        """Ground the operator with a list of arguments."""
        output_args = list()
        if len(args) > 0 and args[-1] is Ellipsis:
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

        if self.is_extended_macro:
            def_name = 'extended-macro'
            arg_string = ', '.join([str(arg) for arg in self.arguments])
            pre_string = '\n'.join([indent_text(str(pre), 1, tabsize=3) for pre in self.preconditions])
            eff_string = '\n'.join([indent_text(str(eff), 1, tabsize=3) for eff in self.effects])
            # pre_string = ' '.join([str(pre) for pre in self.preconditions])
            # eff_string = ' '.join([str(eff) for eff in self.effects])
            sub_string = '\n'.join([indent_text(str(sub), 1, tabsize=3) for sub in self.sub_operators])
            return f'''(:{def_name} {self.name}
 :parameters ({arg_string})
 :precondition (and\n  {pre_string.lstrip()}\n )
 :effect (and\n  {eff_string.lstrip()}\n )
 :body (then
   {sub_string.lstrip()}
 )
)'''
        else:
            arg_string = ' '.join([str(arg) for arg in self.arguments])
            action_string = '\n'.join([indent_text(str(action), 1, tabsize=2) for action in self.sub_operators])
            return f'''(:macro ({self.name} {arg_string}) (then
  {action_string.lstrip()}
))'''


class OperatorApplier(object):
    """An operator applier is essentially a grounded operator, composed of an operator and its arguments."""

    def __init__(self, operator: Union[Operator, MacroOperator], arguments: Sequence[Union[str, Variable, UnnamedPlaceholder, TensorValue]], regression_rule: Optional['RegressionRuleApplier'] = None):
        """Initialize an operator applier."""
        self.operator = operator
        self.arguments = tuple(arguments)
        self.regression_rule = regression_rule

        if len(self.arguments) != len(self.operator.arguments):
            raise ValueError(f'The number of arguments does not match the operator: {self.operator}, arguments: {self.arguments}.')

    operator: Union[Operator, MacroOperator]
    """The operator."""

    arguments: Tuple[Union[str, Variable, UnnamedPlaceholder, TensorValue], ...]
    """The arguments of the grounded operator."""

    regression_rule: Optional['RegressionRuleApplier']
    """The regression rule that generates this operator applier. It is only used in the context of goal regression search."""

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
        if self.regression_rule is not None:
            if arg_string == '':
                arg_string = f'!rule={self.regression_rule}'
            else:
                arg_string = f'{arg_string}, !rule={self.regression_rule}'
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


def gen_all_grounded_actions(
    executor: 'PDSketchExecutor', state: 'State', continuous_values: Optional[Mapping[str, Iterable[TensorValue]]] = None,
    action_names: Optional[Sequence[str]] = None, action_filter: Optional[Callable[[OperatorApplier], bool]] = None, filter_static: bool = True,
    use_only_macro_operators: bool = False, allow_macro_operators: bool = False, allow_nonprimitive_operators: bool = False,
) -> List[OperatorApplier]:
    """Generate all grounded actions in a state. Note that this function does not check if the action is applicable at the current state.

    Args:
        executor: a :class:`~concepts.dm.pdsketch.executor.PDSketchExecutor` object.
        state: the current state.
        continuous_values: a dictionary mapping the typename of continuous types to a list of possible values.
        action_names: a list of action names to generate. If None, all actions will be generated.
        action_filter: a function that takes an :class:`OperatorApplier` object and returns a boolean value indicating whether the action should be included in the result.
        filter_static: whether to use the :func:`filter_static_actions` function to filter out static actions. The function will check all static predicates in the
            domain and remove actions that will never been applicable.
        use_only_macro_operators: whether to use only macro operators.
        allow_macro_operators: whether to allow macro operators.
        allow_nonprimitive_operators: whether to allow non-primitive operators. Those operators are suboperators of other macro operators.
    """
    if action_names is not None:
        action_ops = [executor.domain.operators[x] for x in action_names]
    else:
        action_ops = list(executor.domain.operators.values())

    if use_only_macro_operators:
        action_ops = [x for x in action_ops if x.is_macro]
    else:
        if not allow_macro_operators:
            action_ops = [x for x in action_ops if not x.is_macro]
        if not allow_nonprimitive_operators:
            action_ops = [x for x in action_ops if x.is_primitive or x.is_macro]

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


def filter_static_grounding(executor: 'PDSketchExecutor', state: 'State', actions: Sequence[Union[OperatorApplier, 'RegressionRuleApplier']]) -> List[Union[OperatorApplier, 'RegressionRuleApplier']]:
    """Filter out grounded actions or regression rules that do not satisfy static preconditions.

    Args:
        executor: a :class:`~concepts.dm.pdsketch.executor.PDSketchExecutor` object.
        state: the current state.
        actions: a list of grounded actions to be filtered.

    Returns:
        a list of grounded actions that satisfy static preconditions.
    """

    from concepts.dm.pdsketch.regression_rule import RegressionRuleApplier

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
    For Value-typed arguments, the resulting :class:`~concepts.dm.pdsketch.operators.OperatorApplier` object will have arguments that are
    placeholders (i.e., :class:`~concepts.dsl.dsl_types.UnnamedPlaceholder` objects).
    Note that, as :func:`generate_all_grounded_actions`, this function does not check if the action is applicable at the current state.

    Args:
        executor: a :class:`~concepts.dm.pdsketch.executor.PDSketchExecutor` object.
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

