#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : strips_grounding_onthefly.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/18/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Strips grounding and seawrch using on-the-fly grounding of objects.

TODO
----

- [ ] Rewrite this file using the AtomicStrips interface.
"""

import time
import itertools
import warnings
import functools
import heapq as hq
import jacinle  # noqa
from collections import deque
from typing import Optional, Union, Sequence, Tuple, List, Dict

from concepts.dsl.dsl_types import Variable, ObjectConstant
from concepts.dsl.expression import ObjectConstantExpression, AndExpression
from concepts.pdsketch.predicate import Predicate
from concepts.pdsketch.operator import Operator
from concepts.pdsketch.domain import Domain, Problem
from concepts.pdsketch.strips.strips_expression import SProposition, SState, SStateDict, SBoolPredicateApplicationExpression
from concepts.pdsketch.strips.strips_grounded_expression import GSBoolOutputExpression
from concepts.pdsketch.strips.atomic_strips_domain import AtomicStripsOperator


class GoalNotAConjunctionError(Exception):
    pass


class OnTheFlyGStripsProblem(object):
    def __init__(
        self,
        objects: Dict[str, Sequence[str]],
        object2index: Dict[str, int],
        initial_state: SStateDict,
        predicates: Dict[str, Predicate],
        operators: Dict[str, AtomicStripsOperator],
        constants: Dict[str, ObjectConstant],
        conjunctive_goal: Optional[Sequence[SProposition]] = None,
        complex_goal: Optional[GSBoolOutputExpression] = None,
        use_integer_constants: bool = False
    ):
        self.objects = {k: tuple(v) for k, v in objects.items()}
        self.object2index = object2index
        self.initial_state = initial_state
        self.predicates = predicates
        self.operators = operators
        self.constants = constants.copy()

        assert (conjunctive_goal is None) ^ (complex_goal is None), 'Only one of conjunctive_goal and complex_goal can be specified.'
        self.conjunctive_goal = frozenset(conjunctive_goal) if conjunctive_goal is not None else None
        self.complex_goal = complex_goal
        self.complex_goal_func = complex_goal.compile() if complex_goal is not None else None

        self.use_integer_constants = use_integer_constants

    def goal_test(self, state: SState):
        if self.conjunctive_goal is not None:
            return self.conjunctive_goal.issubset(state)
        else:
            return self.complex_goal_func(state)

    @property
    def has_complex_goal(self):
        return self.complex_goal is not None

    @classmethod
    def from_domain_and_problem(cls, domain: Domain, problem: Problem, use_integer_constants: bool = False) -> 'OnTheFlyGStripsProblem':
        operators = dict()
        for operator in domain.operators.values():
            if isinstance(operator, Operator):
                if operator.is_macro:
                    continue
                # TODO(Jiayuan Mao @ 2023/03/19): support macro operator here.
                strips_operator = AtomicStripsOperator.from_operator(operator)
                operators[operator.name] = strips_operator
        constants = domain.constants

        objects = dict()
        object2index = dict()

        for name, constant in constants.items():
            typename = constant.dtype.typename
            if name not in object2index:
                if typename not in objects:
                    objects[typename] = list()
                objects[typename].append(name)
                object2index[name] = len(objects[typename]) - 1

        for k, typename in problem.objects.items():
            if k not in object2index:
                if typename not in objects:
                    objects[typename] = list()
                objects[typename].append(k)
                object2index[k] = len(objects[typename]) - 1

        initial_state = SStateDict()
        for predicate in problem.predicates:
            name = predicate.function.name
            if use_integer_constants:
                args = [object2index[arg.constant.name] for arg in predicate.arguments]
            else:
                args = [arg.constant.name for arg in predicate.arguments]
            initial_state.add(name, args)

        conjunctive_goal = None
        complex_goal = None
        try:
            conjunctive_goal = list()
            if not isinstance(problem.goal, AndExpression):
                raise GoalNotAConjunctionError()
            for arg in problem.goal.arguments:
                predicate_name = arg.function.name
                args = list()
                for arg in arg.arguments:
                    if not isinstance(arg, ObjectConstantExpression):
                        raise GoalNotAConjunctionError()
                    arg = arg.constant.name
                    args.append(arg)
                conjunctive_goal.append(f'{predicate_name} {" ".join(args)}')
        except GoalNotAConjunctionError:
            from concepts.pdsketch.executor import PDSketchExecutor
            from concepts.pdsketch.strips.strips_grounding import GStripsTranslatorOptimistic
            executor = PDSketchExecutor(domain)
            tensor_state = problem.to_state(executor)
            translator = GStripsTranslatorOptimistic(executor)
            with executor.with_state(tensor_state):
                gstrips_goal, _ = translator.compose_bool_expression(problem.goal)
            conjunctive_goal = None
            complex_goal = gstrips_goal

        return cls(
            objects, object2index, initial_state,
            predicates=domain.functions.copy(),
            operators=operators,
            constants=constants,
            conjunctive_goal=conjunctive_goal,
            complex_goal=complex_goal,
            use_integer_constants=use_integer_constants,
        )

    def decode_plan(self, plan: Sequence[Tuple[AtomicStripsOperator, Dict[str, Union[int, str]]]]) -> List[Tuple[AtomicStripsOperator, Dict[str, str]]]:
        if not self.use_integer_constants:
            return plan

        decoded_plan = list()
        for op, bound_arguments in plan:
            decoded_bound_arguments = dict()
            for arg in op.arguments:
                assert arg.name in bound_arguments, f'Argument {arg.name} is not bound in operator {op.name}.'
                decoded_bound_arguments[arg.name] = self.objects[arg.dtype.typename][bound_arguments[arg.name]]
            decoded_plan.append((op, decoded_bound_arguments))
        return decoded_plan

    def __str__(self) -> str:
        return f"""OnTheFlyGStripsProblem(
  objects={self.objects},
  initial_state={self.initial_state},
  goal={self.conjunctive_goal if self.conjunctive_goal is not None else self.complex_goal},
  operators={self.operators},
  constants={self.constants}
)"""

    def __repr__(self) -> str:
        return self.__str__()


def ogstrips_expand_state_with_negation(problem: OnTheFlyGStripsProblem, state: SState) -> SState:
    warnings.warn('ogstrips_expand_state_with_negation is deprecated. You should avoid using it because it is slow.', DeprecationWarning)
    expanded_state = SStateDict()

    for predicate in problem.predicates.values():
        if problem.use_integer_constants:
            options = itertools.product(*[range(len(problem.objects[arg.typename])) for arg in predicate.arguments])
        else:
            options = itertools.product(*[problem.objects[arg.typename] for arg in predicate.arguments])
        for args in options:
            args = tuple(args)
            if state.contains(predicate.name, args):
                expanded_state.add(predicate.name, args)
            else:
                expanded_state.add(f'{predicate.name}_not', args)

    return expanded_state


def ogstrips_bind_arguments(predicate: SBoolPredicateApplicationExpression, bound_arguments: Dict[str, Union[int, str]], object2index: Dict[str, int]):
    return predicate.name, tuple(bound_arguments[arg.name] if isinstance(arg, Variable) else arg for arg in predicate.arguments)

    # TODO(Jiayuan Mao @ 2023/03/27): bring back.
    boudned_arguments = list()
    for arg in predicate.arguments:
        if isinstance(arg, Variable):
            boudned_arguments.append(bound_arguments[arg.name])
        elif isinstance(arg, str):
            boudned_arguments.append(object2index.get(arg, 100000))
        else:
            raise ValueError(f'Unexpected argument type: {type(arg)}')
    return predicate.name, tuple(boudned_arguments)


def ogstrips_generate_applicable_actions(problem: OnTheFlyGStripsProblem, state: SStateDict, check_negation: bool = False) -> List[AtomicStripsOperator]:
    TOO_MANY, FAILED, PASS = object(), object(), object()

    def compute_possible_grounding(predicate: SBoolPredicateApplicationExpression, bound_arguments: Dict[str, Union[int, str]]):
        unbound_arguments = [arg for arg in predicate.arguments if isinstance(arg, Variable) and arg.name not in bound_arguments]
        if len(unbound_arguments) == 0:
            name, arguments = ogstrips_bind_arguments(predicate, bound_arguments, problem.object2index)
            rv = state.contains(name, arguments, predicate.negated, check_negation=check_negation)
            if not rv:
                return '', FAILED
            return '', PASS
        elif len(unbound_arguments) == 1:
            arg = unbound_arguments[0]
            valid_arguments = list()

            if problem.use_integer_constants:
                options = range(len(problem.objects[arg.typename]))
            else:
                options = problem.objects[arg.typename]

            for o in options:
                bound_arguments[arg.name] = o
                name, arguments = ogstrips_bind_arguments(predicate, bound_arguments, problem.object2index)
                rv = state.contains(name, arguments, predicate.negated, check_negation=check_negation)
                if rv:
                    valid_arguments.append(o)
                del bound_arguments[arg.name]
            return arg.name, valid_arguments
        else:
            return '', TOO_MANY

    # @jacinle.log_function(verbose=False)
    def dfs(preconditions: Sequence[SBoolPredicateApplicationExpression], bound_arguments: Dict[str, int]):
        """Inner DFS function.

        Args:
            preconditions: the preconditions to be satisfied.
            bound_arguments: a mapping from variable name to object.
        """

        # jacinle.log_function.print('dfs', bound_arguments, 'remaining preconditions:', len(preconditions))
        # import ipdb; ipdb.set_trace()

        for i, precondition in enumerate(preconditions):
            name, valid_arguments = compute_possible_grounding(precondition, bound_arguments)
            if valid_arguments == FAILED:
                # jacinle.log_function.print('Failed.')
                return list()
            elif valid_arguments == PASS:
                # jacinle.log_function.print('Pass.')
                return dfs(preconditions[:i] + preconditions[i + 1:], bound_arguments)
            elif valid_arguments == TOO_MANY:
                pass
            else:
                outputs = list()
                for arg in valid_arguments:
                    bound_arguments[name] = arg
                    outputs.extend(dfs(preconditions[:i] + preconditions[i + 1:], bound_arguments))
                    del bound_arguments[name]
                return outputs

        unbound_arguments = [arg for arg in operator.arguments if isinstance(arg, Variable) and arg.name not in bound_arguments]
        # print('unbound_arguments', unbound_arguments, bound_arguments)
        if len(unbound_arguments) == 0:
            # jacinle.log_function.print('Found a grounding:', bound_arguments)
            return [bound_arguments.copy()]

        if problem.use_integer_constants:
            unbound_arguments_possible_values = {arg.name: range(len(problem.objects[arg.typename])) for arg in unbound_arguments}
        else:
            unbound_arguments_possible_values = {arg.name: problem.objects[arg.typename] for arg in unbound_arguments}

        name, valid_arguments = min(unbound_arguments_possible_values.items(), key=lambda x: len(x[1]))
        outputs = list()
        for arg in valid_arguments:
            bound_arguments[name] = arg
            # jacinle.log_function.print('{} = {}'.format(name, arg))
            outputs.extend(dfs(preconditions, bound_arguments))
        del bound_arguments[name]
        return outputs

    for operator in problem.operators.values():
        # jacinle.log_function.print(f'operator: {operator.name}')
        for bound_arguments in dfs(operator.preconditions, dict()):
            # jacinle.log_function.print('yield bound_arguments:', bound_arguments)
            yield operator, bound_arguments


def ogstrips_check_precondition(state: SStateDict, operator: AtomicStripsOperator, bound_arguments: Dict[str, Union[int, str]], object2index: Dict[str, int]):
    for precondition in operator.preconditions:
        name, arguments = ogstrips_bind_arguments(precondition, bound_arguments, object2index)
        if not state.contains(name, arguments, precondition.negated):
            return False
    return True


def ogstrips_apply_operator(state: SStateDict, operator: AtomicStripsOperator, bound_arguments: Dict[str, Union[int, str]], object2index: Dict[str, int]):
    new_state = state.clone()
    for predicate in operator.del_effects:
        name, arguments = ogstrips_bind_arguments(predicate, bound_arguments, object2index)
        new_state.remove(name, arguments)
    for predicate in operator.add_effects:
        name, arguments = ogstrips_bind_arguments(predicate, bound_arguments, object2index)
        new_state.add(name, arguments)
    return new_state


def ogstrips_search(problem: OnTheFlyGStripsProblem, initial_actions: Sequence[Tuple[AtomicStripsOperator, Dict[str, Union[int, str]]]] = tuple(), max_expanded_nodes: int = 1000000, timeout: float = 10.0):
    frontier = deque()

    initial_state = problem.initial_state
    for operator, arguments in initial_actions:
        ogstrips_check_precondition(initial_state, operator, arguments, problem.object2index)
        initial_state = ogstrips_apply_operator(initial_state, operator, arguments, problem.object2index)

    frontier.append((initial_state, list(initial_actions)))
    explored = set()

    start_time = time.time()
    nr_expanded_nodes = 0
    while len(frontier) > 0:
        nr_expanded_nodes += 1
        if nr_expanded_nodes > max_expanded_nodes:
            break

        if nr_expanded_nodes % 100 == 0:
            if time.time() - start_time > timeout:
                print('ogstrips_search::Timeout.')
                break

        state, plan = frontier.popleft()

        # action_strings = [f"{operator.name}({', '.join(bound_arguments.values())})" for operator, bound_arguments in plan]
        # print('State', state, 'Plan', action_strings)
        # print('Plan', action_strings)

        for operator, bound_arguments in ogstrips_generate_applicable_actions(problem, state):
            new_state = ogstrips_apply_operator(state, operator, bound_arguments, problem.object2index)
            new_state_set = new_state.as_state()
            if new_state_set not in explored:
                if problem.goal_test(new_state_set):
                    return plan + [(operator, bound_arguments)]

                frontier.append((new_state, plan + [(operator, bound_arguments)]))
                explored.add(new_state_set)

    return None


def ogstrips_search_with_heuristics(
    problem: OnTheFlyGStripsProblem,
    initial_actions: Sequence[Tuple[AtomicStripsOperator, Dict[str, Union[int, str]]]] = tuple(),
    hfunc_name: str = 'hmax', h_weight: float = 1, g_weight: float = 1,
    max_expanded_nodes: int = 1000000, timeout: float = 10.0,
    verbose: bool = False, hfunc_verbose: bool = False
):

    if hfunc_name == 'hmax':
        hfunc = ogstrips_hmax
    elif hfunc_name == 'hadd':
        hfunc = ogstrips_hadd
    elif hfunc_name == 'hff':
        hfunc = ogstrips_hff
    else:
        raise ValueError(f'Unknown heuristic function name: {hfunc_name}.')

    queue = list()

    initial_state = problem.initial_state
    for operator, arguments in initial_actions:
        ogstrips_check_precondition(initial_state, operator, arguments, problem.object2index)
        initial_state = ogstrips_apply_operator(initial_state, operator, arguments, problem.object2index)

    counter = 0
    # NB(Jiayuan Mao @ 2023/04/04): only verbose printing the hfunc computation for the initial state.
    h = hfunc(problem, initial_state, verbose=hfunc_verbose)
    queue.append((h * h_weight, counter, h, initial_state, list()))
    counter += 1
    explored = set()

    if verbose:
        print('Initial heuristic:', queue[0][2])
        import ipdb; ipdb.set_trace()

    start_time = time.time()
    nr_expanded_nodes = 0
    while len(queue) > 0:
        nr_expanded_nodes += 1
        if nr_expanded_nodes > max_expanded_nodes:
            break

        if nr_expanded_nodes % 10 == 0:
            if time.time() - start_time > timeout:
                print('ogstrips_search_with_heuristics::Timeout.')
                break

        prio, _, h, state, plan = hq.heappop(queue)
        if verbose:
            print('  h =', h, 'prio =', prio, 'plan =', [f"{operator.name}({', '.join(bound_arguments.values())})" for operator, bound_arguments in plan])

        # action_strings = [f"{operator.name}({', '.join(bound_arguments.values())})" for operator, bound_arguments in plan]
        # print('State', state, 'Plan', action_strings)
        # print('Plan', action_strings)

        for operator, bound_arguments in ogstrips_generate_applicable_actions(problem, state):
            new_state = ogstrips_apply_operator(state, operator, bound_arguments, problem.object2index)
            new_state_set = new_state.as_state()
            if new_state_set not in explored:
                new_plan = plan + [(operator, bound_arguments)]
                h = hfunc(problem, new_state)

                if verbose:
                    print('    add new plan, h =', h, f'new action={operator.name}({", ".join(bound_arguments.values())})')

                # if problem.goal_test(new_state_set):
                if h == 0:
                    return problem.decode_plan(new_plan)

                hq.heappush(queue, (h * h_weight + len(new_plan) * g_weight, counter, h, new_state, new_plan))  # GBF!
                counter += 1
                explored.add(new_state_set)


def ogstrips_verify(problem: OnTheFlyGStripsProblem, plan: Sequence[str], from_fast_downward: bool = False):
    state = problem.initial_state

    # For some reason, all the names from Fast Downward are lower-cased.
    op_name_mapping = {op.lower(): op for op in problem.operators.keys()}
    object_name_mapping = {obj.lower(): obj for obj in problem.object2index.keys()}

    for plan_string in plan:
        assert plan_string.startswith('(') and plan_string.endswith(')')
        plan_string = plan_string[1:-1]
        operator_name, arguments = plan_string.split(' ', 1)
        arguments = arguments.split(' ')

        if from_fast_downward:
            operator_name = op_name_mapping[operator_name]
            arguments = [object_name_mapping[arg] for arg in arguments]

        operator = problem.operators[operator_name]
        bound_arguments = {arg.name: problem.object2index[argv] for arg, argv in zip(operator.arguments, arguments)}
        assert ogstrips_check_precondition(state, operator, bound_arguments, problem.object2index)
        state = ogstrips_apply_operator(state, operator, bound_arguments, problem.object2index)

    assert problem.goal_test(state.as_state())


def _ogstrips_backward_relavance(problem: OnTheFlyGStripsProblem, state_dict: SStateDict) -> Tuple[OnTheFlyGStripsProblem, SStateDict]:
    warnings.warn('This function is not implemented yet.', NotImplementedError)
    assert problem.conjunctive_goal is not None

    used_predicates = set()
    used_operators = set()

    for proposition in problem.conjunctive_goal:
        used_predicates.add(proposition.split()[0])

    while True:
        new_operator_added = False
        for operator in problem.operators.values():
            if operator.name in used_operators:
                continue

            op_useful = False
            for effect in itertools.chain(operator.add_effects, operator.del_effects):
                if effect.name in used_predicates:
                    op_useful = True
                    break

            if not op_useful:
                continue

            used_operators.add(operator.name)
            new_operator_added = True

            for precondition in operator.preconditions:
                used_predicates.add(precondition.name)

        if not new_operator_added:
            break

    return problem, state_dict


def ogstrips_delete_relaxation_heuristic(problem: OnTheFlyGStripsProblem, state_dict: SStateDict, htype: str, verbose: bool = False) -> float:
    """Heuristic function for the HFF planner.

    Args:
        state_dict: the current state.
        problem: the problem.
        htype: the type of heuristic function to use: 'hmax' or 'hadd' or 'hff'.

    Returns:
        the heuristic value.
    """
    assert htype in ('hadd', 'hmax', 'hff')

    state = state_dict.as_state()
    F_sets = [set(state)]
    A_sets = []
    F_to_A = dict()

    used_operators = set()

    goal_rv = problem.goal_test(state)
    while not goal_rv:
        if verbose:
            print(f'heuristic::current_level = {len(F_sets)}')
        new_state_dict = state_dict.clone()
        new_ops = list()
        new_facts = list()
        for operator, bound_arguments in ogstrips_generate_applicable_actions(problem, state_dict, check_negation=True):
            if verbose:
                print('  operator:', operator.name, bound_arguments)

            grounded_operator_identifier = (operator.name, ) + tuple(bound_arguments[x.name] for x in operator.arguments)
            if grounded_operator_identifier not in used_operators:
                new_ops.append(grounded_operator_identifier)
                used_operators.add(grounded_operator_identifier)

                for predicate in operator.add_effects:
                    name, arguments = ogstrips_bind_arguments(predicate, bound_arguments, problem.object2index)
                    if not new_state_dict.contains(name, arguments):
                        new_state_dict.add(name, arguments)
                        fact_name = f'{name} {" ".join(map(str, arguments))}'
                        new_facts.append(fact_name)
                        F_to_A[fact_name] = (operator, bound_arguments)
                for predicate in operator.del_effects:
                    name, arguments = ogstrips_bind_arguments(predicate, bound_arguments, problem.object2index)
                    name = name + '_not'
                    if not new_state_dict.contains(name, arguments):
                        new_state_dict.add(name, arguments)
                        fact_name = f'{name} {" ".join(map(str, arguments))}'
                        new_facts.append(fact_name)
                        F_to_A[fact_name] = (operator, bound_arguments)

        if len(new_facts) == 0:
            break

        F_sets.append(set(new_facts))
        A_sets.append(set(new_ops))

        state_dict = new_state_dict
        state = state_dict.as_state()
        goal_rv = problem.goal_test(state)

    if not goal_rv:
        return int(1e9)

    F_levels = dict()
    for i, F_set in enumerate(F_sets):
        for fact in F_set:
            F_levels[fact] = i

    if verbose:
        print('F_levels:')
        for fact, level in F_levels.items():
            print(f'  {fact} -> {level}')

    goal_propositions = list()
    if problem.has_complex_goal:
        with GSBoolOutputExpression.enable_forward_diff_ctx():
            # We need to compute the forward-diff of the goal expression.
            goal_rv = problem.goal_test(state)
        goal_propositions = list(goal_rv.propositions)
    else:
        goal_propositions = problem.conjunctive_goal

    if htype == 'hadd':
        h = 0
        for proposition in goal_propositions:
            h += F_levels[proposition]
        return h
    elif htype == 'hmax':
        h = 0
        for proposition in goal_propositions:
            h = max(h, F_levels[proposition])
        return h
    elif htype == 'hff':
        used_actions = set()

        def backtrace(fact):
            if fact in F_to_A:
                operator, bound_arguments = F_to_A[fact]
                grounded_operator_identifier = (operator.name, ) + tuple(bound_arguments[x.name] for x in operator.arguments)
                if grounded_operator_identifier not in used_actions:
                    used_actions.add(grounded_operator_identifier)
                    if verbose:
                        print('  hff::used_action:', operator.name, bound_arguments, '->', fact)
                    for precondition in operator.preconditions:
                        name, arguments = ogstrips_bind_arguments(precondition, bound_arguments, problem.object2index)
                        new_fact = f'{name} {" ".join(map(str, arguments))}'
                        if new_fact in F_levels and (F_levels[new_fact] < F_levels[fact]):
                            backtrace(new_fact)

        for proposition in goal_propositions:
            backtrace(proposition)

        # print('  hff::used_actions:', len(used_actions))
        return len(used_actions)


ogstrips_hadd = functools.partial(ogstrips_delete_relaxation_heuristic, htype='hadd')
ogstrips_hmax = functools.partial(ogstrips_delete_relaxation_heuristic, htype='hmax')
ogstrips_hff = functools.partial(ogstrips_delete_relaxation_heuristic, htype='hff')

