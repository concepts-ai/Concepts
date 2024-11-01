#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : run.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/31/2024
#
# Distributed under terms of the MIT license.

"""This exmaple demonstrates how to use the CrowExecutor to execute the DSL code with the custom function implementation.

In this case, it is used as a first-order logic engine, and the domain is defined in the Crow language.
"""

import torch
from concepts.dsl.dsl_types import BOOL
from concepts.dsl.tensor_value import TensorValue
from concepts.dsl.tensor_state import StateObjectReference, StateObjectList
import concepts.dm.crow as crow

domain = crow.load_domain_string(f'''
domain "basic"

typedef Object: object
typedef Motion: object

def is_red(o: Object) -> bool
def is_green(o: Object) -> bool
def is_blue(o: Object) -> bool
def is_subject_of(m: Motion, o: Object) -> bool
def is_object_of(m: Motion, o: Object) -> bool
def is_push_motion(m: Motion) -> bool
def is_pull_motion(m: Motion) -> bool
''')

state = crow.CrowState.make_empty_state(
    domain,
    {'object1': 'Object', 'object2': 'Object', 'motion1': 'Motion', 'motion2': 'Motion'}
)

object_valid_list = {
    'object1': False,
    'object2': False
}

object_color_list = {
    'object1': 'red',
    'object2': 'blue'
}

motion_subject_list = {
    'motion1': 'object1',
    'motion2': 'object2'
}

motion_object_list = {
    'motion1': 'object2',
    'motion2': 'object1'
}

motion_type_list = {
    'motion1': 'push',
    'motion2': 'pull'
}

@crow.config_function_implementation(support_batch=False)
def is_subject_of(motion_name: str, object_name: str) -> TensorValue:
    return motion_subject_list[motion_name] == object_name

@crow.config_function_implementation(support_batch=False)
def is_object_of(motion_name: str, object_name: str) -> TensorValue:
    return motion_object_list[motion_name] == object_name

# If you specfiy support_batch=True, the executor will internally do a for-loop over the batch dimension.
@crow.config_function_implementation(support_batch=False)
def is_red(object_name: str) -> TensorValue:
    return object_color_list[object_name] == 'red'

# You can also implement your own function that supports batch processing.
@crow.config_function_implementation(support_batch=True)
def is_red(object_name: str | slice) -> TensorValue:
    if isinstance(object_name, str):
        return object_color_list[object_name] == 'red'
    return TensorValue.from_tensor(
        torch.tensor([object_color_list[on] == 'red' for on in object_color_list], dtype=torch.bool),
        BOOL, batch_variables=['o']  # The batch variable name here need to be consistent with the function signature.
    )

@crow.config_function_implementation(support_batch=False)
def is_green(object_name: str) -> TensorValue:
    return object_color_list[object_name] == 'green'

@crow.config_function_implementation(support_batch=False)
def is_blue(object_name: str) -> TensorValue:
    return object_color_list[object_name] == 'blue'

@crow.config_function_implementation(support_batch=False)
def is_push(motion_name: str) -> TensorValue:
    return motion_type_list[motion_name] == 'push'

@crow.config_function_implementation(support_batch=False)
def is_pull(motion_name: str) -> TensorValue:
    return motion_type_list[motion_name] == 'pull'


executor = crow.CrowExecutor(domain)
executor.register_function_implementation('is_red', is_red)
executor.register_function_implementation('is_green', is_green)
executor.register_function_implementation('is_blue', is_blue)
executor.register_function_implementation('is_subject_of', is_subject_of)
executor.register_function_implementation('is_object_of', is_object_of)
executor.register_function_implementation('is_push_motion', is_push)
executor.register_function_implementation('is_pull_motion', is_pull)

print(executor.execute('exists x: Object: is_red(x)', state=state))
print(executor.execute('exists x: Object: is_green(x)', state=state))
print(executor.execute('exists m: Motion: exists x: Object: exists y: Object: is_push_motion(m) and is_subject_of(m, x) and is_object_of(m, y) and is_red(x) and is_blue(y)', state=state))
print(executor.execute('exists m: Motion: exists x: Object: exists y: Object: is_pull_motion(m) and is_subject_of(m, x) and is_object_of(m, y) and is_red(x) and is_blue(y)', state=state))
print(executor.execute('findone m: Motion: exists x: Object: exists y: Object: is_push_motion(m) and is_subject_of(m, x) and is_object_of(m, y) and is_red(x) and is_blue(y)', state=state))

# domain = domain.clone()
domain.incremental_define(f'''
def get_motion() -> Motion:
  let x = findone x: Object: is_red(x)
  let y = findone y: Object: is_blue(y)
  let m = findone m: Motion: is_push_motion(m) and is_subject_of(m, x) and is_object_of(m, y)
  return m
''')
print(executor.execute('get_motion()', state=state))
del domain.functions['get_motion']

