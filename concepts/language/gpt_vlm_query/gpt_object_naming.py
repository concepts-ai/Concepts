#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : gpt_object_naming.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/12/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import numpy as np
import cv2
import base64
from typing import Tuple, List

from concepts.language.gpt_vlm_query.gpt_image_query_utils import draw_masks
from concepts.language.openai_utils.default_client import get_default_client
from concepts.language.openai_utils.llm_prompting_utils import extract_tag, ParsingFailedError, auto_retry


@auto_retry(5)
def get_object_names(image: np.ndarray, masks: List[np.ndarray], class_names: List[str], client=None) -> List[str]:
    """Get the object names from the given image and masks.

    Args:
        image: the input image.
        masks: a list of binary masks.
        class_names: a list of class names.

    Returns:
        The list of object names.
    """

    if client is None:
        client = get_default_client()

    instruction = 'I have provided an image with masks. Please identify the object that is represented by each mask.\n'
    instruction += 'One mask corresponds to a single object. Your output should be multiple lines, each line corresponding to a mask of the format <name>{object_name}</name>.\n'
    instruction += 'Choose the names from the following list: ' + ', '.join(class_names) + '.\n'
    instruction += 'There are ' + str(len(masks)) + ' masks in the image.'

    visualization = draw_masks(image, masks)
    # Use base64 encoding to send the image to the GPT model.
    encoded_file = cv2.imencode('.jpg', visualization[..., ::-1])
    encoded_image = base64.b64encode(encoded_file[1]).decode('utf-8')
    b64_image = f'data:image/jpeg;base64,{encoded_image}'

    messages = [
        {'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': b64_image}},
            {'type': 'text', 'text': instruction},
        ]}
    ]

    gpt_return_result = client.chat.completions.create(model="gpt-4o", messages=messages)
    print('GPT return result:', gpt_return_result.choices[0].message.content)
    tags = extract_tag(gpt_return_result.choices[0].message.content, 'name')
    if len(tags) != len(masks):
        raise ParsingFailedError('Failed to parse the object names from the GPT model output.')
    return tags


def get_object_names_for_multiple_masks(image_mask_pairs: List[Tuple[np.ndarray, np.ndarray]], class_names: List[str]) -> List[str]:
    """Get the object names from the given image-mask pairs.

    Args:
        image_mask_pairs: a list of image-mask pairs.
        class_names: a list of class names.

    Returns:
        The list of object names.
    """

    # Merge masks belonging to the same image into a single query.
    images = {id(image): image for image, _ in image_mask_pairs}
    image_mask_dict = {image_id: [] for image_id in images}
    for i, (image, mask) in enumerate(image_mask_pairs):
        image_mask_dict[id(image)].append((i, mask))

    object_names = [None for _ in range(len(image_mask_pairs))]
    for image_id, index_mask_pairs in image_mask_dict.items():
        image, masks = images[image_id], [mask for _, mask in index_mask_pairs]
        this_image_object_names = get_object_names(image, masks, class_names)
        for (i, _), object_name in zip(index_mask_pairs, this_image_object_names):
            object_names[i] = object_name

    return object_names

