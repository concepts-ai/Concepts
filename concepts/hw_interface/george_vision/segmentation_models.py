#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : segmentation_models.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/05/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import random
from typing import Tuple, List
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


@dataclass
class InstanceSegmentationResult(object):
    masks: np.ndarray
    """A numpy array of shape (N, H, W) and dtype bool."""

    pred_cls: List[str]
    """A list of length N."""

    pred_boxes: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    """A list of length N. Each element is a tuple of ((x1, y1), (x2, y2))."""

    @property
    def nr_objects(self) -> int:
        return len(self.pred_cls)


class ImageBasedPCDSegmentationModel(object):
    def __init__(self, model_name: str = 'maskrcnn_resnet50_fpn_v2', device: str = 'cpu', score_threshold: float = 0.5):
        from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
        from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights

        self.model_name = model_name
        self.device = torch.device(device)

        if self.model_name == 'maskrcnn_resnet50_fpn':
            model_fn = maskrcnn_resnet50_fpn
            weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
        elif self.model_name == 'maskrcnn_resnet50_fpn_v2':
            model_fn = maskrcnn_resnet50_fpn_v2
            weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        else:
            raise ValueError('Unknown model name: {}.'.format(model_name))

        self.model = model_fn(weights=weights)
        self.model.eval()
        self.model.to(self.device)

        self.preprocess = weights.transforms()
        self.score_threshold = score_threshold

    def segment_image(self, image: np.ndarray) -> InstanceSegmentationResult:
        """Segment an image into objects and background.

        Args:
            image: a numpy array of shape (H, W, 3) and dtype uint8.

        Returns:
            A tuple of (segmented_image, object_names, object_bboxes).
            segmented_image is a numpy array of shape (H, W, 3) and dtype uint8.
            object_names is a list of strings.
            object_bboxes is a list of tuples of ((x1, y1), (x2, y2)).
        """

        image = torch.tensor(image.transpose(2, 0, 1) / 255.0, dtype=torch.float32)
        image = self.preprocess(image).to(self.device)
        pred = self.model([image])[0]

        pred_score = list(pred['scores'].detach().cpu().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > self.score_threshold][-1]

        masks = (pred['masks'] > 0.5).squeeze().detach().cpu().numpy()
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred['labels'].detach().cpu().numpy())]
        pred_boxes = [
            ((int(i[0]), int(i[1])), (int(i[2]), int(i[3])))
            for i in list(pred['boxes'].detach().cpu().numpy().astype(np.int64))
        ]

        masks = masks[:pred_t+1]
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        return InstanceSegmentationResult(masks, pred_class, pred_boxes)


class PointGuidedImageSegmentationModel(object):
    def __init__(self, checkpoint_path, model: str = 'sam_default', device: str = 'cpu'):
        from segment_anything import SamPredictor, sam_model_registry

        if model == 'sam_default':
            model = sam_model_registry['default']
        else:
            raise ValueError('Unknown model name: {}.'.format(model))
        self.model = model(checkpoint=checkpoint_path)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.predicator = SamPredictor(self.model)
        self.last_image_id = None

    def segment_from_point(self, image, point):
        if self.last_image_id is None or self.last_image_id != id(image):

            self.predicator.set_image(image)
            self.last_image_id = id(image)
        masks, _, _ = self.predicator.predict(point_coords=np.array([point]), point_labels=np.array([1]))
        if len(masks) > 1:
            mask = masks[-2].astype(np.uint8)
        else:
            mask = masks[-1].astype(np.uint8)
        mask = remove_remains(mask, point)
        return mask


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def remove_remains(img, interest_point):
    """Remove remains which are not adjacent with interest_point."""
    img = img.copy().astype(np.uint8)
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    img_inv = img.copy()
    cv2.floodFill(img_inv, mask, tuple(interest_point), 0)
    img -= img_inv
    return img


def random_colored_mask(image):
    colours = [
        [0, 255, 0], [0, 0, 255], [255, 0, 0],
        [0, 255, 255], [255, 255, 0], [255, 0, 255],
        [80, 70, 180], [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]
    ]
    rgb = np.array(random.choice(colours))
    rgb = np.reshape(rgb, (1, 1, 3))
    image = image[:, :, None]
    image = np.where(image != 0, image * rgb, image)
    return image.astype(np.uint8)


def visualize_instance_segmentation(image: np.ndarray, result: InstanceSegmentationResult, rect_th: int = 3, text_size: int = 1, text_th: int = 3):
    image = image.copy()
    for i in range(len(result.masks)):
        if 'table' in result.pred_cls[i]:
            continue
        rgb_mask = random_colored_mask(result.masks[i])
        image = cv2.addWeighted(image, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(image, result.pred_boxes[i][0], result.pred_boxes[i][1], color=(0, 255, 0), thickness=rect_th)
        cv2.putText(image, result.pred_cls[i], (result.pred_boxes[i][0][0], result.pred_boxes[i][0][1] + 10), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()

