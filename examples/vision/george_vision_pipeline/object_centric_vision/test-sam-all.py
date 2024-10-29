#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : 1.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/20/2023
#
# Distributed under terms of the MIT license.

import cv2
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

print('Usage: python3 test-sam-all.py <image_path>')

if len(sys.argv) != 2:
    print('Invalid arguments.')
    exit(1)

print('Loading the SAM model...')
sam = sam_model_registry['default'](checkpoint="./data/sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)

print('Loading and segmenting the image...')
img = cv2.imread(sys.argv[1])
masks = mask_generator.generate(img)

segmentation_masks = list()
for mask in masks:
    segmentation_masks.append(mask['segmentation'])

# visualize the segmentation masks
# each mask is a image
for mask in segmentation_masks:
    # 1 x 2 plot, first is the original image, second is the mask
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img)
    ax[1].imshow(mask)
    plt.show()


# visualize the union of the all masks

union_mask = np.zeros_like(segmentation_masks[0], dtype=np.uint8)
for i, mask in enumerate(segmentation_masks[4:]):
    union_mask[mask > 0] = i

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img)
axes[1].imshow(union_mask)
plt.show()

