import gc
from typing import Optional, Union, List

from tqdm import tqdm
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F

from concepts.vision.fm_match.dino.extractor_dino import ViTExtractor


def resize(img: Union[np.ndarray, Image.Image], target_res, resize=True, to_pil=True, edge=False):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    original_width, original_height = img.size
    original_channels = len(img.getbands())
    if not edge:
        canvas = np.zeros([target_res, target_res, 3], dtype=np.uint8)
        if original_channels == 1:
            canvas = np.zeros([target_res, target_res], dtype=np.uint8)
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            canvas[(width - height) // 2: (width + height) // 2] = img
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            canvas[:, (height - width) // 2: (height + width) // 2] = img
    else:
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            top_pad = (target_res - height) // 2
            bottom_pad = target_res - height - top_pad
            img = np.pad(img, pad_width=[(top_pad, bottom_pad), (0, 0), (0, 0)], mode='edge')
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            left_pad = (target_res - width) // 2
            right_pad = target_res - width - left_pad
            img = np.pad(img, pad_width=[(0, 0), (left_pad, right_pad), (0, 0)], mode='edge')
        canvas = img
    if to_pil:
        canvas = Image.fromarray(canvas)
    return canvas


def torch_pca(feature: torch.Tensor, target_dim: int = 256) -> torch.Tensor:
    """
    Perform Principal Component Analysis (PCA) on the input feature tensor.

    Parameters:
    - feature (torch.Tensor): The input tensor with shape (N, D), where N is the number of samples
      and D is the feature dimension.
    - target_dim (int, optional): The target dimension for the output tensor. Defaults to 256.

    Returns:
    - torch.Tensor: The transformed tensor with shape (N, target_dim).
    """
    mean = torch.mean(feature, dim=0, keepdim=True)
    centered_features = feature - mean
    U, S, V = torch.pca_lowrank(centered_features, q=target_dim)
    reduced_features = torch.matmul(centered_features, V[:, :target_dim])

    return reduced_features


def compute_dino_feature(
    source_img: Union[Image.Image, List[Union[np.ndarray, Image.Image]]],
    target_imgs: Optional[List[Union[np.ndarray, Image.Image]]] = None,
    *,
    model_size: str = 'base',
    use_dino_v2: bool = True,
    stride: Optional[int] = None,
    edge_pad: bool = False,
    pca: bool = False,
    pca_dim: int = 256,
    reusable_extractor: Optional[ViTExtractor] = None
) -> tuple[torch.Tensor, List[Image.Image], List[Image.Image]]:
    """
    return: (result, resized_imgs, downsampled_imgs), where result is a tensor of shape (N, pca_dim, num_patches, num_patches),
        resized_imgs is a list of PIL image_scene resized to the input size of the dino model, and downsampled_imgs is a list
        of PIL image_scene resized to the output size of the dino model.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img_size = 840 if use_dino_v2 else 244
    if reusable_extractor is None:
        model_dict = {'small': 'dinov2_vits14',
                      'base': 'dinov2_vitb14',
                      'large': 'dinov2_vitl14',
                      'giant': 'dinov2_vitg14'}

        model_type = model_dict[model_size] if use_dino_v2 else 'dino_vits8'
        layer = 11 if use_dino_v2 else 9
        if 'l' in model_type:
            layer = 23
        elif 'g' in model_type:
            layer = 39
        facet = 'token' if use_dino_v2 else 'key'
        if stride is None:
            stride = 14 if use_dino_v2 else 4
        extractor = ViTExtractor(model_type, stride, device=device)
    else:
        extractor = reusable_extractor

    patch_size = extractor.model.patch_embed.patch_size[0] if use_dino_v2 else extractor.model.patch_embed.patch_size
    num_patches = int(patch_size / stride * (img_size // patch_size - 1) + 1)

    original_imgs = list()
    if isinstance(source_img, (np.ndarray, Image.Image)):
        original_imgs.append(source_img)
    else:
        original_imgs.extend(source_img)
    if target_imgs is not None:
        if isinstance(target_imgs, (np.ndarray, Image.Image)):
            original_imgs.append(target_imgs)
        else:
            original_imgs.extend(target_imgs)

    result = []
    resized_imgs = [resize(img, img_size, resize=True, to_pil=True, edge=edge_pad) for img in original_imgs]

    for img in tqdm(resized_imgs, desc='Extracting dino feature'):
        with torch.no_grad():
            img_batch = extractor.preprocess_pil(img)
            img_desc = extractor.extract_descriptors(
                img_batch.to(device), layer, facet)  # 1,1,num_patches*num_patches, feature_dim
            result.append(img_desc)

    result = torch.concat(result, dim=0)  # N, 1, num_patches*num_patches, feature_dim
    if pca:
        N, _, _, feature_dim = result.shape
        result = result.reshape(-1, feature_dim)
        result = torch_pca(result, pca_dim)
        result = result.reshape(N, 1, -1, pca_dim)

    result = result.permute(0, 1, 3, 2).reshape(result.shape[0], result.shape[-1], num_patches, num_patches)
    result = F.normalize(result, dim=1)

    gc.collect()
    torch.cuda.empty_cache()

    output_size = result.shape[-1]
    downsampled_imgs = [resize(img, output_size, resize=True, to_pil=True, edge=edge_pad) for img in original_imgs]

    return result, resized_imgs, downsampled_imgs
