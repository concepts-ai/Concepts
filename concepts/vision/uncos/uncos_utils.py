import logging
import cv2
import numpy as np
import matplotlib
from typing import List, Tuple, Dict, Union
from dataclasses import dataclass, field
from functools import partial, reduce
import open3d as o3d

MIN_AREA_PERCENTAGE = .001


@dataclass
class MaskWrapper:
    rawmask: Dict

    def __call__(self, *args, **kwargs):
        return self.rawmask['segmentation']

    @property
    def score(self):
        return self.rawmask['predicted_iou']

    def copy(self):
        return MaskWrapper(self.rawmask.copy())


@dataclass
class SegHypothesis:
    """
    A segmentation hypothesis is a partition of the area
    """
    masks: List[np.ndarray] = field(default_factory=list)
    mask_scores: List = field(default_factory=list)
    confidence: int = 1

    def add_part(self, mask, score=.88):
        self.masks.append(mask)
        self.mask_scores.append(score)

    @property
    def area_mask(self):
        return reduce(np.logical_or, self.masks)

    @property
    def mask_num(self):
        return len(self.masks)

    def increment_confidence(self):
        self.confidence += 1


class RegionHypothesis:
    """
    A RegionHypothesis contains multiple SegHypothesis. Add some data structure for compact data management.
    """

    def __init__(self, hypotheses: List[SegHypothesis]):
        distinct_masks_bool, distinct_masks_scores, region_hypotheses_corresponding_masks, \
            region_hypotheses_corresponding_scores = self.create_hypotheses(hypotheses)
        self.masks_union: List[np.ndarray] = distinct_masks_bool
        self.masks_union_scores: List = distinct_masks_scores
        self.region_hypotheses_2d_corresponding_masks = region_hypotheses_corresponding_masks
        self.region_hypotheses_2d_corresponding_scores = region_hypotheses_corresponding_scores

        area_mask = reduce(np.logical_or, self.masks_union)
        self.area_mask = area_mask

    @property
    def hypotheses_num(self):
        return len(self.region_hypotheses_2d_corresponding_masks)

    def get_most_likely_hypothesis(self):
        which_hyp_highest_score = np.argmax(self.region_hypotheses_2d_corresponding_scores)
        return [self.masks_union[i] for i in self.region_hypotheses_2d_corresponding_masks[which_hyp_highest_score]]

    def get_region_hypotheses_masks(self):
        return [[self.masks_union[i] for i in region_hyp_j_corresponding_mask]
            for region_hyp_j_corresponding_mask in self.region_hypotheses_2d_corresponding_masks]

    def create_hypotheses(self, hypotheses: List[SegHypothesis]):
        distinct_masks = []
        distinct_mask_scores = []
        region_hypotheses_2d_corresponding_masks = []
        region_hypotheses_2d_corresponding_scores = []
        for seghypothesis in hypotheses:
            maskid_this_hypothesis = []
            for (mask, mask_score) in zip(seghypothesis.masks, seghypothesis.mask_scores):
                mask2d_iou_to_prevmasks = [iou_fn(mask, mask_prev) for mask_prev in distinct_masks]
                if len(mask2d_iou_to_prevmasks) == 0 or max(mask2d_iou_to_prevmasks) <= .95:
                    maskid_this_hypothesis.append(len(distinct_masks))
                    distinct_masks.append(mask)
                    distinct_mask_scores.append([mask_score])
                    continue
                which_prevmask_same = np.argsort(mask2d_iou_to_prevmasks)[-1]
                # logging.info(f'iou2prevhypo mask is {mask2d_iou_to_prevmasks[which_prevmask_same]}')
                maskid_this_hypothesis.append(which_prevmask_same)
                distinct_mask_scores[which_prevmask_same].append(mask_score)

            if set(maskid_this_hypothesis) in region_hypotheses_2d_corresponding_masks:
                # logging.info(f'mask id this is {maskid_this_hypothesis}. prev is {region_hypotheses_2d_corresponding_masks}. skip cur')
                continue
            skip_cur = False

            filtered_subsets_id = []
            for x_i, x in enumerate(region_hypotheses_2d_corresponding_masks):
                # may be resulted from volume filter
                if set(x).issubset(set(maskid_this_hypothesis)):
                    # logging.info(f'prev set {x} is subset {maskid_this_hypothesis}. pop prev set.')
                    continue
                if set(maskid_this_hypothesis).issubset(set(x)):
                    # logging.info(f'cur set {maskid_this_hypothesis} is subset of prev set {x}. skip cur1')
                    skip_cur = True
                filtered_subsets_id.append(x_i)
            filtered_sets, filtered_scores = [region_hypotheses_2d_corresponding_masks[x_i] for x_i in filtered_subsets_id], \
                [region_hypotheses_2d_corresponding_scores[x_i] for x_i in filtered_subsets_id]
            region_hypotheses_2d_corresponding_masks, region_hypotheses_2d_corresponding_scores = filtered_sets, filtered_scores

            if skip_cur:
                continue
            region_hypotheses_2d_corresponding_masks.append(set(maskid_this_hypothesis))
            region_hypotheses_2d_corresponding_scores.append(seghypothesis.confidence)
        assert len(region_hypotheses_2d_corresponding_masks) == len(region_hypotheses_2d_corresponding_scores)
        distinct_mask_scores_avg = [np.mean(x) for x in distinct_mask_scores]

        return distinct_masks, distinct_mask_scores_avg, region_hypotheses_2d_corresponding_masks, region_hypotheses_2d_corresponding_scores


def draw_seg_on_im(im, pred_masks: List[np.ndarray], alpha=.5, colors=None, plot_anno=None, contour_width=6, fill_paint_area=True):
    """
    Args:
        im:     HxWx3 image array
        pred_masks:   list of HxW binary mask
        alpha:

    Returns:
        im with masks overlayed.
    """
    im = im.copy()
    n_colors = len(pred_masks)
    # cm = sns.color_palette('pastel',n_colors=n_colors)
    cm_fn = matplotlib.colormaps['jet']
    cm = [cm_fn(i / n_colors)[:3] for i in range(n_colors)]

    for i, obj_mask in enumerate(pred_masks):
        contours, hier = cv2.findContours(
            obj_mask.astype(np.uint8).copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        contours = [x for x in contours if cv2.contourArea(x) > 50]
        if colors is not None:
            assert np.array(colors).max() > 2
            contour_colors = [tuple(map(int, np.array(colors))) for _ in range(len(contours))]
        else:
            contour_colors = [tuple(map(int, np.array(cm[i][:3]) * 255))] * len(contours)
        for (contour_color, contour) in zip(contour_colors, contours):
            cv2.drawContours(im, contour, -1, contour_color, thickness=contour_width)
        if fill_paint_area:
            im = im.astype(np.float32)
            im[obj_mask > 0] = im[obj_mask > 0] * alpha + np.array(contour_colors[0]) / np.array(contour_colors[0]).max() * (1 - alpha) * 255
            im = im.astype(np.uint8)
    if plot_anno is not None:
        for line_i, text in enumerate(plot_anno.splitlines()):
            cv2.putText(
                im, text, (0, 20 * (line_i + 1)), fontScale=.3, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA,
                fontFace=cv2.FONT_HERSHEY_COMPLEX
            )
    return im


def overlay_maskwrappers(rgb_im, masks_wrappeed: List[MaskWrapper], colors=None):
    return overlay_masks(rgb_im, [x() for x in masks_wrappeed], colors)


def overlay_masks(rgb_im, masks: List[np.ndarray], colors=None, plot_anno=None, contour_width=2, paint_area=False):
    if len(masks) == 0:
        return rgb_im
    sorted_masks = sorted(masks, key=(lambda x: x.sum()), reverse=True)
    overlayed_mask = draw_seg_on_im(
        rgb_im, sorted_masks, colors=colors, plot_anno=plot_anno,
        contour_width=contour_width, fill_paint_area=paint_area
    )
    return overlayed_mask


def overlay_mask_simple(rgb_im, mask: np.ndarray, colors=None, mask_alpha=.5):
    if rgb_im.max() > 2:
        rgb_im = rgb_im.astype(np.float32) / 255.
    if colors is None:
        colors = np.array([1, 0, 0])
    return (rgb_im * (1 - mask_alpha) + mask[..., np.newaxis] * colors * mask_alpha).copy()


def crop(im, mask, margin_pixel=10, return_bbox=False):
    h, w = im.shape[:2]
    bbox_y, bbox_x = np.where(mask > 0)
    ymin, ymax, xmin, xmax = max(0, bbox_y.min() - margin_pixel), min(bbox_y.max() + margin_pixel, h), \
        max(0, bbox_x.min() - margin_pixel), min(bbox_x.max() + margin_pixel, w)
    if return_bbox:
        return im[ymin:ymax, xmin:xmax], ymin, ymax, xmin, xmax
    return im[ymin:ymax, xmin:xmax]


def iou_fn(a, b):
    return (a.astype(bool) & b.astype(bool)).sum() / (a.astype(bool) | b.astype(bool)).sum()


def iom_fn(a, b):
    return (a.astype(bool) & b.astype(bool)).sum() / min(a.astype(bool).sum(), b.astype(bool).sum())


def intersection_fn(a, b):
    return (a.astype(bool) & b.astype(bool)).sum()


# remove small regions. modified from SAM `https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/utils/amg.py#L267C5-L267C25`
def remove_trivial_regions(
    mask: np.ndarray, is_nontrivial, mode: str
) -> Tuple[np.ndarray, bool]:
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    If using 'holes' mode, will fill in the small holes and add them to `mask`.
    """
    import cv2  # type: ignore

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)

    sizes = stats[:, -1][1:]  # Row 0 is background label
    # small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    small_regions = [i + 1 for i, s in enumerate(sizes) if (not is_nontrivial(regions == i + 1))]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True


def is_degenerated_pointcloud(pointcloud_raw, min_edge_len_threshold=.01):
    if pointcloud_raw is None:
        return False
    if len(pointcloud_raw) < 30:
        return True
    # look at only valid points (depth!=nan)
    pointcloud = pointcloud_raw[(pointcloud_raw != 0).all(axis=1)]

    if len(pointcloud) < 30:
        return True

    # check aabb
    aabb = [pointcloud.min(axis=0), pointcloud.max(axis=0)]
    aabb_len_of_each_edge = [(aabb[1][i] - aabb[0][i]) for i in range(3)]
    if min(aabb_len_of_each_edge) < min_edge_len_threshold:
        return True

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pointcloud))
    oriented_bounding_box = pcd.get_oriented_bounding_box()
    oobb_extent = oriented_bounding_box.extent
    if min(oobb_extent) < min_edge_len_threshold:
        return True
    return False


def is_degenerated_mask(
    mask: np.ndarray, pointcloud=None, min_area_percentage_threshold=MIN_AREA_PERCENTAGE,
    narrow_area_threshold=10
) -> Tuple[Union[np.ndarray, None], bool]:
    """
    Determine if a mask is degenerated. If not, fill in the small holes and remove flying pixels.
    """
    im_h, im_w = mask.shape[:2]
    min_area_threshold = int(min_area_percentage_threshold * im_h * im_w)

    def is_valid(msk, area_thres, bbox_len_thres):
        bbox_y, bbox_x = np.where(msk > 0)
        if not msk.sum() >= area_thres:
            return False
        is_narrow = bbox_y.max() - bbox_y.min() < bbox_len_thres or bbox_x.max() - bbox_x.min() < bbox_len_thres
        return not is_narrow

    if not is_valid(mask, min_area_threshold, narrow_area_threshold):
        return np.zeros_like(mask).astype(bool), True

    mask, is_modified = remove_trivial_regions(
        mask,
        partial(is_valid, area_thres=min_area_threshold * 1.5, bbox_len_thres=narrow_area_threshold),
        mode='holes'
    )
    if not is_valid(mask, min_area_threshold, narrow_area_threshold):
        return np.zeros_like(mask).astype(bool), True
    mask, is_modified = remove_trivial_regions(
        mask,
        partial(is_valid, area_thres=min_area_threshold, bbox_len_thres=narrow_area_threshold),
        mode='islands'
    )
    if not is_valid(mask, min_area_threshold, narrow_area_threshold):
        return np.zeros_like(mask).astype(bool), True
    if pointcloud is not None and is_degenerated_pointcloud(pointcloud[mask]):
        return None, True
    return mask, False


def bfs_cluster(adjacency_matrix, threshold=.5):
    """
    Doing bfs to find clusters according to adjacency matrix
    Args:
        adjacency_matrix: N x N matrix. number indicates 'connectivity' between two nodes. the higher the closer
        threshold: if adjacency > threshold, treat them as 'connected'

    Returns:
        Set of clustered sets (ID) {{0,1,4},{2,3}} .
    """
    clusters = []
    n_nodes, _ = adjacency_matrix.shape
    clustered = set()

    def get_neighbor_from_j_except_self(node_j):
        nb_list = list(np.where(adjacency_matrix[node_j] > threshold)[0])
        nb_list.remove(node_j)
        return sorted(nb_list)

    for i in range(n_nodes):
        if i in clustered:
            continue
        cluster = {i}
        nb_list = get_neighbor_from_j_except_self(i)
        while len(nb_list) > 0:
            next_node = nb_list[0]
            nb_list.remove(next_node)
            if next_node in cluster:
                continue
            cluster.add(next_node)
            nb_of_nextnode = get_neighbor_from_j_except_self(next_node)
            nb_list = sorted(list(set(nb_list).union(set(nb_of_nextnode) - cluster)))
        clusters.append(cluster)
        clustered = clustered.union(cluster)
    np.set_printoptions(suppress=True)
    return clusters


if __name__ == '__main__':
    test_adj_matrix = np.eye(7)
    test_adj_matrix[0, 2] = test_adj_matrix[2, 0] = 1
    test_adj_matrix[2, 4] = test_adj_matrix[4, 2] = 1
    test_adj_matrix[5, 6] = test_adj_matrix[6, 5] = 1
    print(test_adj_matrix)
    clusters = bfs_cluster(test_adj_matrix)
    print('-' * 10)
    print('clusters:')
    for cluster in clusters:
        print(cluster)