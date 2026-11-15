"""I use these helpers to measure box overlap (IoU)."""

import numpy as np


def bbox_iou(box_a, box_b):
    """I compute the IoU of two boxes in [x, y, width, height] format."""
    ax, ay, aw, ah = map(float, box_a)
    bx, by, bw, bh = map(float, box_b)

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, aw) * max(0, ah)
    area_b = max(0, bw) * max(0, bh)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def compute_iou_matrix(boxes_a, boxes_b):
    """I compute the IoU for every pair of boxes, with shape (len(boxes_a), len(boxes_b))."""
    matrix = np.zeros((len(boxes_a), len(boxes_b)))
    for i, box_a in enumerate(boxes_a):
        for j, box_b in enumerate(boxes_b):
            matrix[i, j] = bbox_iou(box_a, box_b)
    return matrix
