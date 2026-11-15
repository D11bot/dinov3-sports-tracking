"""I create synthetic missing detections, box jitter, false positives and gaps.
I keep the GT IDs for evaluation, while my trackers match with boxes and features.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from .association import bbox_iou
from .features import extract_features


def jitter_box(box, image_size, noise, rng):
    """I move the center by N(0, noise * w) in x and N(0, noise * h) in y, and scale
    w and h by exp(N(0, noise)). I keep the box inside the image."""
    W, H = image_size
    x, y, w, h = map(float, box)

    cx = x + w / 2 + rng.normal(0, noise * w)
    cy = y + h / 2 + rng.normal(0, noise * h)

    w = max(3.0, w * np.exp(rng.normal(0, noise)))
    h = max(3.0, h * np.exp(rng.normal(0, noise)))

    x1 = np.clip(cx - w / 2, 0, W - 2)
    y1 = np.clip(cy - h / 2, 0, H - 2)
    x2 = np.clip(cx + w / 2, x1 + 1, W)
    y2 = np.clip(cy + h / 2, y1 + 1, H)

    return np.array([x1, y1, x2 - x1, y2 - y1])


def create_false_box(image_size, existing_boxes, rng):
    """I place a random player sized box with IoU < 0.05 to all real boxes (up to 100 tries)."""
    W, H = image_size

    for _ in range(100):
        w = float(rng.uniform(25, 70))
        h = float(rng.uniform(55, 140))
        x = float(rng.uniform(0, max(1, W - w)))
        y = float(rng.uniform(0, max(1, H - h)))

        candidate = np.array([x, y, w, h])
        if all(bbox_iou(candidate, b) < 0.05 for b in existing_boxes):
            return candidate

    return None


def corrupt_detection_data(
    source,
    missing_rate=0.0,
    box_noise=0.0,
    fp_per_frame=0,
    seed=42,
    image_dir=None,
    processor=None,
    model=None,
    device=None,
):
    """I apply missing detections, box jitter and false positives frame by frame.

    I extract new features for every jittered or false positive box, so the
    appearance input matches the changed crop. I give false positives GT ID -1.
    """
    if image_dir is None:
        raise ValueError("image_dir must be provided explicitly")
    image_dir = Path(image_dir)
    rng = np.random.default_rng(seed)
    result = {}

    for frame_id, data in sorted(source.items()):
        image = Image.open(image_dir / f"{frame_id:06d}.jpg").convert("RGB")

        keep = rng.random(len(data["boxes"])) >= missing_rate
        boxes = data["boxes"][keep].copy()
        gt_ids = data["gt_ids"][keep].copy()
        features = data["features"][keep].copy()

        changed = []

        for i in range(len(boxes)):
            if box_noise > 0:
                boxes[i] = jitter_box(boxes[i], image.size, box_noise, rng)
                changed.append(i)

        for _ in range(fp_per_frame):
            fp_box = create_false_box(image.size, data["boxes"], rng)
            if fp_box is not None:
                boxes = np.vstack([boxes, fp_box])
                gt_ids = np.append(gt_ids, -1)
                features = np.vstack([features, np.zeros((1, features.shape[1]))])
                changed.append(len(boxes) - 1)

        if changed:
            # I use batch size 32 here, as in the runs behind results/reference/.
            features[changed] = extract_features(
                image, boxes[changed], processor, model, device, batch_size=32
            )

        result[frame_id] = {"boxes": boxes, "features": features, "gt_ids": gt_ids}

    return result


def create_detection_gap(source_data, target_gt_id, start_frame, gap_length):
    """I remove target_gt_id from gap_length frames, starting at start_frame.
    I leave the reference data unchanged."""
    modified = {}

    for frame_id, data in source_data.items():
        keep = np.ones(len(data["gt_ids"]), dtype=bool)
        if start_frame <= frame_id < start_frame + gap_length:
            keep = data["gt_ids"] != target_gt_id

        modified[frame_id] = {
            "boxes": data["boxes"][keep].copy(),
            "features": data["features"][keep].copy(),
            "gt_ids": data["gt_ids"][keep].copy(),
        }

    return modified
