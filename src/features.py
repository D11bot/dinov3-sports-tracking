"""I extract frozen DINOv3 embeddings from player crops.

I import torch and transformers inside the functions that need them, so I can
run the tracking code and the tests without them.
"""

from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"
EMBED_DIM = 384  # ViT-S/16


def load_dinov3(model_id=DEFAULT_MODEL, token=None):
    import torch
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_id, token=token)
    model = AutoModel.from_pretrained(model_id, token=token)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return processor, model.to(device).eval(), device


def crop_window(image, box):
    """I turn an [x, y, w, h] box into an integer crop window inside the image."""
    x, y, w, h = box
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(image.width, int(x + w))
    y2 = min(image.height, int(y + h))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop for box {box}")
    return x1, y1, x2, y2


def extract_features(image, boxes, processor, model, device, batch_size=16):
    """I return one L2-normalized DINOv3 embedding for each box."""
    if not len(boxes):
        return np.empty((0, EMBED_DIM), dtype=np.float32)

    import torch
    import torch.nn.functional as F

    crops = [image.crop(crop_window(image, box)) for box in boxes]
    result = []
    for start in range(0, len(crops), batch_size):
        inputs = processor(images=crops[start : start + batch_size], return_tensors="pt").to(device)
        with torch.inference_mode():
            embeddings = F.normalize(model(**inputs).pooler_output, dim=-1)
        result.append(embeddings.cpu().numpy())
    return np.concatenate(result, axis=0)


def extract_sequence(sequence_dir, processor, model, device, num_frames=120, batch_size=16):
    """I read the GT boxes, extract DINOv3 features and keep the GT IDs for the first frames."""
    from .data import read_ground_truth

    sequence_dir = Path(sequence_dir)
    gt = read_ground_truth(sequence_dir)
    result = {}

    for frame_id in range(1, num_frames + 1):
        path = sequence_dir / "img1" / f"{frame_id:06d}.jpg"
        if not path.exists():
            raise FileNotFoundError(f"Expected consecutive frame: {path}")
        image = Image.open(path).convert("RGB")

        rows = gt[gt.iloc[:, 0] == frame_id]
        boxes = []
        ids = []
        for _, row in rows.iterrows():
            if int(row.iloc[6]) != 1:
                continue
            x, y, w, h = map(float, row.iloc[2:6])
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(image.width, int(x + w))
            y2 = min(image.height, int(y + h))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            ids.append(int(row.iloc[1]))

        boxes = np.array(boxes, dtype=float).reshape(-1, 4)
        result[frame_id] = {
            "boxes": boxes,
            "features": extract_features(image, boxes, processor, model, device, batch_size),
            "gt_ids": np.asarray(ids, dtype=int),
        }

    return result
