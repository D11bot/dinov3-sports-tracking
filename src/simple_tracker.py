"""I compare three simple tracking methods with fixed matching thresholds."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .association import compute_iou_matrix

METHODS = ("iou", "appearance", "fusion")

IOU_THRESHOLD = 0.1
APPEARANCE_THRESHOLD = 0.78
FUSION_THRESHOLD = 0.55
FUSION_MIN_IOU = 0.05


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    feature: np.ndarray
    missed: int = 0
    last_seen: int = 0


class SimpleMOTTracker:
    """I match detections to the last box and feature of each track.

    iou:        I use IoU as the score and accept IoU >= 0.1
    appearance: I use cosine similarity and accept values >= 0.78
    fusion:     I use alpha * (cos + 1) / 2 + (1 - alpha) * IoU and accept
                score >= 0.55 when IoU >= 0.05 or cos >= 0.78
    """

    def __init__(self, method="fusion", max_age=5, alpha=0.5):
        if method not in METHODS:
            raise ValueError(f"Unknown tracking method: {method}")

        self.method = method
        self.max_age = max_age
        self.alpha = alpha
        self.tracks = []
        self.next_id = 1

    def update(self, frame_id, boxes, features, gt_ids=None):
        """I return one track ID for each detection.

        I pass gt_ids only to the logging hooks in DiagnosticMOTTracker.
        I match with boxes and features.
        """
        boxes = np.asarray(boxes).reshape(-1, 4)
        features = np.asarray(features)

        n = len(boxes)
        assigned = np.full(n, -1, dtype=int)
        old_tracks = self.tracks

        if old_tracks and n:
            previous_boxes = np.stack([t.box for t in old_tracks])
            previous_features = np.stack([t.feature for t in old_tracks])

            iou = compute_iou_matrix(previous_boxes, boxes)
            appearance = np.clip(previous_features @ features.T, -1, 1)

            if self.method == "iou":
                score = iou
                allowed = iou >= IOU_THRESHOLD
            elif self.method == "appearance":
                score = appearance
                allowed = appearance >= APPEARANCE_THRESHOLD
            else:
                app_norm = (appearance + 1) / 2
                score = self.alpha * app_norm + (1 - self.alpha) * iou
                allowed = (score >= FUSION_THRESHOLD) & (
                    (iou >= FUSION_MIN_IOU) | (appearance >= APPEARANCE_THRESHOLD)
                )

            self._log_candidates(frame_id, old_tracks, iou, appearance, score, allowed, gt_ids)

            # I give pairs below my threshold a very high cost and skip them after assignment.
            cost = -score.copy()
            cost[~allowed] = 1e6
            rows, cols = linear_sum_assignment(cost)

            for i, j in zip(rows, cols):
                accepted = bool(allowed[i, j])
                self._log_assignment(i, j, accepted)
                if not accepted:
                    continue

                track = old_tracks[i]
                track.box = boxes[j].copy()
                track.feature = features[j].copy()
                track.missed = 0
                track.last_seen = frame_id
                assigned[j] = track.track_id

        for track in old_tracks:
            if track.last_seen != frame_id:
                track.missed += 1

        self.tracks = [t for t in old_tracks if t.missed <= self.max_age]

        # I start a new track for each unmatched detection.
        for j in range(n):
            if assigned[j] != -1:
                continue

            self.tracks.append(
                Track(
                    track_id=self.next_id,
                    box=boxes[j].copy(),
                    feature=features[j].copy(),
                    missed=0,
                    last_seen=frame_id,
                )
            )
            assigned[j] = self.next_id
            self.next_id += 1

        return assigned

    # I fill in these logging hooks in DiagnosticMOTTracker.
    def _log_candidates(self, frame_id, tracks, iou, appearance, score, allowed, gt_ids):
        pass

    def _log_assignment(self, i, j, accepted):
        pass
