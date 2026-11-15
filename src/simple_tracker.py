"""I compare three simple tracking methods with fixed matching thresholds."""
import numpy as np
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
from .association import compute_iou_matrix

from dataclasses import dataclass

@dataclass
class Track:
    track_id: int
    box: np.ndarray
    feature: np.ndarray
    missed: int = 0
    last_seen: int = 0

from scipy.optimize import linear_sum_assignment

class SimpleMOTTracker:
    def __init__(self, method="fusion", max_age=5, alpha=0.5):
        self.method = method
        self.max_age = max_age
        self.alpha = alpha
        self.tracks = []
        self.next_id = 1

    def update(self, frame_id, boxes, features):
        boxes = np.asarray(boxes)
        features = np.asarray(features)

        n = len(boxes)
        old_tracks = self.tracks

        # I keep the assigned track ID for each detection.
        assigned = np.full(n, -1, dtype=int)

        if old_tracks and n:
            previous_boxes = np.stack(
                [t.box for t in old_tracks]
            )
            previous_features = np.stack(
                [t.feature for t in old_tracks]
            )

            iou = compute_iou_matrix(
                previous_boxes, boxes
            )

            appearance = previous_features @ features.T
            appearance = np.clip(appearance, -1, 1)

            if self.method == "iou":
                score = iou
                allowed = iou >= 0.1

            elif self.method == "appearance":
                score = appearance
                allowed = appearance >= 0.78

            elif self.method == "fusion":
                app_norm = (appearance + 1) / 2
                score = (
                    self.alpha * app_norm
                    + (1-self.alpha) * iou
                )
                allowed = (
                    (score >= 0.55)
                    & ((iou >= 0.05) | (appearance >= 0.78))
                )

            else:
                raise ValueError("Unknown method")

            # I exclude pairs that do not pass my matching threshold.
            cost = -score.copy()
            cost[~allowed] = 1e6

            rows, cols = linear_sum_assignment(cost)

            for i, j in zip(rows, cols):
                if not allowed[i, j]:
                    continue

                track = old_tracks[i]

                track.box = boxes[j].copy()
                track.feature = features[j].copy()
                track.missed = 0
                track.last_seen = frame_id

                assigned[j] = track.track_id

        # I count frames since I last saw each track.
        for track in old_tracks:
            if track.last_seen != frame_id:
                track.missed += 1

        # I finish tracks after too many missing frames.
        self.tracks = [
            t for t in old_tracks
            if t.missed <= self.max_age
        ]

        # I start a new track for each unmatched detection.
        for j in range(n):
            if assigned[j] != -1:
                continue

            track = Track(
                track_id=self.next_id,
                box=boxes[j].copy(),
                feature=features[j].copy(),
                missed=0,
                last_seen=frame_id
            )

            self.tracks.append(track)
            assigned[j] = self.next_id
            self.next_id += 1

        return assigned
