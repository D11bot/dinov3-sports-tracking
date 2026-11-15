"""I log the matching decisions from my IoU, appearance and fusion trackers."""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
from .association import compute_iou_matrix


@dataclass
class DiagnosticTrack:
    track_id: int
    box: np.ndarray
    feature: np.ndarray
    missed: int = 0
    last_seen: int = 0


class DiagnosticMOTTracker:

    def __init__(
        self,
        method="fusion",
        max_age=5,
        alpha=0.5
    ):
        self.method = method
        self.max_age = max_age
        self.alpha = alpha
        self.tracks = []
        self.next_id = 1
        self.logs = []

    def update(self, frame_id, boxes, features, gt_ids=None):

        boxes = np.asarray(boxes).reshape(-1, 4)
        features = np.asarray(features)

        n = len(boxes)
        assigned = np.full(n, -1, dtype=int)

        old_tracks = self.tracks

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

            appearance = np.clip(
                previous_features @ features.T,
                -1, 1
            )

            appearance_normalized = (appearance + 1) / 2

            fusion = (
                self.alpha * appearance_normalized
                + (1 - self.alpha) * iou
            )

            if self.method == "iou":
                score = iou
                allowed = iou >= 0.1

            elif self.method == "appearance":
                score = appearance
                allowed = appearance >= 0.78

            elif self.method == "fusion":
                score = fusion
                allowed = (
                    (score >= 0.55)
                    & ((iou >= 0.05) | (appearance >= 0.78))
                )
            else:
                raise ValueError("Unknown method")

            # I record each candidate pair before matching.
            log_positions = {}

            for i, track in enumerate(old_tracks):
                for j in range(n):

                    record = {
                        "Frame": frame_id,
                        "Method": self.method,
                        "Track_ID": track.track_id,
                        "Detection_Index": j,
                        "Detection_GT_ID": (
                            int(gt_ids[j])
                            if gt_ids is not None else None
                        ),
                        "Track_Missed": track.missed,
                        "Appearance": float(appearance[i, j]),
                        "IoU": float(iou[i, j]),
                        "Fusion": float(fusion[i, j]),
                        "Score": float(score[i, j]),
                        "Allowed": bool(allowed[i, j]),
                        "Selected": False,
                        "Accepted": False
                    }

                    log_positions[(i, j)] = len(self.logs)
                    self.logs.append(record)

            cost = -score.copy()
            cost[~allowed] = 1e6

            rows, cols = linear_sum_assignment(cost)

            for i, j in zip(rows, cols):

                pos = log_positions[(i, j)]
                self.logs[pos]["Selected"] = True

                if not allowed[i, j]:
                    continue

                self.logs[pos]["Accepted"] = True

                track = old_tracks[i]

                track.box = boxes[j].copy()
                track.feature = features[j].copy()
                track.missed = 0
                track.last_seen = frame_id

                assigned[j] = track.track_id

        for track in old_tracks:
            if track.last_seen != frame_id:
                track.missed += 1

        self.tracks = [
            t for t in old_tracks
            if t.missed <= self.max_age
        ]

        for j in range(n):
            if assigned[j] != -1:
                continue

            self.tracks.append(
                DiagnosticTrack(
                    track_id=self.next_id,
                    box=boxes[j].copy(),
                    feature=features[j].copy(),
                    missed=0,
                    last_seen=frame_id
                )
            )

            assigned[j] = self.next_id
            self.next_id += 1

        return assigned
