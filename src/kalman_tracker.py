"""I compare Kalman motion matching (kf_motion) with Kalman plus DINOv3 (kf_dinov3).
I also test different R values with the same matching rules.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman import CVKalman

METHODS = ("kf_motion", "kf_dinov3")


@dataclass
class KFTrack:
    track_id: int
    kf: CVKalman
    feature: np.ndarray
    missed: int = 0
    last_seen: int = 0


class KalmanMOTTracker:
    """I predict each track with a Kalman filter, keep candidates that pass the
    Mahalanobis gate and assign them with the Hungarian algorithm.

    I use the same motion gate (d2 <= gate_d2) for both methods. In kf_dinov3 I add
    an appearance term to the score, so appearance chooses among the candidates
    that passed the motion gate.

    I set the measurement noise with r_variance (px^2) for every new track.
    """

    def __init__(
        self,
        method="kf_motion",
        max_age=10,
        app_weight=0.45,
        gate_d2=9.21,
        r_variance=16.0,
    ):
        if method not in METHODS:
            raise ValueError(f"Unknown tracking method: {method}")

        self.method = method
        self.max_age = max_age
        self.app_weight = app_weight
        self.gate_d2 = gate_d2
        self.r_variance = float(r_variance)

        self.tracks = []
        self.next_id = 1

    def update(self, frame_id, boxes, features, gt_ids=None):
        """I return one track ID for each detection.

        I pass gt_ids only to the logging hooks in DiagnosticKalmanTracker.
        I match with boxes and features.
        """
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        features = np.asarray(features, dtype=float)

        n = len(boxes)
        assigned = np.full(n, -1, dtype=int)
        old_tracks = self.tracks

        for track in old_tracks:
            track.kf.predict()

        if old_tracks and n:
            d2 = np.array([[t.kf.innovation(box)[2] for box in boxes] for t in old_tracks])
            motion_score = np.exp(-0.5 * np.minimum(d2, 100.0))
            allowed = d2 <= self.gate_d2

            previous_features = np.stack([t.feature for t in old_tracks])
            appearance = np.clip(previous_features @ features.T, -1.0, 1.0)

            if self.method == "kf_motion":
                score = motion_score
            else:
                app_norm = (appearance + 1.0) / 2.0
                score = (1 - self.app_weight) * motion_score + self.app_weight * app_norm

            self._log_candidates(
                frame_id, old_tracks, d2, motion_score, appearance, score, allowed, gt_ids
            )

            cost = -score.copy()
            cost[~allowed] = 1e6
            rows, cols = linear_sum_assignment(cost)

            for i, j in zip(rows, cols):
                accepted = bool(allowed[i, j])
                self._log_assignment(
                    frame_id, old_tracks[i], i, j, accepted, d2, appearance, gt_ids
                )
                if not accepted:
                    continue

                track = old_tracks[i]
                track.kf.update(boxes[j])
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

            track_id = self.next_id
            self.tracks.append(
                KFTrack(
                    track_id=track_id,
                    kf=CVKalman(boxes[j], r_variance=self.r_variance),
                    feature=features[j].copy(),
                    missed=0,
                    last_seen=frame_id,
                )
            )
            self._log_new_track(frame_id, track_id, j, gt_ids)
            assigned[j] = track_id
            self.next_id += 1

        return assigned

    # I fill in these logging hooks in DiagnosticKalmanTracker.
    def _log_candidates(self, frame_id, tracks, d2, motion, appearance, score, allowed, gt_ids):
        pass

    def _log_assignment(self, frame_id, track, i, j, accepted, d2, appearance, gt_ids):
        pass

    def _log_new_track(self, frame_id, track_id, j, gt_ids):
        pass


def run_tracker(frame_data, method, r_variance=16.0, max_age=10, app_weight=0.45, gate_d2=9.21):
    """I run a Kalman tracker over consecutive frames and return the outputs per frame."""
    tracker = KalmanMOTTracker(
        method=method,
        max_age=max_age,
        app_weight=app_weight,
        gate_d2=gate_d2,
        r_variance=r_variance,
    )
    outputs = {}
    previous = None

    for frame_id in sorted(frame_data):
        if previous is not None and frame_id != previous + 1:
            raise ValueError("Frame IDs must be consecutive (one Kalman predict step per frame)")

        data = frame_data[frame_id]
        ids = tracker.update(frame_id, data["boxes"], data["features"])
        outputs[frame_id] = {
            "boxes": data["boxes"].copy(),
            "pred_ids": ids.copy(),
            "gt_ids": data["gt_ids"].copy(),
        }
        previous = frame_id

    return outputs, tracker
