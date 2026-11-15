"""I compare Kalman motion matching with Kalman plus DINOv3.
I also test different R values using the same matching rules.
"""
import numpy as np
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
from .kalman import CVKalman, CalibratedCVKalman

from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment

@dataclass
class KFTrack:
    track_id: int
    kf: CVKalman
    feature: np.ndarray
    missed: int = 0
    last_seen: int = 0


class KalmanMOTTracker:

    def __init__(
        self,
        method="kf_motion",
        max_age=10,
        app_weight=0.45,
        gate_d2=9.21
    ):
        assert method in ("kf_motion", "kf_dinov3")

        self.method = method
        self.max_age = max_age
        self.app_weight = app_weight
        self.gate_d2 = gate_d2

        self.tracks = []
        self.next_id = 1

    def update(self, frame_id, boxes, features):

        boxes = np.asarray(
            boxes, dtype=float
        ).reshape(-1, 4)

        features = np.asarray(
            features, dtype=float
        )

        n = len(boxes)
        assigned = np.full(n, -1, dtype=int)
        old_tracks = self.tracks

        # I predict each track position before matching.
        for track in old_tracks:
            track.kf.predict()

        if old_tracks and n > 0:

            d2 = np.array([
                [
                    track.kf.innovation(box)[2]
                    for box in boxes
                ]
                for track in old_tracks
            ])

            motion_score = np.exp(
                -0.5 * np.minimum(d2, 100.)
            )

            # I use the same motion gate for both methods.
            allowed = d2 <= self.gate_d2

            if self.method == "kf_motion":
                score = motion_score

            else:
                previous_features = np.stack([
                    t.feature for t in old_tracks
                ])

                appearance = np.clip(
                    previous_features @ features.T,
                    -1.0, 1.0
                )

                app_norm = (appearance + 1.0) / 2.0

                score = (
                    (1 - self.app_weight) * motion_score
                    + self.app_weight * app_norm
                )

            cost = -score.copy()
            cost[~allowed] = 1e6

            rows, cols = linear_sum_assignment(cost)

            for i, j in zip(rows, cols):

                if not allowed[i, j]:
                    continue

                track = old_tracks[i]

                track.kf.update(boxes[j])
                track.feature = features[j].copy()
                track.missed = 0
                track.last_seen = frame_id

                assigned[j] = track.track_id

        # I count missing frames for tracks I did not match.
        for track in old_tracks:
            if track.last_seen != frame_id:
                track.missed += 1

        self.tracks = [
            t for t in old_tracks
            if t.missed <= self.max_age
        ]

        # I create tracks for detections without a match.
        for j in range(n):

            if assigned[j] != -1:
                continue

            new_track = KFTrack(
                track_id=self.next_id,
                kf=CVKalman(boxes[j]),
                feature=features[j].copy(),
                missed=0,
                last_seen=frame_id
            )

            self.tracks.append(new_track)
            assigned[j] = self.next_id
            self.next_id += 1

        return assigned

import numpy as np
from scipy.optimize import linear_sum_assignment

class CalibratedKalmanMOTTracker(KalmanMOTTracker):

    def __init__(
        self,
        method="kf_motion",
        max_age=10,
        app_weight=0.45,
        gate_d2=9.21,
        r_variance=16.0
    ):
        super().__init__(
            method=method,
            max_age=max_age,
            app_weight=app_weight,
            gate_d2=gate_d2
        )

        self.r_variance = float(r_variance)
        self.new_tracks_created = 0

    def update(self, frame_id, boxes, features):

        boxes = np.asarray(
            boxes, dtype=float
        ).reshape(-1, 4)

        features = np.asarray(features, dtype=float)

        n = len(boxes)
        assigned = np.full(n, -1, dtype=int)

        old_tracks = self.tracks

        # I predict positions with Kalman.
        for track in old_tracks:
            track.kf.predict()

        # I match predictions to the current detections.
        if old_tracks and n:

            d2 = np.array([
                [
                    t.kf.innovation(box)[2]
                    for box in boxes
                ]
                for t in old_tracks
            ])

            motion_score = np.exp(
                -0.5 * np.minimum(d2, 100.0)
            )

            allowed = d2 <= self.gate_d2

            if self.method == "kf_motion":
                score = motion_score

            elif self.method == "kf_dinov3":

                previous_features = np.stack([
                    t.feature for t in old_tracks
                ])

                appearance = np.clip(
                    previous_features @ features.T,
                    -1.0, 1.0
                )

                app_norm = (appearance + 1.0) / 2.0

                score = (
                    (1 - self.app_weight) * motion_score
                    + self.app_weight * app_norm
                )

            else:
                raise ValueError("Unknown tracking method")

            cost = -score.copy()
            cost[~allowed] = 1e6

            rows, cols = linear_sum_assignment(cost)

            for i, j in zip(rows, cols):

                if not allowed[i, j]:
                    continue

                track = old_tracks[i]

                track.kf.update(boxes[j])
                track.feature = features[j].copy()
                track.missed = 0
                track.last_seen = frame_id

                assigned[j] = track.track_id

        # I update the track states.
        for track in old_tracks:
            if track.last_seen != frame_id:
                track.missed += 1

        self.tracks = [
            t for t in old_tracks
            if t.missed <= self.max_age
        ]

        # I initialize tracks for new detections.
        for j in range(n):

            if assigned[j] != -1:
                continue

            self.tracks.append(
                KFTrack(
                    track_id=self.next_id,
                    kf=CalibratedCVKalman(
                        boxes[j],
                        r_variance=self.r_variance
                    ),
                    feature=features[j].copy(),
                    missed=0,
                    last_seen=frame_id
                )
            )

            assigned[j] = self.next_id
            self.next_id += 1
            self.new_tracks_created += 1

        return assigned

def run_tracker(frame_data, method, r_variance=16., max_age=10, app_weight=0.45, gate_d2=9.21):
    tracker = CalibratedKalmanMOTTracker(method=method, r_variance=r_variance,
        max_age=max_age, app_weight=app_weight, gate_d2=gate_d2)
    outputs = {}
    previous = None
    for frame_id in sorted(frame_data):
        if previous is not None and frame_id != previous + 1:
            raise ValueError("Frame IDs must be consecutive (one Kalman predict step per frame)")
        data = frame_data[frame_id]
        ids = tracker.update(frame_id, data['boxes'], data['features'])
        outputs[frame_id] = {'boxes': data['boxes'].copy(),
                             'pred_ids': ids.copy(), 'gt_ids': data['gt_ids'].copy()}
        previous = frame_id
    return outputs, tracker
