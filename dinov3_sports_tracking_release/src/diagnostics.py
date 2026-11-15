"""I record Kalman candidate scores and matching decisions.
I use the GT IDs to check the decisions after the tracker runs.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from .kalman import CVKalman
from .kalman_tracker import KalmanMOTTracker, KFTrack


class DiagnosticKalmanTracker(KalmanMOTTracker):

    def __init__(self, method="kf_dinov3",
                 max_age=10, app_weight=0.45,
                 gate_d2=9.21):
        super().__init__(
            method=method,
            max_age=max_age,
            app_weight=app_weight,
            gate_d2=gate_d2
        )
        self.candidate_logs = []
        self.event_logs = []

        # I keep GT labels for checking results after matching.
        self.last_gt = {}

    def update(self, frame_id, boxes, features, gt_ids=None):
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        features = np.asarray(features, dtype=float)
        n = len(boxes)
        assigned = np.full(n, -1, dtype=int)
        old_tracks = self.tracks

        for track in old_tracks:
            track.kf.predict()

        if old_tracks and n:
            d2 = np.array([
                [t.kf.innovation(box)[2] for box in boxes]
                for t in old_tracks
            ])

            motion = np.exp(-0.5 * np.minimum(d2, 100.0))
            allowed = d2 <= self.gate_d2

            old_features = np.stack(
                [t.feature for t in old_tracks]
            )
            appearance = np.clip(
                old_features @ features.T, -1, 1
            )

            app_norm = (appearance + 1) / 2

            if self.method == "kf_motion":
                score = motion
            else:
                score = (
                    (1-self.app_weight)*motion
                    + self.app_weight*app_norm
                )

            log_index = {}

            for i, track in enumerate(old_tracks):
                for j in range(n):
                    new_gt = (
                        int(gt_ids[j])
                        if gt_ids is not None else None
                    )
                    previous_gt = self.last_gt.get(track.track_id)

                    rec = {
                        "Frame": frame_id,
                        "Method": self.method,
                        "Track_ID": track.track_id,
                        "Previous_GT": previous_gt,
                        "Detection_GT": new_gt,
                        "Detection_Index": j,
                        "Missed": track.missed,
                        "D2": float(d2[i, j]),
                        "Motion": float(motion[i, j]),
                        "Appearance": float(appearance[i, j]),
                        "Score": float(score[i, j]),
                        "Gate_Passed": bool(allowed[i, j]),
                        "Selected": False,
                        "Accepted": False,
                        "Same_GT_Candidate": (
                            previous_gt is not None
                            and new_gt is not None
                            and previous_gt >= 0
                            and previous_gt == new_gt
                        )
                    }

                    log_index[i, j] = len(self.candidate_logs)
                    self.candidate_logs.append(rec)

            cost = -score.copy()
            cost[~allowed] = 1e6
            rows, cols = linear_sum_assignment(cost)

            for i, j in zip(rows, cols):
                rec = self.candidate_logs[log_index[i, j]]
                rec["Selected"] = True

                if not allowed[i, j]:
                    continue

                rec["Accepted"] = True
                track = old_tracks[i]

                previous_gt = self.last_gt.get(track.track_id)
                new_gt = (
                    int(gt_ids[j])
                    if gt_ids is not None else None
                )

                if (
                    previous_gt is not None
                    and new_gt is not None
                    and previous_gt != new_gt
                ):
                    self.event_logs.append({
                        "Frame": frame_id,
                        "Method": self.method,
                        "Event": "WRONG_ASSOCIATION",
                        "Track_ID": track.track_id,
                        "Previous_GT": previous_gt,
                        "Current_GT": new_gt,
                        "D2": float(d2[i, j]),
                        "Appearance": float(appearance[i, j])
                    })

                track.kf.update(boxes[j])
                track.feature = features[j].copy()
                track.missed = 0
                track.last_seen = frame_id
                assigned[j] = track.track_id

                if new_gt is not None:
                    self.last_gt[track.track_id] = new_gt

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

            new_id = self.next_id
            self.tracks.append(
                KFTrack(
                    track_id=new_id,
                    kf=CVKalman(boxes[j]),
                    feature=features[j].copy(),
                    missed=0,
                    last_seen=frame_id
                )
            )

            current_gt = (
                int(gt_ids[j])
                if gt_ids is not None else None
            )

            self.event_logs.append({
                "Frame": frame_id,
                "Method": self.method,
                "Event": "NEW_TRACK",
                "Track_ID": new_id,
                "Previous_GT": None,
                "Current_GT": current_gt,
                "D2": np.nan,
                "Appearance": np.nan
            })

            if current_gt is not None:
                self.last_gt[new_id] = current_gt

            assigned[j] = new_id
            self.next_id += 1

        return assigned
