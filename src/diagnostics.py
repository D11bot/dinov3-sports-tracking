"""I record candidate scores and matching decisions for my trackers.

I use the GT IDs to check the decisions after matching. The trackers here
assign the same IDs as KalmanMOTTracker and SimpleMOTTracker.
"""

import numpy as np

from .kalman_tracker import KalmanMOTTracker
from .simple_tracker import SimpleMOTTracker


def _gt_at(gt_ids, j):
    return int(gt_ids[j]) if gt_ids is not None else None


class DiagnosticKalmanTracker(KalmanMOTTracker):
    """I log candidate pairs, wrong associations and new tracks for the Kalman trackers."""

    def __init__(
        self, method="kf_dinov3", max_age=10, app_weight=0.45, gate_d2=9.21, r_variance=16.0
    ):
        super().__init__(
            method=method,
            max_age=max_age,
            app_weight=app_weight,
            gate_d2=gate_d2,
            r_variance=r_variance,
        )
        self.candidate_logs = []
        self.event_logs = []
        self.last_gt = {}  # I keep the GT ID that each track was last matched to
        self._log_index = {}

    def _log_candidates(self, frame_id, tracks, d2, motion, appearance, score, allowed, gt_ids):
        self._log_index = {}
        for i, track in enumerate(tracks):
            previous_gt = self.last_gt.get(track.track_id)
            for j in range(d2.shape[1]):
                new_gt = _gt_at(gt_ids, j)
                self._log_index[i, j] = len(self.candidate_logs)
                self.candidate_logs.append(
                    {
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
                        ),
                    }
                )

    def _log_assignment(self, frame_id, track, i, j, accepted, d2, appearance, gt_ids):
        record = self.candidate_logs[self._log_index[i, j]]
        record["Selected"] = True
        if not accepted:
            return

        record["Accepted"] = True
        previous_gt = self.last_gt.get(track.track_id)
        new_gt = _gt_at(gt_ids, j)

        if previous_gt is not None and new_gt is not None and previous_gt != new_gt:
            self.event_logs.append(
                {
                    "Frame": frame_id,
                    "Method": self.method,
                    "Event": "WRONG_ASSOCIATION",
                    "Track_ID": track.track_id,
                    "Previous_GT": previous_gt,
                    "Current_GT": new_gt,
                    "D2": float(d2[i, j]),
                    "Appearance": float(appearance[i, j]),
                }
            )

        if new_gt is not None:
            self.last_gt[track.track_id] = new_gt

    def _log_new_track(self, frame_id, track_id, j, gt_ids):
        current_gt = _gt_at(gt_ids, j)
        self.event_logs.append(
            {
                "Frame": frame_id,
                "Method": self.method,
                "Event": "NEW_TRACK",
                "Track_ID": track_id,
                "Previous_GT": None,
                "Current_GT": current_gt,
                "D2": np.nan,
                "Appearance": np.nan,
            }
        )
        if current_gt is not None:
            self.last_gt[track_id] = current_gt


class DiagnosticMOTTracker(SimpleMOTTracker):
    """I log the scores of every candidate pair for the IoU, appearance and fusion trackers."""

    def __init__(self, method="fusion", max_age=5, alpha=0.5):
        super().__init__(method=method, max_age=max_age, alpha=alpha)
        self.logs = []
        self._log_index = {}

    def _log_candidates(self, frame_id, tracks, iou, appearance, score, allowed, gt_ids):
        fusion = self.alpha * ((appearance + 1) / 2) + (1 - self.alpha) * iou
        self._log_index = {}
        for i, track in enumerate(tracks):
            for j in range(iou.shape[1]):
                self._log_index[i, j] = len(self.logs)
                self.logs.append(
                    {
                        "Frame": frame_id,
                        "Method": self.method,
                        "Track_ID": track.track_id,
                        "Detection_Index": j,
                        "Detection_GT_ID": _gt_at(gt_ids, j),
                        "Track_Missed": track.missed,
                        "Appearance": float(appearance[i, j]),
                        "IoU": float(iou[i, j]),
                        "Fusion": float(fusion[i, j]),
                        "Score": float(score[i, j]),
                        "Allowed": bool(allowed[i, j]),
                        "Selected": False,
                        "Accepted": False,
                    }
                )

    def _log_assignment(self, i, j, accepted):
        record = self.logs[self._log_index[i, j]]
        record["Selected"] = True
        if accepted:
            record["Accepted"] = True
