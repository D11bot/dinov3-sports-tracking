"""I use small synthetic examples to check my tracking code."""

import numpy as np
import pytest
from PIL import Image

from src.association import bbox_iou
from src.corruption import create_detection_gap
from src.data import filter_target_ids, load_feature_cache, save_feature_cache
from src.diagnostics import DiagnosticKalmanTracker, DiagnosticMOTTracker
from src.evaluation import evaluate_known_correspondence, evaluate_tracking
from src.features import crop_window
from src.kalman import CVKalman
from src.kalman_tracker import KalmanMOTTracker, run_tracker
from src.simple_tracker import SimpleMOTTracker


def fixture_frames():
    result = {}
    for t in range(1, 9):
        result[t] = {
            "boxes": np.array([[10 + t, 12, 20, 40], [100 + t, 25, 20, 40]], dtype=float),
            "features": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
            "gt_ids": np.array([7, 9]),
        }
    return result


def crossing_frames(seed=0, frames=30):
    """I move four players towards each other with noisy boxes, so the trackers have real choices."""
    rng = np.random.default_rng(seed)
    starts = np.array([[50.0, 100.0], [250.0, 110.0], [60.0, 300.0], [240.0, 290.0]])
    velocity = np.array([[4.0, 1.0], [-4.0, 0.5], [3.5, -1.0], [-3.5, -0.5]])
    features = rng.normal(size=(4, 8))
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    result = {}
    for t in range(1, frames + 1):
        centers = starts + velocity * t + rng.normal(0, 3, size=(4, 2))
        boxes = np.column_stack(
            [centers[:, 0] - 15, centers[:, 1] - 40, np.full(4, 30.0), np.full(4, 80.0)]
        )
        noisy = features + rng.normal(0, 0.2, size=features.shape)
        noisy /= np.linalg.norm(noisy, axis=1, keepdims=True)
        result[t] = {"boxes": boxes, "features": noisy, "gt_ids": np.array([1, 2, 3, 4])}
    return result


def test_iou_and_single_frame_assignment():
    assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert bbox_iou([0, 0, 10, 10], [100, 100, 10, 10]) == 0
    tracker = SimpleMOTTracker(method="fusion")
    out = tracker.update(1, fixture_frames()[1]["boxes"], fixture_frames()[1]["features"])
    assert len(set(out)) == 2


def test_kalman_trackers_keep_identities():
    data = fixture_frames()
    for method in ("kf_motion", "kf_dinov3"):
        outputs, _ = run_tracker(data, method)
        assert evaluate_tracking(outputs)["Identity Changes"] == 0
        assert evaluate_known_correspondence(outputs, data)["Contaminated_Tracks"] == 0


def test_larger_r_trusts_the_measurement_less():
    small = CVKalman([0, 0, 10, 10], r_variance=16)
    large = CVKalman([0, 0, 10, 10], r_variance=256)
    for kf in (small, large):
        kf.predict()
        kf.update([20, 0, 10, 10])
    assert 0 < large.x[0] < small.x[0] < 25


def test_far_detection_starts_a_new_track():
    tracker = KalmanMOTTracker(method="kf_motion")
    first = tracker.update(1, [[10, 10, 20, 40]], [[1.0, 0.0]])
    second = tracker.update(2, [[600, 10, 20, 40]], [[1.0, 0.0]])
    assert first[0] != second[0]


def test_logging_does_not_change_matching():
    data = crossing_frames()
    for method in ("kf_motion", "kf_dinov3"):
        plain = KalmanMOTTracker(method=method, r_variance=64)
        logged = DiagnosticKalmanTracker(method=method, r_variance=64)
        for frame, d in sorted(data.items()):
            a = plain.update(frame, d["boxes"], d["features"])
            b = logged.update(frame, d["boxes"], d["features"], gt_ids=d["gt_ids"])
            np.testing.assert_array_equal(a, b)
        assert logged.candidate_logs
    for method in ("iou", "appearance", "fusion"):
        plain = SimpleMOTTracker(method=method, max_age=10)
        logged = DiagnosticMOTTracker(method=method, max_age=10)
        for frame, d in sorted(data.items()):
            a = plain.update(frame, d["boxes"], d["features"])
            b = logged.update(frame, d["boxes"], d["features"], gt_ids=d["gt_ids"])
            np.testing.assert_array_equal(a, b)


def test_crop_window_stays_inside_the_image():
    image = Image.new("RGB", (100, 50))
    assert crop_window(image, [-5.5, 10.2, 30, 20]) == (0, 10, 24, 30)
    assert crop_window(image, [90.7, 40.1, 30, 30]) == (90, 40, 100, 50)
    with pytest.raises(ValueError):
        crop_window(image, [120, 10, 5, 5])


def test_feature_cache_and_filter(tmp_path):
    data = fixture_frames()
    path = tmp_path / "features.npz"
    save_feature_cache(path, data)
    loaded = load_feature_cache(path)
    assert list(loaded) == list(data)
    filtered = filter_target_ids(loaded, [9])
    assert len(filtered[1]["boxes"]) == 1
    assert loaded[1]["boxes"].shape == (2, 4)


def test_gap_does_not_modify_reference():
    data = fixture_frames()
    corrupted = create_detection_gap(data, 7, 3, 2)
    assert len(corrupted[3]["boxes"]) == 1
    assert len(data[3]["boxes"]) == 2
    assert len(corrupted[5]["boxes"]) == 2


def test_frames_must_be_consecutive():
    data = fixture_frames()
    del data[3]
    with pytest.raises(ValueError, match="consecutive"):
        run_tracker(data, "kf_motion")
