"""I run my DINOv3 football tracking experiments on SoccerNet sequences."""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from src.corruption import corrupt_detection_data, create_detection_gap
from src.data import (
    extract_zip_sequence,
    filter_target_ids,
    load_feature_cache,
    save_feature_cache,
)
from src.diagnostics import DiagnosticKalmanTracker, DiagnosticMOTTracker
from src.evaluation import (
    audit_track_identity,
    evaluate_gap_recovery,
    evaluate_known_correspondence,
    evaluate_quality_tracking,
    evaluate_tracking,
)
from src.features import DEFAULT_MODEL, extract_sequence, load_dinov3
from src.kalman_tracker import run_tracker
from src.simple_tracker import SimpleMOTTracker

ALL_METHODS = ["iou", "appearance", "fusion", "kf_motion", "kf_dinov3"]
EXPERIMENTS = ["baseline", "gap", "quality", "calibration", "diagnostics", "gap_logs", "all"]
# I run everything with "all" except gap_logs, which looks at one gap case.
NEEDS_MODEL = ["quality", "calibration", "diagnostics", "all"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-dir", required=True, help="I read img1/ and gt/gt.txt from this folder"
    )
    parser.add_argument(
        "--train-zip", help="I extract the sequence from this SoccerNet training ZIP"
    )
    parser.add_argument(
        "--sequence", help="The sequence I extract from the ZIP, such as train/SNMOT-061"
    )
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument(
        "--exclude-ids",
        type=int,
        nargs="*",
        default=[],
        help="I exclude these GT IDs after checking which objects are not players, such as the ball",
    )
    parser.add_argument("--cache", help="I save and reuse my DINOv3 features in this NPZ file")
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jitter", type=float, default=0.15)
    parser.add_argument("--max-age", type=int, default=10)
    parser.add_argument("--r-values", type=float, nargs="+", default=[16, 64, 144, 256])
    parser.add_argument("--gap-targets", type=int, nargs="+", default=[1, 3, 7, 10, 12, 15])
    parser.add_argument("--gap-starts", type=int, nargs="+", default=[35, 60])
    parser.add_argument("--gap-lengths", type=int, nargs="+", default=[2, 3, 5, 8])
    parser.add_argument("--log-target", type=int, default=7, help="gap_logs: the GT ID I remove")
    parser.add_argument(
        "--log-start", type=int, default=40, help="gap_logs: the first frame I remove"
    )
    parser.add_argument(
        "--log-length", type=int, default=5, help="gap_logs: how many frames I remove"
    )
    return parser.parse_args()


def simple_run(data, method, max_age=10):
    tracker = SimpleMOTTracker(method=method, max_age=max_age, alpha=0.5)
    output = {}
    for frame, d in sorted(data.items()):
        pred = tracker.update(frame, d["boxes"], d["features"])
        output[frame] = {
            "gt_ids": d["gt_ids"].copy(),
            "pred_ids": pred.copy(),
            "boxes": d["boxes"].copy(),
        }
    return output, tracker


def run_method(data, method, max_age, r_variance=16.0):
    if method.startswith("kf_"):
        return run_tracker(data, method, max_age=max_age, r_variance=r_variance)
    return simple_run(data, method, max_age=max_age)


def write_csv(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print("Saved", path)


def baseline(data, out, max_age):
    records = []
    for method in ALL_METHODS:
        output, tracker = run_method(data, method, max_age)
        stats = evaluate_tracking(output)
        stats.update(audit_track_identity(output))
        records.append({"Method": method, "Total_Created": tracker.next_id - 1, **stats})
    write_csv(out / "baseline_tracking.csv", pd.DataFrame(records))


def gap(data, out, args):
    cases = []
    for target in args.gap_targets:
        for gap_length in args.gap_lengths:
            for start in args.gap_starts:
                end = start + gap_length
                # I only use cases where the target is visible from one frame before to one frame after the gap.
                if not all(
                    frame in data and target in data[frame]["gt_ids"]
                    for frame in range(start - 1, end + 1)
                ):
                    continue
                modified = create_detection_gap(data, target, start, gap_length)
                for method in ALL_METHODS:
                    outputs, _ = run_method(modified, method, args.max_age)
                    metric = evaluate_gap_recovery(outputs, target, start, gap_length)
                    cases.append(
                        {
                            "GT_ID": target,
                            "Start_Frame": start,
                            "Gap_Length": gap_length,
                            "Method": method,
                            **metric,
                        }
                    )
    detail = pd.DataFrame(cases)
    write_csv(out / "gap_cases.csv", detail)
    if not detail.empty:
        group = detail.groupby(["Gap_Length", "Method"], as_index=False).agg(
            Tests=("Recovered", "size"),
            Successful=("Recovered", "sum"),
            Recovery_Rate=("Recovered", "mean"),
            Hijacks=("Hijacked", "sum"),
        )
        write_csv(out / "gap_summary.csv", group)


def corrupt(data, args, model_bundle, **settings):
    processor, model, device = model_bundle
    return corrupt_detection_data(
        data,
        seed=args.seed,
        image_dir=Path(args.sequence_dir) / "img1",
        processor=processor,
        model=model,
        device=device,
        **settings,
    )


def quality(data, out, args, model_bundle):
    conditions = {
        "clean": (0, 0, 0),
        "missing_20pct": (0.20, 0, 0),
        "box_jitter_15pct": (0, 0.15, 0),
        "false_positive_1": (0, 0, 1),
    }
    records = []
    for label, (missing, jitter, fp) in conditions.items():
        if label == "clean":
            noisy = data
        else:
            noisy = corrupt(
                data, args, model_bundle, missing_rate=missing, box_noise=jitter, fp_per_frame=fp
            )
        for method in ["kf_motion", "kf_dinov3"]:
            outputs, tracker = run_tracker(noisy, method, max_age=args.max_age)
            # I count Identity_Changes with the known source identity of each box.
            metrics = evaluate_known_correspondence(outputs, noisy)
            # I also keep ID_Changes_Diagnostic, which only counts boxes matched to GT at IoU >= 0.5.
            stats = evaluate_quality_tracking(outputs, data)
            detection = [
                "TP",
                "FP",
                "FN",
                "Detection_Recall",
                "Detection_Precision",
                "Mean_Matched_IoU",
                "ID_Changes_Diagnostic",
            ]
            records.append(
                {
                    "Condition": label,
                    "Method": method,
                    "Tracks_Created": tracker.next_id - 1,
                    **metrics,
                    **{k: v for k, v in stats.items() if k in detection},
                }
            )
    write_csv(out / "quality_stress_test.csv", pd.DataFrame(records))


def calibration(data, out, args, model_bundle):
    noisy = corrupt(data, args, model_bundle, box_noise=args.jitter)
    records = []
    for r in args.r_values:
        for method in ["kf_motion", "kf_dinov3"]:
            outputs, tracker = run_tracker(noisy, method, max_age=args.max_age, r_variance=r)
            records.append(
                {
                    "R_Variance": r,
                    "Method": method,
                    "Total_Tracks": tracker.next_id - 1,
                    **evaluate_known_correspondence(outputs, noisy),
                }
            )
    write_csv(out / "r_sensitivity.csv", pd.DataFrame(records))


def diagnostics(data, out, args, model_bundle):
    noisy = corrupt(data, args, model_bundle, box_noise=args.jitter)
    for method in ["kf_motion", "kf_dinov3"]:
        tracker = DiagnosticKalmanTracker(method=method, max_age=args.max_age)
        outputs = {}
        for frame, d in sorted(noisy.items()):
            ids = tracker.update(frame, d["boxes"], d["features"], gt_ids=d["gt_ids"])
            outputs[frame] = {"gt_ids": d["gt_ids"], "pred_ids": ids, "boxes": d["boxes"]}
        write_csv(out / f"candidate_logs_{method}.csv", pd.DataFrame(tracker.candidate_logs))
        write_csv(out / f"event_logs_{method}.csv", pd.DataFrame(tracker.event_logs))
        print(method, "diagnostic metrics:", evaluate_known_correspondence(outputs, noisy))


def gap_logs(data, out, args):
    """I log the candidate scores of the IoU, appearance and fusion trackers for one gap case."""
    modified = create_detection_gap(data, args.log_target, args.log_start, args.log_length)
    for method in ["iou", "appearance", "fusion"]:
        tracker = DiagnosticMOTTracker(method=method, max_age=args.max_age, alpha=0.5)
        for frame, d in sorted(modified.items()):
            tracker.update(frame, d["boxes"], d["features"], gt_ids=d["gt_ids"])
        write_csv(out / f"gap_candidate_logs_{method}.csv", pd.DataFrame(tracker.logs))


def main():
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    sequence_dir = Path(args.sequence_dir)

    if args.train_zip:
        if not args.sequence:
            raise ValueError("--train-zip requires --sequence")
        extract_zip_sequence(args.train_zip, args.sequence, sequence_dir, args.frames)

    model_bundle = (None, None, None)
    if args.cache and Path(args.cache).exists():
        data = load_feature_cache(args.cache)
        print("Loaded features from", args.cache)
    else:
        model_bundle = load_dinov3(args.model_id, token=os.getenv("HF_TOKEN"))
        data = extract_sequence(sequence_dir, *model_bundle, args.frames)
        if args.cache:
            save_feature_cache(args.cache, data)

    data = filter_target_ids(data, args.exclude_ids)
    print("Frames:", len(data), "exclusions:", args.exclude_ids)

    config = {
        **vars(args),
        "note": "I use GT identities to evaluate my results and select excluded objects. "
        "The tracker uses boxes and image features for matching.",
    }
    (out / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

    # I load DINOv3 again when an experiment changes the image crops.
    if args.experiment in NEEDS_MODEL and model_bundle[1] is None:
        model_bundle = load_dinov3(args.model_id, token=os.getenv("HF_TOKEN"))

    if args.experiment in ["baseline", "all"]:
        baseline(data, out, args.max_age)
    if args.experiment in ["gap", "all"]:
        gap(data, out, args)
    if args.experiment in ["quality", "all"]:
        quality(data, out, args, model_bundle)
    if args.experiment in ["calibration", "all"]:
        calibration(data, out, args, model_bundle)
    if args.experiment in ["diagnostics", "all"]:
        diagnostics(data, out, args, model_bundle)
    if args.experiment == "gap_logs":
        gap_logs(data, out, args)


if __name__ == "__main__":
    main()
