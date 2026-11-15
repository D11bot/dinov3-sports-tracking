"""I measure identity continuity, recovery and detection quality in my experiments."""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from .association import bbox_iou
from .corruption import box_iou_xywh

import pandas as pd

def evaluate_tracking(outputs):
    previous = {}
    changes = 0
    correct_links = 0
    total_links = 0
    observations = 0

    for frame_id in sorted(outputs):
        data = outputs[frame_id]

        for gt_id, pred_id in zip(
            data["gt_ids"], data["pred_ids"]
        ):
            gt_id = int(gt_id)
            pred_id = int(pred_id)
            observations += 1

            if gt_id in previous:
                last_frame, last_pred = previous[gt_id]

                if pred_id != last_pred:
                    changes += 1

                if frame_id == last_frame + 1:
                    total_links += 1
                    if pred_id == last_pred:
                        correct_links += 1

            previous[gt_id] = (frame_id, pred_id)

    return {
        "Identity Changes": changes,
        "Consecutive Link Accuracy": (
            correct_links / total_links
            if total_links else float("nan")
        ),
        "GT Observations": observations,
        "Consecutive Links": total_links
    }


import pandas as pd

def audit_track_identity(outputs):
    records = []

    for frame_id in sorted(outputs):
        data = outputs[frame_id]

        for gt_id, pred_id in zip(
            data["gt_ids"], data["pred_ids"]
        ):
            records.append({
                "Frame": int(frame_id),
                "GT_ID": int(gt_id),
                "Pred_ID": int(pred_id)
            })

    df = pd.DataFrame(records)

    # I check whether a predicted track was used by several GT identities.
    pred_to_gt = (
        df.groupby("Pred_ID")["GT_ID"]
        .nunique()
    )

    # I check whether a GT identity received several track IDs.
    gt_to_pred = (
        df.groupby("GT_ID")["Pred_ID"]
        .nunique()
    )

    # I check for repeated predicted IDs in the same frame.
    duplicates = df.duplicated(
        subset=["Frame", "Pred_ID"]
    ).sum()

    return {
        "Pred_IDs_with_multiple_GT_IDs": int(
            (pred_to_gt > 1).sum()
        ),
        "GT_IDs_with_multiple_Pred_IDs": int(
            (gt_to_pred > 1).sum()
        ),
        "Same_frame_duplicate_Pred_IDs": int(duplicates),
        "Total_predicted_tracks": int(
            df["Pred_ID"].nunique()
        ),
        "Total_GT_identities": int(
            df["GT_ID"].nunique()
        )
    }


def get_track_for_gt(outputs, frame_id, gt_id):
    data = outputs[frame_id]

    indices = np.where(data["gt_ids"] == gt_id)[0]

    if len(indices) != 1:
        return None

    return int(data["pred_ids"][indices[0]])


def evaluate_gap_recovery(
    outputs,
    target_gt_id,
    start_frame,
    gap_length
):
    before_frame = start_frame - 1
    after_frame = start_frame + gap_length

    before_id = get_track_for_gt(
        outputs, before_frame, target_gt_id
    )

    after_id = get_track_for_gt(
        outputs, after_frame, target_gt_id
    )

    if before_id is None or after_id is None:
        return {
            "Recovered": False,
            "Hijacked": False,
            "Before_ID": before_id,
            "After_ID": after_id
        }

    # I check whether another GT identity used this track during the gap.
    hijacked = False

    for frame_id in range(start_frame, after_frame + 1):
        data = outputs[frame_id]

        for gt_id, pred_id in zip(
            data["gt_ids"],
            data["pred_ids"]
        ):
            if (
                int(pred_id) == before_id
                and int(gt_id) != target_gt_id
            ):
                hijacked = True

    recovered = (
        before_id == after_id
        and not hijacked
    )

    return {
        "Recovered": recovered,
        "Hijacked": hijacked,
        "Before_ID": before_id,
        "After_ID": after_id
    }

import numpy as np
import pandas as pd

def evaluate_known_correspondence(outputs, corrupted_data):
    records = []

    for frame_id in sorted(outputs):
        pred = outputs[frame_id]["pred_ids"]
        source_ids = corrupted_data[frame_id]["gt_ids"]

        assert len(pred) == len(source_ids)

        for gt_id, pred_id in zip(source_ids, pred):
            records.append({
                "Frame": int(frame_id),
                "GT_ID": int(gt_id),
                "Pred_ID": int(pred_id)
            })

    df = pd.DataFrame(records)

    # I check real GT identities separately from synthetic false positives.
    genuine = df[df["GT_ID"] >= 0].copy()
    fps = df[df["GT_ID"] < 0].copy()

    identity_changes = 0

    for gt_id, group in genuine.groupby("GT_ID"):
        group = group.sort_values("Frame")
        ids = group["Pred_ID"].to_numpy()
        identity_changes += int(np.sum(ids[1:] != ids[:-1]))

    gt_fragments = (
        genuine.groupby("GT_ID")["Pred_ID"].nunique()
    )

    pred_contamination = (
        genuine.groupby("Pred_ID")["GT_ID"].nunique()
    )

    fp_track_ids = set(fps["Pred_ID"])
    genuine_track_ids = set(genuine["Pred_ID"])

    return {
        "Identity_Changes": identity_changes,
        "Fragmented_GT_IDs": int((gt_fragments > 1).sum()),
        "Contaminated_Tracks": int((pred_contamination > 1).sum()),
        "FP_Tracks_Shared_With_GT": len(
            fp_track_ids & genuine_track_ids
        )
    }

def evaluate_quality_tracking(outputs, gt_reference, iou_threshold=0.5):
    observations = []
    gt_total = 0
    pred_total = 0
    matched_total = 0
    iou_sum = 0.0

    for frame_id in sorted(gt_reference):
        gt_data = gt_reference[frame_id]
        pred_data = outputs[frame_id]

        gt_boxes = gt_data["boxes"]
        gt_ids = gt_data["gt_ids"]

        pred_boxes = pred_data["boxes"]
        pred_ids = pred_data["pred_ids"]

        gt_total += len(gt_boxes)
        pred_total += len(pred_boxes)

        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            continue

        ious = np.array([
            [
                box_iou_xywh(gbox, pbox)
                for pbox in pred_boxes
            ]
            for gbox in gt_boxes
        ])

        # I match GT and predicted boxes spatially for evaluation.
        rows, cols = linear_sum_assignment(-ious)

        for i, j in zip(rows, cols):
            if ious[i, j] < iou_threshold:
                continue

            matched_total += 1
            iou_sum += float(ious[i, j])

            observations.append({
                "Frame": frame_id,
                "GT_ID": int(gt_ids[i]),
                "Pred_ID": int(pred_ids[j])
            })

    tp = matched_total
    fn = gt_total - tp
    fp = pred_total - tp

    df = pd.DataFrame(
        observations,
        columns=["Frame", "GT_ID", "Pred_ID"]
    )

    id_changes = 0
    contaminated_tracks = 0
    fragmented_identities = 0

    if not df.empty:
        gt_to_pred = df.groupby("GT_ID")["Pred_ID"].nunique()
        pred_to_gt = df.groupby("Pred_ID")["GT_ID"].nunique()

        fragmented_identities = int((gt_to_pred > 1).sum())
        contaminated_tracks = int((pred_to_gt > 1).sum())

        for gt_id, group in df.groupby("GT_ID"):
            ids = group.sort_values("Frame")["Pred_ID"].to_numpy()
            id_changes += int(np.count_nonzero(ids[1:] != ids[:-1]))

    return {
        "GT_Count": gt_total,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Detection_Recall": tp / gt_total if gt_total else np.nan,
        "Detection_Precision": tp / pred_total if pred_total else np.nan,
        "Mean_Matched_IoU": iou_sum / tp if tp else np.nan,
        "ID_Changes_Diagnostic": id_changes,
        "Fragmented_GT_IDs": fragmented_identities,
        "Contaminated_Tracks": contaminated_tracks
    }
