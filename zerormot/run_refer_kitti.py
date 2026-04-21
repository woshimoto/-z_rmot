from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .io import (
    detections_from_detector_json,
    dump_json,
    find_image_dir,
    find_label_dir,
    image_size,
    list_expression_files,
    list_frame_stems,
    load_config,
    load_detector_json,
    read_expression,
    read_kitti_label_file,
    write_mot_file,
)
from .query import parse_query
from .scoring import attach_class_text_from_config, majority_class_text, score_tracklet
from .structures import Detection
from .tracker import SimpleIoUTracker, group_by_source_track_id


def _frame_image_size(image_dir: Path | None, stem: str) -> tuple[int, int] | None:
    if image_dir is None:
        return None
    for suffix in [".png", ".jpg", ".jpeg", ".bmp"]:
        path = image_dir / f"{stem}{suffix}"
        if path.exists():
            return image_size(path)
    return None


def _load_sequence_detections(
    data_root: Path,
    seq: str,
    frame_stems: list[str],
    proposal_source: str,
    detector_data: dict[str, list[dict[str, Any]]] | None,
    config: dict[str, Any],
) -> list[Detection]:
    all_dets: list[Detection] = []
    image_dir = find_image_dir(data_root, seq)
    label_dir = find_label_dir(data_root, seq)
    class_map = config.get("kitti_class_names", {})

    for frame_idx, stem in enumerate(frame_stems):
        if proposal_source in {"labels", "oracle"}:
            if label_dir is None:
                continue
            size = _frame_image_size(image_dir, stem)
            frame_dets = read_kitti_label_file(label_dir / f"{stem}.txt", frame=frame_idx, image_size=size)
        else:
            if detector_data is None:
                frame_dets = []
            else:
                keys = [
                    f"{seq}/{stem}",
                    f"{seq}/{frame_idx}",
                    f"{seq}/{frame_idx:06d}",
                    f"{seq}/{stem}.png",
                    f"{seq}/{stem}.jpg",
                ]
                frame_dets = detections_from_detector_json(detector_data, keys, frame=frame_idx)

        for det in frame_dets:
            attach_class_text_from_config(det, class_map)
            all_dets.append(det)
    return all_dets


def _build_tracklets(
    detections: list[Detection],
    proposal_source: str,
    config: dict[str, Any],
):
    min_track_length = int(config.get("min_track_length", 1))
    has_ids = proposal_source in {"labels", "oracle"} or any(det.source_track_id is not None for det in detections)
    if has_ids:
        return group_by_source_track_id(detections, min_track_length=min_track_length)

    tracker = SimpleIoUTracker(
        iou_threshold=float(config.get("tracker_iou_threshold", 0.3)),
        max_missing_frames=int(config.get("max_missing_frames", 2)),
    )
    by_frame: dict[int, list[Detection]] = {}
    for det in detections:
        by_frame.setdefault(det.frame, []).append(det)
    for frame in sorted(by_frame):
        tracker.update(by_frame[frame])
    return tracker.finish(min_track_length=min_track_length)


def _prediction_rows(tracklets, scores: dict[int, dict[str, float]], threshold: float):
    rows = []
    for tracklet in tracklets:
        score = scores[tracklet.track_id]["total"]
        if score < threshold:
            continue
        for det in tracklet.detections:
            rows.append((det.frame + 1, tracklet.track_id, det.box, score))
    return rows


def _gt_rows(
    data_root: Path,
    seq: str,
    frame_stems: list[str],
    labels_by_frame: dict[int, list[int]],
) -> list[tuple[int, int, Any, float]]:
    label_dir = find_label_dir(data_root, seq)
    image_dir = find_image_dir(data_root, seq)
    if label_dir is None:
        return []
    rows = []
    for frame_idx, target_ids in labels_by_frame.items():
        if frame_idx < 0 or frame_idx >= len(frame_stems):
            continue
        stem = frame_stems[frame_idx]
        size = _frame_image_size(image_dir, stem)
        target_set = set(target_ids)
        dets = read_kitti_label_file(label_dir / f"{stem}.txt", frame=frame_idx, image_size=size)
        if dets and not any(det.source_track_id in target_set for det in dets):
            plus_one_stem = f"{frame_idx + 1:06d}"
            plus_one_path = label_dir / f"{plus_one_stem}.txt"
            plus_one_size = _frame_image_size(image_dir, plus_one_stem) or size
            plus_one_dets = read_kitti_label_file(plus_one_path, frame=frame_idx, image_size=plus_one_size)
            if any(det.source_track_id in target_set for det in plus_one_dets):
                dets = plus_one_dets
        for det in dets:
            if det.source_track_id in target_set:
                rows.append((frame_idx + 1, int(det.source_track_id), det.box, 1.0))
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    config = load_config(args.config)
    detector_data = load_detector_json(args.detector_json) if args.detector_json else None
    expr_files = list_expression_files(data_root, args.seq, args.expression_root)
    if args.limit:
        expr_files = expr_files[: args.limit]

    seq_to_frames: dict[str, list[str]] = {}
    seq_to_dets: dict[str, list[Detection]] = {}
    seq_to_tracklets: dict[str, Any] = {}

    summary: dict[str, Any] = {
        "data_root": str(data_root),
        "proposal_source": args.proposal_source,
        "num_expressions": 0,
        "num_predictions": 0,
        "items": [],
    }

    for expr_path in expr_files:
        seq = expr_path.parent.name
        if seq not in seq_to_frames:
            seq_to_frames[seq] = list_frame_stems(data_root, seq)
            seq_to_dets[seq] = _load_sequence_detections(
                data_root,
                seq,
                seq_to_frames[seq],
                args.proposal_source,
                detector_data,
                config,
            )
            seq_to_tracklets[seq] = _build_tracklets(seq_to_dets[seq], args.proposal_source, config)

        sentence, labels_by_frame = read_expression(expr_path)
        query = parse_query(sentence)
        threshold = float(args.score_threshold if args.score_threshold is not None else config.get("score_threshold", 0.15))
        tracklets = seq_to_tracklets[seq]
        detections = seq_to_dets[seq]
        scores = {
            tracklet.track_id: score_tracklet(tracklet, query, detections, config)
            for tracklet in tracklets
        }
        gt_rows = _gt_rows(data_root, seq, seq_to_frames[seq], labels_by_frame)
        if args.proposal_source == "oracle":
            pred_rows = gt_rows
        else:
            pred_rows = _prediction_rows(tracklets, scores, threshold)

        pred_path = output_dir / "pred" / seq / f"{expr_path.stem}.txt"
        gt_path = output_dir / "gt" / seq / f"{expr_path.stem}.txt"
        write_mot_file(pred_rows, pred_path)
        write_mot_file(gt_rows, gt_path)

        ranked = sorted(tracklets, key=lambda t: scores[t.track_id]["total"], reverse=True)[:10]
        item = {
            "seq": seq,
            "expression_file": str(expr_path),
            "sentence": sentence,
            "parsed": {
                "categories": sorted(query.categories),
                "attributes": sorted(query.attributes),
                "relations": sorted(query.relations),
            },
            "num_tracklets": len(tracklets),
            "num_pred_rows": len(pred_rows),
            "num_gt_rows": len(gt_rows),
            "top_tracklets": [
                {
                    "track_id": t.track_id,
                    "score": scores[t.track_id],
                    "length": len(t.detections),
                    "class_text": majority_class_text(t),
                }
                for t in ranked
            ],
        }
        summary["items"].append(item)
        summary["num_expressions"] += 1
        summary["num_predictions"] += len(pred_rows)

    dump_json(summary, output_dir / "summary.json")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a minimal Refer-KITTI zero-shot RMOT baseline.")
    parser.add_argument("--data-root", required=True, help="Path to Refer-KITTI root.")
    parser.add_argument("--output-dir", default="outputs/refer_kitti_baseline")
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--seq", default=None, help="Optional sequence id, e.g. 0000.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of expression files for smoke tests.")
    parser.add_argument("--expression-root", default=None, help="Optional expression directory; defaults to expression/ auto-detection.")
    parser.add_argument("--proposal-source", choices=["labels", "detector_json", "oracle"], default="labels")
    parser.add_argument("--detector-json", default=None, help="Detector JSON when proposal source is detector_json.")
    parser.add_argument("--score-threshold", type=float, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    summary = run(args)
    print(
        f"Done. expressions={summary['num_expressions']} "
        f"prediction_rows={summary['num_predictions']} "
        f"summary={Path(args.output_dir) / 'summary.json'}"
    )


if __name__ == "__main__":
    main()
