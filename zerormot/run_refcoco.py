from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .io import dump_json, load_config, load_detector_json, load_json, load_pickle_or_json
from .query import parse_query
from .scoring import score_tracklet
from .structures import Box, Detection, Tracklet, iou


def _sentence_from_ref(ref: dict[str, Any]) -> str:
    if ref.get("sentence"):
        return str(ref["sentence"])
    sentences = ref.get("sentences") or []
    if sentences:
        first = sentences[0]
        if isinstance(first, dict):
            return str(first.get("raw") or first.get("sent") or first.get("sentence") or "")
        return str(first)
    return str(ref.get("query") or "")


def _build_coco_indexes(instances: dict[str, Any]):
    categories = {cat["id"]: cat["name"] for cat in instances.get("categories", [])}
    images = {img["id"]: img for img in instances.get("images", [])}
    anns_by_image: dict[int, list[dict[str, Any]]] = {}
    anns_by_id: dict[int, dict[str, Any]] = {}
    for ann in instances.get("annotations", []):
        image_id = int(ann["image_id"])
        anns_by_image.setdefault(image_id, []).append(ann)
        anns_by_id[int(ann["id"])] = ann
    return categories, images, anns_by_image, anns_by_id


def _candidate_detections_from_coco(
    anns: list[dict[str, Any]],
    categories: dict[int, str],
) -> list[Detection]:
    detections: list[Detection] = []
    for ann in anns:
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        category_id = int(ann.get("category_id", -1))
        detections.append(
            Detection(
                frame=0,
                box=Box(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                score=1.0,
                class_id=category_id,
                class_text=categories.get(category_id, str(category_id)),
                source_track_id=int(ann["id"]),
            )
        )
    return detections


def _candidate_detections_from_json(
    detector_data: dict[str, list[dict[str, Any]]],
    image_id: int,
    file_name: str | None,
) -> list[Detection]:
    raw_items = detector_data.get(str(image_id))
    if raw_items is None and file_name:
        raw_items = detector_data.get(file_name) or detector_data.get(Path(file_name).name)
    if raw_items is None:
        return []
    detections: list[Detection] = []
    for idx, item in enumerate(raw_items):
        bbox = item.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        detections.append(
            Detection(
                frame=0,
                box=Box(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                score=float(item.get("score", 1.0)),
                class_id=item.get("class_id"),
                class_text=item.get("class_text") or item.get("label") or item.get("category"),
                source_track_id=item.get("ann_id", idx + 1),
            )
        )
    return detections


def _score_detection(det: Detection, query, all_dets: list[Detection], config: dict[str, Any]) -> dict[str, float]:
    tracklet = Tracklet(track_id=int(det.source_track_id or 1), detections=[det])
    return score_tracklet(tracklet, query, all_dets, config)


def run(args: argparse.Namespace) -> dict[str, Any]:
    instances = load_json(args.instances_json)
    refs = load_pickle_or_json(args.refs)
    if not isinstance(refs, list):
        raise ValueError("Refs file must contain a list of RefCOCO ref dictionaries.")

    config = load_config(args.config)
    detector_data = load_detector_json(args.detector_json) if args.detector_json else None
    categories, images, anns_by_image, anns_by_id = _build_coco_indexes(instances)

    predictions: list[dict[str, Any]] = []
    correct = total = 0
    for ref in refs:
        if args.split and ref.get("split") != args.split:
            continue
        if args.limit and total >= args.limit:
            break

        image_id = int(ref["image_id"])
        image_info = images.get(image_id, {})
        sentence = _sentence_from_ref(ref)
        query = parse_query(sentence)

        if args.proposal_source == "coco":
            detections = _candidate_detections_from_coco(anns_by_image.get(image_id, []), categories)
        else:
            detections = _candidate_detections_from_json(detector_data or {}, image_id, image_info.get("file_name"))

        scored = [(det, _score_detection(det, query, detections, config)) for det in detections]
        scored.sort(key=lambda item: item[1]["total"], reverse=True)
        best_det = scored[0][0] if scored else None
        best_score = scored[0][1] if scored else {"total": 0.0}

        gt_ann = anns_by_id.get(int(ref["ann_id"])) if ref.get("ann_id") is not None else None
        gt_bbox = gt_ann.get("bbox") if gt_ann else None
        best_iou = 0.0
        is_correct = False
        if best_det is not None and gt_bbox:
            gt_box = Box(float(gt_bbox[0]), float(gt_bbox[1]), float(gt_bbox[2]), float(gt_bbox[3]))
            best_iou = iou(best_det.box, gt_box)
            is_correct = best_iou >= args.iou_thr
            correct += int(is_correct)
        total += 1

        predictions.append(
            {
                "ref_id": ref.get("ref_id"),
                "image_id": image_id,
                "ann_id": ref.get("ann_id"),
                "sentence": sentence,
                "pred_bbox": best_det.box.as_xywh() if best_det else None,
                "pred_score": best_score,
                "iou": best_iou,
                "correct": is_correct,
                "parsed": {
                    "categories": sorted(query.categories),
                    "attributes": sorted(query.attributes),
                    "relations": sorted(query.relations),
                },
            }
        )

    metrics = {
        "split": args.split,
        "proposal_source": args.proposal_source,
        "total": total,
        "correct": correct,
        "acc_at_iou": correct / total if total else 0.0,
        "iou_thr": args.iou_thr,
    }
    output = {"metrics": metrics, "predictions": predictions}
    dump_json(output, args.output)
    return metrics


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a simple RefCOCO grounding baseline.")
    parser.add_argument("--instances-json", required=True, help="COCO instances JSON.")
    parser.add_argument("--refs", required=True, help="RefCOCO refs pickle or JSON list.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", default="outputs/refcoco_predictions.json")
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--proposal-source", choices=["coco", "detector_json"], default="coco")
    parser.add_argument("--detector-json", default=None)
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    metrics = run(args)
    print(
        f"Done. split={metrics['split']} total={metrics['total']} "
        f"acc@{metrics['iou_thr']:.2f}={metrics['acc_at_iou']:.4f} output={args.output}"
    )


if __name__ == "__main__":
    main()

