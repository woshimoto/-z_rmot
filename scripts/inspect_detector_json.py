from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zerormot.io import load_detector_json


def _class_text(item: dict[str, Any]) -> str:
    return str(item.get("class_text") or item.get("label") or item.get("category") or "<missing>")


def inspect_detector_json(path: str | Path, top_k: int = 20) -> dict[str, Any]:
    data = load_detector_json(path)
    frame_counts = {key: len(items) for key, items in data.items()}
    class_counter: Counter[str] = Counter()
    score_values: list[float] = []
    invalid_boxes = 0
    missing_class = 0

    for items in data.values():
        for item in items:
            bbox = item.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                invalid_boxes += 1
            if not (_class_text(item) and _class_text(item) != "<missing>"):
                missing_class += 1
            class_counter[_class_text(item)] += 1
            try:
                score_values.append(float(item.get("score", 0.0)))
            except (TypeError, ValueError):
                pass

    total_frames = len(data)
    total_detections = sum(frame_counts.values())
    empty_frames = sum(1 for count in frame_counts.values() if count == 0)
    non_empty_frames = total_frames - empty_frames
    avg_per_frame = total_detections / total_frames if total_frames else 0.0
    min_score = min(score_values) if score_values else None
    max_score = max(score_values) if score_values else None

    return {
        "path": str(path),
        "total_frames": total_frames,
        "non_empty_frames": non_empty_frames,
        "empty_frames": empty_frames,
        "total_detections": total_detections,
        "avg_detections_per_frame": avg_per_frame,
        "min_score": min_score,
        "max_score": max_score,
        "invalid_boxes": invalid_boxes,
        "missing_class_text": missing_class,
        "top_classes": class_counter.most_common(top_k),
        "sample_keys": sorted(data.keys())[: min(top_k, total_frames)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect detector_json proposal files.")
    parser.add_argument("--detector-json", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    report = inspect_detector_json(args.detector_json, top_k=args.top_k)
    print(f"detector_json={report['path']}")
    print(f"frames={report['total_frames']} non_empty={report['non_empty_frames']} empty={report['empty_frames']}")
    print(
        f"detections={report['total_detections']} "
        f"avg_per_frame={report['avg_detections_per_frame']:.2f} "
        f"score_range=({report['min_score']}, {report['max_score']})"
    )
    print(f"invalid_boxes={report['invalid_boxes']} missing_class_text={report['missing_class_text']}")
    print("top_classes:")
    for label, count in report["top_classes"]:
        print(f"  {label}: {count}")
    print("sample_keys:")
    for key in report["sample_keys"]:
        print(f"  {key}")


if __name__ == "__main__":
    main()

