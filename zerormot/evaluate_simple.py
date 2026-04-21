from __future__ import annotations

import argparse
from pathlib import Path

from .io import load_mot_file
from .structures import Box, iou


def _match_frame(preds: list[tuple[int, Box, float]], gts: list[tuple[int, Box, float]], iou_thr: float):
    pairs: list[tuple[float, int, int]] = []
    for pi, (_, pbox, _) in enumerate(preds):
        for gi, (_, gbox, _) in enumerate(gts):
            pairs.append((iou(pbox, gbox), pi, gi))
    pairs.sort(reverse=True, key=lambda x: x[0])
    used_p: set[int] = set()
    used_g: set[int] = set()
    matches = 0
    for score, pi, gi in pairs:
        if score < iou_thr:
            break
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matches += 1
    return matches, len(preds) - matches, len(gts) - matches


def evaluate_pair(pred_path: Path, gt_path: Path, iou_thr: float):
    pred = load_mot_file(pred_path)
    gt = load_mot_file(gt_path)
    frames = sorted(set(pred) | set(gt))
    tp = fp = fn = 0
    for frame in frames:
        m, f_p, f_n = _match_frame(pred.get(frame, []), gt.get(frame, []), iou_thr)
        tp += m
        fp += f_p
        fn += f_n
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple IoU-based sanity evaluation for baseline outputs.")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--iou-thr", type=float, default=0.5)
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    totals = {"tp": 0, "fp": 0, "fn": 0}
    count = 0
    for pred_path in sorted(pred_dir.glob("*/*.txt")):
        rel = pred_path.relative_to(pred_dir)
        gt_path = gt_dir / rel
        result = evaluate_pair(pred_path, gt_path, args.iou_thr)
        for key in totals:
            totals[key] += result[key]
        count += 1

    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"files={count}")
    print(f"tp={totals['tp']} fp={totals['fp']} fn={totals['fn']}")
    print(f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")


if __name__ == "__main__":
    main()

