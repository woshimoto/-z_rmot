from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zerormot.io import (
    dump_json,
    find_image_dir,
    find_label_dir,
    image_size,
    list_frame_stems,
)
from zerormot.proposals import (
    HFGroundingDINOProposalGenerator,
    collect_sequence_prompts,
    export_with_generator,
    load_baseline_config,
    read_labels_as_detector_items,
)


def _frame_image_size(image_dir: Path | None, stem: str) -> tuple[int, int] | None:
    if image_dir is None:
        return None
    for suffix in [".png", ".jpg", ".jpeg", ".bmp"]:
        path = image_dir / f"{stem}{suffix}"
        if path.exists():
            return image_size(path)
    return None


def _list_sequences(data_root: Path, seq: str | None) -> list[str]:
    if seq:
        return [seq]
    candidates = [
        data_root / "KITTI" / "training" / "image_02",
        data_root / "training" / "image_02",
        data_root / "image_02",
    ]
    for candidate in candidates:
        if candidate.exists():
            return sorted(path.name for path in candidate.iterdir() if path.is_dir())
    return []


def export_labels_backend(args: argparse.Namespace) -> dict[str, Any]:
    data_root = Path(args.data_root)
    config = load_baseline_config(args.config)
    sequences = _list_sequences(data_root, args.seq)
    detector_data: dict[str, Any] = {}
    meta = {
        "backend": args.backend,
        "data_root": str(data_root),
        "keep_track_ids": args.keep_track_ids,
        "default_score": args.default_score,
        "prompts_by_seq": {},
    }

    exported_frames = 0
    for seq in sequences:
        label_dir = find_label_dir(data_root, seq)
        image_dir = find_image_dir(data_root, seq)
        if label_dir is None:
            continue
        frame_stems = list_frame_stems(data_root, seq)
        if args.limit_frames:
            frame_stems = frame_stems[: args.limit_frames]

        meta["prompts_by_seq"][seq] = collect_sequence_prompts(
            data_root,
            seq,
            expression_root=args.expression_root,
            max_prompts=args.max_prompts,
        )

        for frame_idx, stem in enumerate(frame_stems):
            size = _frame_image_size(image_dir, stem)
            items = read_labels_as_detector_items(
                label_dir / f"{stem}.txt",
                frame=frame_idx,
                image_size=size,
                config=config,
                keep_track_id=args.keep_track_ids,
                default_score=args.default_score,
            )
            detector_data[f"{seq}/{stem}"] = items
            exported_frames += 1

    return {
        "frames": detector_data,
        "meta": meta,
        "num_sequences": len(meta["prompts_by_seq"]),
        "num_frames": exported_frames,
    }


def export_hf_grounding_dino_backend(args: argparse.Namespace) -> dict[str, Any]:
    if not args.model_id:
        raise ValueError("--model-id is required for --backend hf_grounding_dino")
    config = load_baseline_config(args.config)
    generator = HFGroundingDINOProposalGenerator(
        model_id=args.model_id,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )
    detector_data, meta = export_with_generator(
        data_root=args.data_root,
        generator=generator,
        config=config,
        seq=args.seq,
        expression_root=args.expression_root,
        prompt_scope=args.prompt_scope,
        max_prompts=args.max_prompts,
        limit_frames=args.limit_frames,
    )
    meta.update(
        {
            "backend": args.backend,
            "data_root": str(args.data_root),
            "model_id": args.model_id,
            "device": args.device,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
        }
    )
    return {
        "frames": detector_data,
        "meta": meta,
        "num_sequences": len(meta.get("prompts_by_seq", {})),
        "num_frames": len(detector_data),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Refer-KITTI per-frame proposals into detector_json format."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True, help="Output detector JSON path.")
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--backend", choices=["labels", "hf_grounding_dino"], default="labels")
    parser.add_argument("--seq", default=None)
    parser.add_argument("--expression-root", default=None)
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--max-prompts", type=int, default=32)
    parser.add_argument("--default-score", type=float, default=0.95)
    parser.add_argument("--model-id", default=None, help="HF model id for hf_grounding_dino backend.")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu or cuda.")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--prompt-scope", choices=["per_seq", "global"], default="per_seq")
    parser.add_argument(
        "--keep-track-ids",
        action="store_true",
        help="Keep GT track ids in the exported detector JSON. Off by default so tracker branch is exercised.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.backend == "labels":
        export = export_labels_backend(args)
    elif args.backend == "hf_grounding_dino":
        export = export_hf_grounding_dino_backend(args)
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")
    detector_json = export["frames"]
    dump_json(detector_json, args.output)
    meta_path = Path(args.output).with_suffix(Path(args.output).suffix + ".meta.json")
    dump_json(export["meta"], meta_path)
    print(
        f"Done. backend={args.backend} sequences={export['num_sequences']} "
        f"frames={export['num_frames']} output={args.output} meta={meta_path}"
    )


if __name__ == "__main__":
    main()
