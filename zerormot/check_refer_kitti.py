from __future__ import annotations

import argparse
from pathlib import Path

from .io import (
    find_expression_root,
    find_image_dir,
    find_label_dir,
    list_expression_files,
    list_frame_stems,
    read_expression,
    read_kitti_label_file,
)


def _ids_in_label_file(path: Path) -> set[int]:
    ids: set[int] = set()
    if not path.exists():
        return ids
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            ids.add(int(float(parts[1])))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Check that a Refer-KITTI tree is readable by the baseline.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--expression-root", default=None)
    parser.add_argument("--seq", default=None)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    root = Path(args.data_root)
    expr_root = find_expression_root(root, args.expression_root)
    expr_files = list_expression_files(root, args.seq, args.expression_root)
    if args.limit:
        expr_files = expr_files[: args.limit]

    print(f"data_root={root}")
    print(f"expression_root={expr_root}")
    print(f"sample_expressions={len(expr_files)}")
    if not expr_files:
        raise SystemExit("No expression JSON files found.")

    checked = 0
    missing_image_dirs: set[str] = set()
    missing_label_dirs: set[str] = set()
    missing_label_files = 0
    missing_ref_ids = 0
    exact_frame_hits = 0
    plus_one_frame_hits = 0

    for expr_path in expr_files:
        seq = expr_path.parent.name
        image_dir = find_image_dir(root, seq)
        label_dir = find_label_dir(root, seq)
        frame_stems = list_frame_stems(root, seq)
        if image_dir is None:
            missing_image_dirs.add(seq)
        if label_dir is None:
            missing_label_dirs.add(seq)
            continue

        sentence, labels_by_frame = read_expression(expr_path)
        print(f"[{seq}/{expr_path.name}] sentence={sentence!r} frames={len(labels_by_frame)} images={len(frame_stems)}")
        for frame_id, ref_ids in sorted(labels_by_frame.items())[:5]:
            checked += 1
            exact_path = label_dir / f"{frame_id:06d}.txt"
            plus_one_path = label_dir / f"{frame_id + 1:06d}.txt"
            exact_ids = _ids_in_label_file(exact_path)
            plus_one_ids = _ids_in_label_file(plus_one_path)
            ref_set = set(ref_ids)
            if exact_ids:
                exact_frame_hits += 1
            if plus_one_ids:
                plus_one_frame_hits += 1
            if not exact_ids:
                missing_label_files += 1
            if exact_ids and not (exact_ids & ref_set):
                missing_ref_ids += 1
                print(
                    f"  warning: frame {frame_id:06d} label exists but ref ids {sorted(ref_set)} "
                    f"not in ids sample={sorted(exact_ids)[:10]}"
                )

        if label_dir and frame_stems:
            sample = label_dir / f"{frame_stems[0]}.txt"
            dets = read_kitti_label_file(sample, frame=0, image_size=None)
            print(f"  label_dir={label_dir} first_label={sample.name} rows={len(dets)}")

    print("summary:")
    print(f"  checked_frame_refs={checked}")
    print(f"  missing_image_dirs={sorted(missing_image_dirs)}")
    print(f"  missing_label_dirs={sorted(missing_label_dirs)}")
    print(f"  missing_exact_label_files={missing_label_files}")
    print(f"  missing_ref_ids_in_exact_labels={missing_ref_ids}")
    print(f"  exact_frame_label_hits={exact_frame_hits}")
    print(f"  plus_one_frame_label_hits={plus_one_frame_hits}")
    if missing_image_dirs or missing_label_dirs:
        raise SystemExit("Directory check failed.")
    if missing_label_files:
        print("note: some expression frame keys did not have exact label files; inspect exact vs plus-one hits above.")
    print("OK: baseline can read this Refer-KITTI tree.")


if __name__ == "__main__":
    main()

