from __future__ import annotations

import json
import pickle
import struct
from pathlib import Path
from typing import Any

from .structures import Box, Detection


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return load_json(path)


def find_expression_root(data_root: str | Path, expression_root: str | Path | None = None) -> Path | None:
    root = Path(data_root)
    if expression_root is not None:
        path = Path(expression_root)
        if not path.is_absolute():
            path = root / path
        return path if path.exists() else None
    candidates = [
        root / "expression",
        root / "expression_clean",
        root / "KITTI" / "expression",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def list_expression_files(
    data_root: str | Path,
    seq: str | None = None,
    expression_root: str | Path | None = None,
) -> list[Path]:
    expr_root = find_expression_root(data_root, expression_root)
    if expr_root is None:
        return []
    if seq:
        return sorted((expr_root / seq).glob("*.json"))
    return sorted(expr_root.glob("*/*.json"))


def find_image_dir(data_root: str | Path, seq: str) -> Path | None:
    root = Path(data_root)
    candidates = [
        root / "KITTI" / "training" / "image_02" / seq,
        root / "training" / "image_02" / seq,
        root / "image_02" / seq,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def find_label_dir(data_root: str | Path, seq: str) -> Path | None:
    root = Path(data_root)
    candidates = [
        root / "KITTI" / "labels_with_ids" / "image_02" / seq,
        root / "labels_with_ids" / "image_02" / seq,
        root / "KITTI" / "labels_with_ids" / seq,
        root / "labels_with_ids" / seq,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def list_frame_stems(data_root: str | Path, seq: str) -> list[str]:
    image_dir = find_image_dir(data_root, seq)
    if image_dir:
        stems = sorted(path.stem for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)
        if stems:
            return stems
    label_dir = find_label_dir(data_root, seq)
    if label_dir:
        return sorted(path.stem for path in label_dir.glob("*.txt"))
    return []


def image_size(path: str | Path) -> tuple[int, int] | None:
    path = Path(path)
    try:
        with path.open("rb") as f:
            header = f.read(32)
            if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
                width, height = struct.unpack(">II", header[16:24])
                return int(width), int(height)
            if header[:2] == b"\xff\xd8":
                f.seek(2)
                while True:
                    marker_start = f.read(1)
                    if not marker_start:
                        return None
                    if marker_start != b"\xff":
                        continue
                    marker = f.read(1)
                    while marker == b"\xff":
                        marker = f.read(1)
                    if marker in {b"\xd8", b"\xd9"}:
                        continue
                    raw_len = f.read(2)
                    if len(raw_len) != 2:
                        return None
                    seg_len = struct.unpack(">H", raw_len)[0]
                    if marker in {
                        b"\xc0",
                        b"\xc1",
                        b"\xc2",
                        b"\xc3",
                        b"\xc5",
                        b"\xc6",
                        b"\xc7",
                        b"\xc9",
                        b"\xca",
                        b"\xcb",
                        b"\xcd",
                        b"\xce",
                        b"\xcf",
                    }:
                        data = f.read(5)
                        if len(data) != 5:
                            return None
                        height, width = struct.unpack(">HH", data[1:5])
                        return int(width), int(height)
                    f.seek(seg_len - 2, 1)
    except OSError:
        return None
    return None


def read_expression(path: str | Path) -> tuple[str, dict[int, list[int]]]:
    obj = load_json(path)
    sentence = obj.get("sentence") or obj.get("query") or obj.get("expression") or ""
    labels: dict[int, list[int]] = {}
    raw_label = obj.get("label", {})
    if isinstance(raw_label, dict):
        for key, value in raw_label.items():
            try:
                frame = int(key)
            except ValueError:
                continue
            if isinstance(value, list):
                labels[frame] = [int(v) for v in value]
            elif value is not None:
                labels[frame] = [int(value)]
    return sentence, labels


def read_kitti_label_file(path: str | Path, frame: int, image_size: tuple[int, int] | None = None) -> list[Detection]:
    path = Path(path)
    if not path.exists():
        return []
    detections: list[Detection] = []
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in lines:
        parts = line.split()
        if len(parts) < 6:
            continue
        class_id = int(float(parts[0]))
        track_id = int(float(parts[1]))
        x = float(parts[2])
        y = float(parts[3])
        w = float(parts[4])
        h = float(parts[5])
        if image_size is not None:
            width, height = image_size
            x *= width
            y *= height
            w *= width
            h *= height
        detections.append(
            Detection(
                frame=frame,
                box=Box(x=x, y=y, w=w, h=h),
                score=1.0,
                class_id=class_id,
                source_track_id=track_id,
            )
        )
    return detections


def load_detector_json(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    obj = load_json(path)
    if isinstance(obj, dict):
        return obj
    raise ValueError(f"Detector JSON must be an object keyed by frame/image id: {path}")


def detections_from_detector_json(
    detector_data: dict[str, list[dict[str, Any]]],
    keys: list[str],
    frame: int,
) -> list[Detection]:
    raw_items: list[dict[str, Any]] | None = None
    for key in keys:
        if key in detector_data:
            raw_items = detector_data[key]
            break
    if raw_items is None:
        return []
    detections: list[Detection] = []
    for item in raw_items:
        bbox = item.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        detections.append(
            Detection(
                frame=frame,
                box=Box(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                score=float(item.get("score", 1.0)),
                class_id=item.get("class_id"),
                class_text=item.get("class_text") or item.get("label") or item.get("category"),
                source_track_id=item.get("track_id"),
                meta=dict(item),
            )
        )
    return detections


def write_mot_file(rows: list[tuple[int, int, Box, float]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for frame, track_id, box, score in sorted(rows, key=lambda r: (r[0], r[1])):
            f.write(
                f"{frame},{track_id},{box.x:.2f},{box.y:.2f},"
                f"{box.w:.2f},{box.h:.2f},{score:.4f},-1,-1,-1\n"
            )


def load_mot_file(path: str | Path) -> dict[int, list[tuple[int, Box, float]]]:
    path = Path(path)
    frames: dict[int, list[tuple[int, Box, float]]] = {}
    if not path.exists():
        return frames
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        frame = int(float(parts[0]))
        track_id = int(float(parts[1]))
        box = Box(float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
        score = float(parts[6]) if len(parts) > 6 else 1.0
        frames.setdefault(frame, []).append((track_id, box, score))
    return frames


def load_pickle_or_json(path: str | Path) -> Any:
    path = Path(path)
    if path.suffix.lower() in {".p", ".pkl", ".pickle"}:
        with path.open("rb") as f:
            return pickle.load(f)
    return load_json(path)
