from __future__ import annotations

import argparse
import json
import struct
import zlib
from pathlib import Path


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def write_blank_png(path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = b"".join(b"\x00" + b"\xff\xff\xff" * width for _ in range(height))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny Refer-KITTI-like toy dataset.")
    parser.add_argument("--out", default="toy_data")
    args = parser.parse_args()

    root = Path(args.out)
    seq = "0000"
    width, height = 100, 100
    image_dir = root / "KITTI" / "training" / "image_02" / seq
    label_dir = root / "KITTI" / "labels_with_ids" / "image_02" / seq
    expr_dir = root / "expression" / seq
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    expr_dir.mkdir(parents=True, exist_ok=True)

    for frame in range(3):
        stem = f"{frame:06d}"
        write_blank_png(image_dir / f"{stem}.png", width, height)
        car_x = 0.10 + frame * 0.05
        bus_x = 0.55
        lines = [
            f"1 7 {car_x:.4f} 0.5000 0.2000 0.2000\n",
            f"3 9 {bus_x:.4f} 0.4500 0.3000 0.2500\n",
        ]
        (label_dir / f"{stem}.txt").write_text("".join(lines), encoding="utf-8")

    expr = {
        "sentence": "the car on the left",
        "label": {
            "0": [7],
            "1": [7],
            "2": [7],
        },
    }
    (expr_dir / "000.json").write_text(json.dumps(expr, indent=2), encoding="utf-8")
    print(root)


if __name__ == "__main__":
    main()
