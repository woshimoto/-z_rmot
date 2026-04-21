from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def as_xywh(self) -> list[float]:
        return [self.x, self.y, self.w, self.h]


@dataclass
class Detection:
    frame: int
    box: Box
    score: float = 1.0
    class_id: int | None = None
    class_text: str | None = None
    source_track_id: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Tracklet:
    track_id: int
    detections: list[Detection] = field(default_factory=list)

    def add(self, det: Detection) -> None:
        self.detections.append(det)

    @property
    def start_frame(self) -> int:
        return min(det.frame for det in self.detections)

    @property
    def end_frame(self) -> int:
        return max(det.frame for det in self.detections)

    @property
    def mean_score(self) -> float:
        if not self.detections:
            return 0.0
        return sum(det.score for det in self.detections) / len(self.detections)

    @property
    def class_texts(self) -> list[str]:
        return [det.class_text for det in self.detections if det.class_text]

    @property
    def class_ids(self) -> list[int]:
        return [det.class_id for det in self.detections if det.class_id is not None]


def iou(a: Box, b: Box) -> float:
    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return inter / union

