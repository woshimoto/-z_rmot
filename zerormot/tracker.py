from __future__ import annotations

from dataclasses import dataclass

from .structures import Detection, Tracklet, iou


@dataclass
class _ActiveTrack:
    tracklet: Tracklet
    last_det: Detection
    missing: int = 0


def group_by_source_track_id(detections: list[Detection], min_track_length: int = 1) -> list[Tracklet]:
    grouped: dict[int, Tracklet] = {}
    next_id = 1
    for det in sorted(detections, key=lambda d: (d.frame, d.source_track_id or 0)):
        source_id = det.source_track_id
        if source_id is None:
            source_id = next_id
            next_id += 1
        if source_id not in grouped:
            grouped[source_id] = Tracklet(track_id=source_id)
        grouped[source_id].add(det)
    return [t for t in sorted(grouped.values(), key=lambda t: t.track_id) if len(t.detections) >= min_track_length]


class SimpleIoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missing_frames: int = 2) -> None:
        self.iou_threshold = iou_threshold
        self.max_missing_frames = max_missing_frames
        self._next_id = 1
        self._active: list[_ActiveTrack] = []
        self._finished: list[Tracklet] = []

    def update(self, frame_dets: list[Detection]) -> None:
        for active in self._active:
            active.missing += 1

        pairs: list[tuple[float, int, int]] = []
        for ai, active in enumerate(self._active):
            for di, det in enumerate(frame_dets):
                pairs.append((iou(active.last_det.box, det.box), ai, di))
        pairs.sort(reverse=True, key=lambda x: x[0])

        matched_active: set[int] = set()
        matched_dets: set[int] = set()
        for score, ai, di in pairs:
            if score < self.iou_threshold:
                break
            if ai in matched_active or di in matched_dets:
                continue
            active = self._active[ai]
            det = frame_dets[di]
            active.tracklet.add(det)
            active.last_det = det
            active.missing = 0
            matched_active.add(ai)
            matched_dets.add(di)

        for di, det in enumerate(frame_dets):
            if di in matched_dets:
                continue
            tracklet = Tracklet(track_id=self._next_id)
            self._next_id += 1
            tracklet.add(det)
            self._active.append(_ActiveTrack(tracklet=tracklet, last_det=det, missing=0))

        still_active: list[_ActiveTrack] = []
        for active in self._active:
            if active.missing > self.max_missing_frames:
                self._finished.append(active.tracklet)
            else:
                still_active.append(active)
        self._active = still_active

    def finish(self, min_track_length: int = 1) -> list[Tracklet]:
        all_tracks = self._finished + [active.tracklet for active in self._active]
        self._active = []
        self._finished = []
        return [t for t in sorted(all_tracks, key=lambda t: t.track_id) if len(t.detections) >= min_track_length]

