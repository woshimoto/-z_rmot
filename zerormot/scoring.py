from __future__ import annotations

from collections import Counter
from typing import Any

from .query import ParsedQuery, category_aliases
from .structures import Detection, Tracklet


def attach_class_text_from_config(det: Detection, class_map: dict[str, list[str]]) -> None:
    if det.class_text:
        det.class_text = det.class_text.lower()
        return
    if det.class_id is None:
        return
    names = class_map.get(str(det.class_id))
    if names:
        det.class_text = " ".join(name.lower() for name in names)


def _track_words(tracklet: Tracklet) -> set[str]:
    words: set[str] = set()
    for text in tracklet.class_texts:
        for token in text.lower().replace("-", " ").split():
            words.add(token)
    return words


def category_score(tracklet: Tracklet, query: ParsedQuery) -> float:
    if not query.categories:
        return 1.0
    words = _track_words(tracklet)
    if not words:
        return 0.5
    best = 0.0
    for category in query.categories:
        aliases = category_aliases(category)
        if words & aliases:
            best = max(best, 1.0)
        elif category in words:
            best = max(best, 1.0)
    return best


def attribute_score(tracklet: Tracklet, query: ParsedQuery) -> float:
    if not query.attributes:
        return 1.0
    words = _track_words(tracklet)
    if not words:
        return 0.5
    hits = len(query.attributes & words)
    if hits:
        return min(1.0, hits / max(1, len(query.attributes)))
    return 0.0


def _frame_extents(all_dets: list[Detection]) -> dict[int, tuple[float, float, float, float]]:
    extents: dict[int, tuple[float, float, float, float]] = {}
    by_frame: dict[int, list[Detection]] = {}
    for det in all_dets:
        by_frame.setdefault(det.frame, []).append(det)
    for frame, dets in by_frame.items():
        min_x = min(det.box.cx for det in dets)
        max_x = max(det.box.cx for det in dets)
        min_area = min(det.box.area for det in dets)
        max_area = max(det.box.area for det in dets)
        extents[frame] = (min_x, max_x, min_area, max_area)
    return extents


def relation_score(tracklet: Tracklet, query: ParsedQuery, all_dets: list[Detection]) -> float:
    if not query.relations:
        return 1.0
    extents = _frame_extents(all_dets)
    scores: list[float] = []
    for det in tracklet.detections:
        if det.frame not in extents:
            continue
        min_x, max_x, min_area, max_area = extents[det.frame]
        x_den = max(1e-6, max_x - min_x)
        area_den = max(1e-6, max_area - min_area)
        per_relation: list[float] = []
        if "left_of" in query.relations:
            per_relation.append(1.0 - (det.box.cx - min_x) / x_den)
        if "right_of" in query.relations:
            per_relation.append((det.box.cx - min_x) / x_den)
        if "behind" in query.relations:
            per_relation.append(1.0 - (det.box.area - min_area) / area_den)
        if "in_front_of" in query.relations:
            per_relation.append((det.box.area - min_area) / area_den)
        if "near" in query.relations:
            per_relation.append(0.5)
        if per_relation:
            scores.append(sum(per_relation) / len(per_relation))
    if not scores:
        return 0.5
    return max(0.0, min(1.0, sum(scores) / len(scores)))


def detector_score(tracklet: Tracklet) -> float:
    return max(0.0, min(1.0, tracklet.mean_score))


def score_tracklet(
    tracklet: Tracklet,
    query: ParsedQuery,
    all_dets: list[Detection],
    config: dict[str, Any],
) -> dict[str, float]:
    cat = category_score(tracklet, query)
    attr = attribute_score(tracklet, query)
    rel = relation_score(tracklet, query, all_dets)
    det = detector_score(tracklet)
    weights = {
        "detector": float(config.get("detector_score_weight", 0.15)),
        "category": float(config.get("category_weight", 0.45)),
        "attribute": float(config.get("attribute_weight", 0.20)),
        "relation": float(config.get("relation_weight", 0.20)),
    }
    total_weight = sum(weights.values()) or 1.0
    total = (
        weights["detector"] * det
        + weights["category"] * cat
        + weights["attribute"] * attr
        + weights["relation"] * rel
    ) / total_weight
    return {
        "total": total,
        "detector": det,
        "category": cat,
        "attribute": attr,
        "relation": rel,
    }


def majority_class_text(tracklet: Tracklet) -> str | None:
    if not tracklet.class_texts:
        return None
    return Counter(tracklet.class_texts).most_common(1)[0][0]

