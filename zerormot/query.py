from __future__ import annotations

import re
from dataclasses import dataclass, field


DEFAULT_CATEGORY_SYNONYMS = {
    "person": {"person", "pedestrian", "man", "woman", "boy", "girl", "people"},
    "vehicle": {"vehicle", "vehicles"},
    "car": {"car", "cars", "sedan", "suv", "taxi"},
    "truck": {"truck", "lorry"},
    "bus": {"bus"},
    "cyclist": {"cyclist", "rider", "bicyclist"},
    "bicycle": {"bicycle", "bike", "cycle"},
    "motorcycle": {"motorcycle", "motorbike", "scooter"},
}

COLORS = {
    "white",
    "black",
    "red",
    "blue",
    "green",
    "yellow",
    "gray",
    "grey",
    "silver",
    "brown",
    "orange",
    "pink",
    "purple",
}

RELATION_PATTERNS = {
    "left_of": [r"\bleft of\b", r"\bon the left\b", r"\bleftmost\b"],
    "right_of": [r"\bright of\b", r"\bon the right\b", r"\brightmost\b"],
    "near": [r"\bnear\b", r"\bnext to\b", r"\bbeside\b", r"\bclose to\b"],
    "behind": [r"\bbehind\b", r"\bfollowing\b", r"\bin back of\b"],
    "in_front_of": [r"\bin front of\b", r"\bahead of\b", r"\bbefore\b"],
}


@dataclass
class ParsedQuery:
    raw: str
    categories: set[str] = field(default_factory=set)
    attributes: set[str] = field(default_factory=set)
    relations: set[str] = field(default_factory=set)
    tokens: set[str] = field(default_factory=set)


def normalize_text(text: str) -> str:
    text = text.lower().replace("-", " ")
    return re.sub(r"[^a-z0-9 ]+", " ", text)


def parse_query(text: str) -> ParsedQuery:
    norm = normalize_text(text)
    tokens = set(norm.split())
    categories: set[str] = set()
    for canonical, synonyms in DEFAULT_CATEGORY_SYNONYMS.items():
        if tokens & synonyms:
            categories.add(canonical)

    attributes = tokens & COLORS
    if "grey" in attributes:
        attributes.add("gray")
    if "gray" in attributes:
        attributes.add("grey")

    relations: set[str] = set()
    for relation, patterns in RELATION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, norm):
                relations.add(relation)
                break

    return ParsedQuery(
        raw=text,
        categories=categories,
        attributes=attributes,
        relations=relations,
        tokens=tokens,
    )


def category_aliases(category: str) -> set[str]:
    return set(DEFAULT_CATEGORY_SYNONYMS.get(category, {category}))
