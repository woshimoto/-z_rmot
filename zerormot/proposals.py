from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .io import find_image_dir, image_size, list_expression_files, list_frame_stems, load_config, read_expression, read_kitti_label_file
from .query import ParsedQuery, parse_query
from .structures import Detection


DEFAULT_OPEN_VOCAB_PROMPTS = [
    "car",
    "vehicle",
    "sedan",
    "truck",
    "bus",
    "van",
    "person",
    "pedestrian",
    "cyclist",
    "bicycle",
    "motorcycle",
]


class ProposalGenerator:
    """Interface for future open-vocabulary proposal backends."""

    def generate_for_frame(self, image_path: str, prompts: list[str], frame: int) -> list[Detection]:
        raise NotImplementedError


class HFGroundingDINOProposalGenerator(ProposalGenerator):
    """Optional Grounding DINO backend using Hugging Face Transformers."""

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> None:
        try:
            import torch
            from PIL import Image
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "HFGroundingDINOProposalGenerator requires optional dependencies: "
                "`torch`, `transformers`, and `Pillow`."
            ) from exc

        self._torch = torch
        self._Image = Image
        self.model_id = model_id
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.model.eval()

    def generate_for_frame(self, image_path: str, prompts: list[str], frame: int) -> list[Detection]:
        if not prompts:
            prompts = list(DEFAULT_OPEN_VOCAB_PROMPTS)
        image = self._Image.open(image_path).convert("RGB")
        h, w = image.size[1], image.size[0]
        text_labels = [[_normalize_prompt_for_gdino(prompt) for prompt in prompts]]
        inputs = self.processor(images=image, text=text_labels, return_tensors="pt").to(self.model.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(h, w)],
        )
        result = results[0]
        raw_labels = result.get("text_labels") or result.get("labels") or []
        detections: list[Detection] = []
        for box, score, label in zip(result["boxes"], result["scores"], raw_labels):
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            class_text = _denormalize_detected_label(label, prompts)
            detections.append(
                Detection(
                    frame=frame,
                    box=_xyxy_to_detection_box(x1, y1, x2, y2),
                    score=float(score.item() if hasattr(score, "item") else score),
                    class_text=class_text,
                )
            )
        return detections


def class_text_from_id(class_id: int | None, config: dict[str, Any]) -> str | None:
    if class_id is None:
        return None
    class_map = config.get("kitti_class_names", {})
    names = class_map.get(str(class_id))
    if not names:
        return None
    if isinstance(names, list):
        return names[0]
    return str(names)


def _normalize_prompt_for_gdino(prompt: str) -> str:
    prompt = prompt.strip()
    if not prompt:
        return prompt
    if prompt.endswith("."):
        return prompt
    return prompt


def _denormalize_detected_label(label: Any, prompts: list[str]) -> str:
    if isinstance(label, str):
        return label.strip().rstrip(".")
    if isinstance(label, int) and 0 <= label < len(prompts):
        return prompts[label]
    return str(label).strip().rstrip(".")


def _xyxy_to_detection_box(x1: float, y1: float, x2: float, y2: float):
    from .structures import Box

    return Box(x=x1, y=y1, w=max(0.0, x2 - x1), h=max(0.0, y2 - y1))


def detection_to_json_item(
    det: Detection,
    config: dict[str, Any],
    default_score: float = 1.0,
    keep_track_id: bool = False,
) -> dict[str, Any]:
    item = {
        "bbox": [round(v, 4) for v in det.box.as_xywh()],
        "score": float(det.score if det.score is not None else default_score),
    }
    class_text = det.class_text or class_text_from_id(det.class_id, config)
    if det.class_id is not None:
        item["class_id"] = int(det.class_id)
    if class_text:
        item["class_text"] = class_text
    if keep_track_id and det.source_track_id is not None:
        item["track_id"] = int(det.source_track_id)
    return item


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def prompts_from_query(query: ParsedQuery) -> list[str]:
    prompts: list[str] = []
    for category in sorted(query.categories):
        prompts.append(category)
    for category in sorted(query.categories):
        for attr in sorted(query.attributes):
            prompts.append(f"{attr} {category}")
    for token in sorted(query.tokens):
        if token in {"the", "a", "an", "on", "in", "of", "to", "and", "with"}:
            continue
        prompts.append(token)
    return dedupe_preserve_order(prompts)


def collect_sequence_prompts(
    data_root: str | Path,
    seq: str,
    expression_root: str | Path | None = None,
    max_prompts: int = 32,
) -> list[str]:
    expr_files = list_expression_files(data_root, seq=seq, expression_root=expression_root)
    counter: Counter[str] = Counter()
    for expr_path in expr_files:
        sentence, _ = read_expression(expr_path)
        query = parse_query(sentence)
        for prompt in prompts_from_query(query):
            counter[prompt] += 1
    if not counter:
        return list(DEFAULT_OPEN_VOCAB_PROMPTS)
    ranked = [prompt for prompt, _ in counter.most_common(max_prompts)]
    merged = ranked + [p for p in DEFAULT_OPEN_VOCAB_PROMPTS if p not in counter]
    return dedupe_preserve_order(merged)[:max_prompts]


def collect_global_prompts(
    data_root: str | Path,
    expression_root: str | Path | None = None,
    max_prompts: int = 64,
) -> list[str]:
    expr_files = list_expression_files(data_root, seq=None, expression_root=expression_root)
    counter: Counter[str] = Counter()
    for expr_path in expr_files:
        sentence, _ = read_expression(expr_path)
        query = parse_query(sentence)
        for prompt in prompts_from_query(query):
            counter[prompt] += 1
    if not counter:
        return list(DEFAULT_OPEN_VOCAB_PROMPTS)
    ranked = [prompt for prompt, _ in counter.most_common(max_prompts)]
    merged = ranked + [p for p in DEFAULT_OPEN_VOCAB_PROMPTS if p not in counter]
    return dedupe_preserve_order(merged)[:max_prompts]


def read_labels_as_detector_items(
    label_path: str | Path,
    frame: int,
    image_size: tuple[int, int] | None,
    config: dict[str, Any],
    keep_track_id: bool = False,
    default_score: float = 1.0,
) -> list[dict[str, Any]]:
    dets = read_kitti_label_file(label_path, frame=frame, image_size=image_size)
    items = []
    for det in dets:
        if det.score is None:
            det.score = default_score
        items.append(
            detection_to_json_item(
                det,
                config=config,
                default_score=default_score,
                keep_track_id=keep_track_id,
            )
        )
    return items


def load_baseline_config(path: str | Path | None) -> dict[str, Any]:
    return load_config(path)


def export_with_generator(
    data_root: str | Path,
    generator: ProposalGenerator,
    config: dict[str, Any],
    seq: str | None = None,
    expression_root: str | Path | None = None,
    prompt_scope: str = "per_seq",
    max_prompts: int = 32,
    limit_frames: int | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    root = Path(data_root)
    sequences: list[str]
    if seq is not None:
        sequences = [seq]
    else:
        candidates = []
        for base in [
            root / "KITTI" / "training" / "image_02",
            root / "training" / "image_02",
            root / "image_02",
        ]:
            if base.exists():
                candidates = sorted(path.name for path in base.iterdir() if path.is_dir())
                break
        sequences = candidates

    detector_data: dict[str, list[dict[str, Any]]] = {}
    meta: dict[str, Any] = {
        "prompt_scope": prompt_scope,
        "prompts_by_seq": {},
    }
    global_prompts = None
    if prompt_scope == "global":
        global_prompts = collect_global_prompts(root, expression_root=expression_root, max_prompts=max_prompts)
        meta["global_prompts"] = global_prompts

    for seq_name in sequences:
        image_dir = find_image_dir(root, seq_name)
        if image_dir is None:
            continue
        frame_stems = list_frame_stems(root, seq_name)
        if limit_frames:
            frame_stems = frame_stems[:limit_frames]
        prompts = (
            global_prompts
            if global_prompts is not None
            else collect_sequence_prompts(root, seq_name, expression_root=expression_root, max_prompts=max_prompts)
        )
        meta["prompts_by_seq"][seq_name] = prompts
        for frame_idx, stem in enumerate(frame_stems):
            image_path = _resolve_frame_image_path(image_dir, stem)
            if image_path is None:
                continue
            detections = generator.generate_for_frame(str(image_path), prompts, frame=frame_idx)
            detector_data[f"{seq_name}/{stem}"] = [
                detection_to_json_item(det, config=config, default_score=1.0, keep_track_id=False)
                for det in detections
            ]
    return detector_data, meta


def _resolve_frame_image_path(image_dir: Path, stem: str) -> Path | None:
    for suffix in [".png", ".jpg", ".jpeg", ".bmp"]:
        path = image_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    return None
