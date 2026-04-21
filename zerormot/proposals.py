from __future__ import annotations

from .structures import Detection


class ProposalGenerator:
    """Interface for future open-vocabulary proposal backends."""

    def generate_for_frame(self, image_path: str, prompts: list[str], frame: int) -> list[Detection]:
        raise NotImplementedError

