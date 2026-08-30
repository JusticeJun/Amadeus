from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Protocol, Sequence

from .base import CapabilityMatch, RouteDecision, RoutingRequest


ARTIFACT_SCHEMA = 1


class SentenceEncoder(Protocol):
    def encode(
        self,
        sentences: str | Sequence[str],
        *,
        normalize_embeddings: bool,
        show_progress_bar: bool = False,
    ): ...


@dataclass(frozen=True)
class LinearSentenceLabelModel:
    intercept: float
    threshold: float
    coefficients: tuple[float, ...]


class SentenceMlSemanticRouter:
    """Multi-label routing over a fixed multilingual sentence encoder."""

    def __init__(self, artifact_path: Path, encoder: SentenceEncoder | None = None) -> None:
        data = json.loads(artifact_path.read_text(encoding="utf-8"))
        if data.get("schema_version") != ARTIFACT_SCHEMA:
            raise ValueError(f"unsupported sentence semantic artifact: {artifact_path}")
        self._encoder_name = str(data["encoder"]["name"])
        self._dimension = int(data["encoder"]["dimension"])
        self._labels = tuple(str(label) for label in data["labels"])
        self._models = {
            label: LinearSentenceLabelModel(
                intercept=float(data["models"][label]["intercept"]),
                threshold=float(data["models"][label]["threshold"]),
                coefficients=tuple(float(value) for value in data["models"][label]["coefficients"]),
            )
            for label in self._labels
        }
        if any(len(model.coefficients) != self._dimension for model in self._models.values()):
            raise ValueError(f"invalid sentence semantic dimensions: {artifact_path}")
        self._encoder = encoder or self._load_encoder()

    def _load_encoder(self) -> SentenceEncoder:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence semantic routing requires the semantic-routing optional dependency"
            ) from exc
        return SentenceTransformer(self._encoder_name, local_files_only=True)

    def route(self, request: RoutingRequest) -> RouteDecision:
        text = request.text
        if request.history:
            recent = request.history[-2:]
            text = " ".join(message.content for message in recent) + " [current] " + text
        encoded = self._encoder.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vector = tuple(float(value) for value in encoded)
        if len(vector) != self._dimension:
            raise ValueError("sentence encoder returned an unexpected embedding dimension")
        matches = []
        for label in self._labels:
            model = self._models[label]
            logit = model.intercept + sum(
                value * weight for value, weight in zip(vector, model.coefficients)
            )
            confidence = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))
            if confidence >= model.threshold:
                matches.append(CapabilityMatch(label, confidence))
        return RouteDecision(tuple(matches))
