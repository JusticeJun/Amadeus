from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import unicodedata

from .base import CapabilityMatch, RouteDecision, RoutingRequest


ARTIFACT_SCHEMA = 1


class ModelArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class LinearLabelModel:
    bias: float
    threshold: float
    weights: dict[str, float]


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def character_ngrams(text: str, minimum: int, maximum: int) -> Counter[str]:
    normalized = f" {normalize_text(text)} "
    return Counter(
        normalized[start:start + size]
        for size in range(minimum, maximum + 1)
        for start in range(max(0, len(normalized) - size + 1))
    )


class LocalMlSemanticRouter:
    """Multi-label local inference over a versioned, offline-trained artifact."""

    def __init__(self, artifact_path: Path) -> None:
        try:
            raw = artifact_path.read_bytes()
            data = json.loads(raw)
            if data.get("schema_version") != ARTIFACT_SCHEMA:
                raise ValueError("unsupported schema version")
            ngrams = data["preprocessing"]["character_ngrams"]
            self._minimum = int(ngrams["minimum"])
            self._maximum = int(ngrams["maximum"])
            self._idf = {str(key): float(value) for key, value in data["idf"].items()}
            self._labels = tuple(str(label) for label in data["labels"])
            models = data["models"]
            self._models = {
                label: LinearLabelModel(
                    float(models[label]["bias"]),
                    float(models[label]["threshold"]),
                    {str(key): float(value) for key, value in models[label]["weights"].items()},
                )
                for label in self._labels
            }
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ModelArtifactError(f"invalid local semantic model: {artifact_path}") from exc

    def route(self, request: RoutingRequest) -> RouteDecision:
        features = self._features(request)
        matches: list[CapabilityMatch] = []
        for label in self._labels:
            model = self._models[label]
            score = model.bias + sum(
                value * model.weights.get(feature, 0.0)
                for feature, value in features.items()
            )
            confidence = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score))))
            if confidence >= model.threshold:
                matches.append(CapabilityMatch(label, confidence))
        return RouteDecision(tuple(matches))

    def _features(self, request: RoutingRequest) -> dict[str, float]:
        text = request.text
        if request.history:
            recent = request.history[-2:]
            text = " ".join(message.content for message in recent) + " [current] " + text
        counts = character_ngrams(text, self._minimum, self._maximum)
        weighted = {
            feature: (1.0 + math.log(count)) * self._idf[feature]
            for feature, count in counts.items()
            if feature in self._idf
        }
        norm = math.sqrt(sum(value * value for value in weighted.values())) or 1.0
        return {feature: value / norm for feature, value in weighted.items()}


def artifact_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

