"""Compare semantic routers for one utterance without executing any Tool."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import warnings

import numpy as np


BRIDGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BRIDGE_ROOT))

from app.pc_control import default_app_registry  # noqa: E402
from app.routing import CapabilityMatch, RouteDecision, RoutingRequest  # noqa: E402
from app.routing.capabilities import CAPABILITIES  # noqa: E402
from app.routing.composite import (  # noqa: E402
    CapabilityFilterSemanticRouter,
    PlanningGuardSemanticRouter,
)
from app.routing.defaults import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    create_default_semantic_router,
    create_rule_based_semantic_router,
)
from app.routing.fallback import NoMatchFallbackSemanticRouter  # noqa: E402
from app.routing.local_ml import LocalMlSemanticRouter  # noqa: E402
from app.routing.music_control_rules import detect_conditional_music_planning  # noqa: E402
from app.routing.pc_control_rules import detect_conditional_pc_planning  # noqa: E402


DEFAULT_SETFIT_MODEL = BRIDGE_ROOT / "research_artifacts" / "semantic-router-v4-setfit"


def local_ml_scores(router: LocalMlSemanticRouter, request: RoutingRequest) -> dict[str, float]:
    """Return pre-threshold probabilities from the loaded production artifact."""
    features = router._features(request)  # noqa: SLF001 - read-only diagnostic surface
    result = {}
    for label in router._labels:  # noqa: SLF001
        model = router._models[label]  # noqa: SLF001
        logit = model.bias + sum(
            value * model.weights.get(feature, 0.0)
            for feature, value in features.items()
        )
        result[label] = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))
    return result


class ReadOnlySetFitRouter:
    def __init__(self, model_path: Path) -> None:
        from setfit import SetFitModel
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self._model = SetFitModel.from_pretrained(model_path, local_files_only=True)
        metadata = json.loads((model_path / "amadeus_metadata.json").read_text(encoding="utf-8"))
        self.labels = tuple(str(label) for label in metadata["labels"])
        self.thresholds = {
            label: float(metadata["thresholds"][label]) for label in self.labels
        }

    def scores(self, request: RoutingRequest) -> dict[str, float]:
        text = request.text
        if request.history:
            recent = request.history[-2:]
            text = " ".join(message.content for message in recent) + " [current] " + text
        probabilities = np.asarray(
            self._model.predict_proba([text], show_progress_bar=False),
        ).reshape(-1)
        return dict(zip(self.labels, (float(value) for value in probabilities)))

    def route_from_scores(self, scores: dict[str, float]) -> RouteDecision:
        return RouteDecision(tuple(
            CapabilityMatch(label, scores[label])
            for label in self.labels
            if scores[label] >= self.thresholds[label]
        ))


def decision_dict(decision: RouteDecision) -> dict[str, object]:
    return {
        "matches": [
            {"capability": match.capability, "confidence": match.confidence}
            for match in decision.matches
        ],
        "no_match": not decision.matches,
        "planning_required": decision.planning_required,
        "planning_reason": decision.planning_reason,
    }


def inspect(text: str, setfit_model_path: Path = DEFAULT_SETFIT_MODEL) -> dict[str, object]:
    request = RoutingRequest(text)
    apps = default_app_registry()
    rule_router = create_rule_based_semantic_router(apps)
    rule_decision = rule_router.route(request)

    tfidf_router = LocalMlSemanticRouter(DEFAULT_MODEL_PATH)
    tfidf_scores = local_ml_scores(tfidf_router, request)
    tfidf_thresholds = {
        label: tfidf_router._models[label].threshold  # noqa: SLF001
        for label in tfidf_router._labels  # noqa: SLF001
    }
    tfidf_standalone = tfidf_router.route(request)

    setfit_router = ReadOnlySetFitRouter(setfit_model_path)
    setfit_scores = setfit_router.scores(request)
    setfit_standalone = setfit_router.route_from_scores(setfit_scores)

    tfidf_hybrid = create_default_semantic_router(apps).route(request)
    allowed = frozenset(item.name for item in CAPABILITIES if item.ml_fallback_enabled)
    setfit_fallback = PlanningGuardSemanticRouter(
        CapabilityFilterSemanticRouter(FixedDecisionRouter(setfit_standalone), allowed),
        (detect_conditional_pc_planning, detect_conditional_music_planning),
    )
    setfit_hybrid = NoMatchFallbackSemanticRouter(rule_router, setfit_fallback).route(request)

    return {
        "input": text,
        "rule": decision_dict(rule_decision),
        "tfidf_v1": {
            "raw_capability_scores": tfidf_scores,
            "thresholds": tfidf_thresholds,
            "accepted_labels": sorted(tfidf_standalone.required_capabilities),
            "standalone": decision_dict(tfidf_standalone),
        },
        "setfit": {
            "raw_capability_scores": setfit_scores,
            "thresholds": setfit_router.thresholds,
            "accepted_labels": sorted(setfit_standalone.required_capabilities),
            "standalone": decision_dict(setfit_standalone),
        },
        "tfidf_hybrid": decision_dict(tfidf_hybrid),
        "setfit_hybrid": decision_dict(setfit_hybrid),
        "side_effects": "none; routing inference only",
    }


class FixedDecisionRouter:
    """Immutable one-request router used to compose the existing hybrid policy."""

    def __init__(self, decision: RouteDecision) -> None:
        self._decision = decision

    def route(self, request: RoutingRequest) -> RouteDecision:
        return self._decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="+", help="one utterance to inspect")
    parser.add_argument("--setfit-model", type=Path, default=DEFAULT_SETFIT_MODEL)
    args = parser.parse_args()
    print(json.dumps(inspect(" ".join(args.text), args.setfit_model), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
