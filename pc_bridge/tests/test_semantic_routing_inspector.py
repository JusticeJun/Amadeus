from __future__ import annotations

from app.routing import CapabilityMatch, RouteDecision
from tools.inspect_semantic_routing import decision_dict


def test_inspector_serializes_no_match_and_planning_state() -> None:
    assert decision_dict(RouteDecision()) == {
        "matches": [],
        "no_match": True,
        "planning_required": False,
        "planning_reason": "",
    }
    assert decision_dict(RouteDecision(
        (CapabilityMatch("weather", 0.75),),
        planning_required=True,
        planning_reason="needs dependency planning",
    )) == {
        "matches": [{"capability": "weather", "confidence": 0.75}],
        "no_match": False,
        "planning_required": True,
        "planning_reason": "needs dependency planning",
    }
