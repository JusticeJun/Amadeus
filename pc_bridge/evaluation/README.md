# Semantic Routing Evaluation

This directory contains capability-specific offline corpora and a shared evaluator for tool-routing decisions. It does not change production routing.

Each file under `cases/` evaluates one pilot capability. New capabilities can add their own JSONL file without changing the evaluator. `expected_tools` is a list and can contain multiple capabilities. `expected_tools: null` marks an intentionally ambiguous case that is reported but excluded from precision and recall.

Cases may optionally include `tags`, `intent`, `reason`, and prior `context`. These fields document cross-cutting failure modes and preserve future context-aware evaluation without affecting the current rule matcher.

Side-effecting capabilities should emphasize hard negatives, minimal pairs,
unsupported actions, and safety boundaries. An `unsupported_action` still labels
the requested capability when the user is asking for that external action; it
does not imply that the production tool may execute it. A `planning_required`
case preserves every required capability while documenting that unconditional
fast-path execution is unsafe without dependency-aware planning.

Run the current rule-based Weather matcher against the corpus:

```bash
python tools/evaluate_semantic_routing.py
```

Use `--json` for machine-readable output or `--fail-on-mismatch` when a future routing implementation is expected to satisfy the full scored corpus.

This is a holdout evaluation asset, not classifier training data. Future training and validation data must be maintained separately to avoid measuring a model on examples it learned from.

## Label policy

- Label a capability as required when answering correctly needs current or future external data, or when an external action must be performed.
- Do not require a capability for general knowledge, metaphors, opinions, ordinary conversation, or descriptions of completed events.
- Use `context_required` when the current utterance has a defensible label only with prior conversation context.
- Use `unsupported_capability` when external data or an action is required but the currently implemented capability cannot provide it. Label the capability that would actually be needed rather than forcing the case into an existing Tool.
- Keep `expected_tools: null` and category `ambiguous` when a person cannot reasonably decide from the available context. Do not force these cases into positive or negative labels.

The report separates standard scored single-turn cases, context-required cases, unsupported-capability cases, and ambiguous cases. Overall metrics remain available, but slice metrics should be consulted before comparing routers with different context or capability support.
