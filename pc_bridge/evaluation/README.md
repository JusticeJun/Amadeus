# Semantic Routing Evaluation

This directory contains capability-specific offline corpora and a shared evaluator for tool-routing decisions. It does not change production routing.

Each file under `cases/` evaluates one pilot capability. New capabilities can add their own JSONL file without changing the evaluator. `expected_tools` is a list and can contain multiple capabilities. `expected_tools: null` marks an intentionally ambiguous case that is reported but excluded from precision and recall.

Cases may optionally include `tags`, `intent`, `reason`, and prior `context`. These fields document cross-cutting failure modes and preserve future context-aware evaluation without affecting the current rule matcher.

Run the current rule-based Weather matcher against the corpus:

```bash
python tools/evaluate_semantic_routing.py
```

Use `--json` for machine-readable output or `--fail-on-mismatch` when a future routing implementation is expected to satisfy the full scored corpus.

This is a holdout evaluation asset, not classifier training data. Future training and validation data must be maintained separately to avoid measuring a model on examples it learned from.
