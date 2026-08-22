# Semantic Routing Evaluation

This directory contains an offline corpus and evaluator for tool-routing decisions. It does not change production routing.

Each JSONL case includes an ID, Korean input text, expected tool names, and a semantic boundary category. `expected_tools: null` marks an intentionally ambiguous case that is reported but excluded from precision and recall.

Run the current rule-based Weather matcher against the corpus:

```bash
python tools/evaluate_semantic_routing.py
```

Use `--json` for machine-readable output or `--fail-on-mismatch` when a future routing implementation is expected to satisfy the full scored corpus.
