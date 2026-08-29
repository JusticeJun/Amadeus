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

Compare the unchanged rule baseline, the standalone local model, and the production
hybrid on the same fixed corpus:

```bash
python tools/evaluate_semantic_routing.py --router rule
python tools/evaluate_semantic_routing.py --router ml
python tools/evaluate_semantic_routing.py --router hybrid
```

The report includes per-capability and micro/macro precision, recall, and F1 plus
human-readable false positives and false negatives. Dataset preparation and threshold
selection must not use these cases. `tools/check_semantic_dataset_leakage.py` performs
only the final normalized exact-overlap audit.

Use `--json` for machine-readable output or `--fail-on-mismatch` when a future routing implementation is expected to satisfy the full scored corpus.

This is a holdout evaluation asset, not classifier training data. Future training and validation data must be maintained separately to avoid measuring a model on examples it learned from.

## Music structured interpretation evaluation

Music-internal interpretation is measured separately from capability routing:

```bash
python tools/evaluate_music_interpretation.py
```

The prediction boundary includes the current Music routing decision and the
Music action parser, but never calls a controller or playback backend. Each
case labels one of four outcomes: `parsed`, `ambiguous`, `unsupported`, or
`rejected`. Parsed cases contain an ordered list of actions using the production
action enum and the `title`, `artist`, and `playlist` arguments.

The corpus is split into:

- `development.jsonl`: public examples for evaluator tests and future prompt or
  rule development.
- `holdout.jsonl`: the fixed comparison set for Rule, LLM-assisted, and future
  Local-ML-plus-LLM systems. Do not tune rules, prompts, or thresholds against
  individual holdout sentences. Add newly discovered field failures without
  rewriting older cases to improve a score.

Metrics have deliberately separate meanings:

- action accuracy: position-aligned action-type accuracy across expected parsed
  requests, with missing and extra actions penalized;
- entity accuracy: normalized exact match for applicable title, artist, and
  playlist slots;
- full structured-request accuracy: outcome, ordered actions, and every entity
  must all match;
- action-sequence exact accuracy: outcome and the ordered action types match;
- ambiguity and context-dependent accuracy: exact success on those slices;
- alternate-query coverage: each explicitly required canonical query is
  present after NFKC, case, and whitespace normalization;
- alternate-query boundedness: no more than four non-empty, normalized-unique
  alternatives per predicted action.

Entity normalization permits only Unicode NFKC, case folding, and whitespace
removal. The evaluator does not translate names or invent aliases. Alternate
queries never improve primary action/entity accuracy, and generating more
queries does not increase coverage beyond satisfying an explicitly labeled
requirement.

## Label policy

- Label a capability as required when answering correctly needs current or future external data, or when an external action must be performed.
- Do not require a capability for general knowledge, metaphors, opinions, ordinary conversation, or descriptions of completed events.
- Use `context_required` when the current utterance has a defensible label only with prior conversation context.
- Use `unsupported_capability` when external data or an action is required but the currently implemented capability cannot provide it. Label the capability that would actually be needed rather than forcing the case into an existing Tool.
- Keep `expected_tools: null` and category `ambiguous` when a person cannot reasonably decide from the available context. Do not force these cases into positive or negative labels.

The report separates standard scored single-turn cases, context-required cases, unsupported-capability cases, and ambiguous cases. Overall metrics remain available, but slice metrics should be consulted before comparing routers with different context or capability support.
