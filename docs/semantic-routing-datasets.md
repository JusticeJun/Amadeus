# Semantic Routing Dataset Research

The production label vocabulary remains Amadeus-owned (`weather`, `music_control`, and
`pc_control`). External intent names are evidence sources, not runtime labels. The fixed
corpora under `pc_bridge/evaluation/cases/` remain independent holdout assets. Dataset
preparation reads only normalized fingerprints for final collision removal; holdout text
and labels are not used for mapping, generation, training, validation, or thresholds.

## Dataset review

| Dataset | License | Language / scale | Structure | Decision |
|---|---|---|---|---|
| [MASSIVE 1.1](https://github.com/alexa/massive) | CC BY 4.0 | 52 languages including Korean; Korean has 16,521 utterances | 60 single intents, slots, official train/dev/test | Adopt Korean train/dev. Explicitly map seven intents; balance unmapped intents as no-match. Use source test only in the separate external evaluation. |
| [CLINC150](https://archive.ics.uci.edu/dataset/570/clinc150) | CC BY 4.0 | English; 23,700 queries | 150 single intents plus OOS; official train/validation/test | Adopt mapped intents, balanced unrelated intents, and OOS. It adds an independent taxonomy and explicit out-of-scope examples. |
| [HWU64 / NLU Evaluation Data](https://github.com/xliuhw/NLU-Evaluation-Data) | CC BY 4.0 | English; 25,716 annotated home-domain utterances | 68 scenario/intent pairs (commonly collapsed to 64 intents), entities, published cross-validation | Partial adoption as an English assistant supplement. It is SLURP-family data and semantically related to MASSIVE, so it is not counted as an independent semantic source. |
| [KoBlendX / KoMixX](https://github.com/HYU-NLP/BlendX) | GPL-2.0 repository contents | Korean translations of multi-intent ATIS, Banking77, and CLINC150 derivatives | Multi-label train/test variants; translation and generated blends | Reference and later multi-label ablation candidate. Do not mix into the first artifact because downstream source licensing and synthetic construction need a separate provenance review. |
| [3i4K](https://github.com/warnikchow/3i4k) | CC BY-SA 4.0 | Korean | Intonation-aided speech-act/intention classes; official train/validation/test | Reference-only. Useful for Korean utterance style, but taxonomy is not assistant capability routing and share-alike derivative obligations need separate review. |
| [KLUE](https://arxiv.org/abs/2105.09680) | Task-specific dataset terms | Korean | Eight NLU tasks including topic, NLI, and dialogue state tracking; no assistant intent taxonomy | Reference-only. Do not invent Tool labels from topic or dialogue-state annotations. |
| MixATIS / MixSNIPS | Common research benchmark, but redistributed dataset license is unclear | English | Synthetic concatenated multi-intent utterances | Reject for direct training until dataset-level redistribution and derivative licenses are explicit. |
| General Korean conversation corpora | Corpus-specific licenses and mostly non-intent labels | Korean conversational text | Dialogue, sentiment, or topic labels | Reference-only for style coverage. Do not invent capability labels for unrelated conversation. |

Primary sources: the MASSIVE official repository and paper, the UCI CLINC150 record,
the official NLU Evaluation Data repository, and the official BlendX repository/paper.

MASSIVE-derived rows are redistributed under CC BY 4.0. Attribute Jack FitzGerald et al.,
“MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51
Typologically-Diverse Languages” (2022), and Emanuele Bastianelli et al., “SLURP: A
Spoken Language Understanding Resource Package” (EMNLP 2020), whose English data was
used as MASSIVE seed data. The pinned upstream archive includes the full license text.

## Mapping policy

`sources.json` is the reviewable mapping and license manifest. `weather_query` maps to
`weather`; `play_music` and `music_query` map to `music_control`; volume intents map to
`pc_control`. Other intents are sampled as no-match because forcing calendar, alarms,
IoT, transport, general QA, or other unsupported capabilities into an existing Tool is
unsafe. CLINC `weather`, music/playlist/now-playing intents, and volume map explicitly;
its OOS and balanced unrelated intents map to no-match. HWU uses scenario plus intent to
avoid collisions such as `weather/query` versus `calendar/query`. External official test
partitions are evaluated separately and never merged with the Amadeus holdout.

The classifier still produces confidence for every capability. Runtime fallback promotion
is separately controlled by the capability catalog: only read-only weather is enabled.
Side-effecting music and PC predictions remain measurable but cannot reach Tool execution
until independent evidence supports a safe policy. Semantic false positives are not
corrected with capability-specific keyword or phrase predicates.

The checked-in hand-authored corpus is retained as `amadeus-hand-authored-baseline-v1`,
but is quarantined wholesale from the final prepared corpus after an independent leakage
audit found normalized overlap with the existing holdout. Selecting individual safe rows
would tune data against holdout, so no row from that provenance is reused. Multi-label
coverage is instead composed deterministically from mapped MASSIVE rows and retains both
parent IDs. Prepared rows retain source, source split, source intent,
adaptation, and tags. Runtime artifacts contain only learned features and aggregate
provenance metadata.

## Reproduction and quality gates

Run `python tools/prepare_semantic_dataset.py`. The command downloads pinned MASSIVE,
HWU64, and CLINC150 sources, verifies SHA-256 values, applies explicit mappings, balances
no-match intents deterministically, and writes train, validation, separate external-test
JSONL, and a report. It excludes exact/normalized duplicates, adjacent within-intent near
duplicates at a 0.94 sequence-similarity threshold, conflicting labels, and split overlap.
The final quality gate compares only normalized exact fingerprints against the independent
Amadeus holdout and removes collisions; it
does not expose holdout labels, use holdout sentences as generation seeds, or make model,
mapping, representation, or threshold decisions from them.

Future capabilities repeat the same process: research a licensed source, add an explicit
intent mapping, supplement missing Korean and pairwise hard negatives, run quality gates,
retrain, select thresholds on validation, then evaluate once on the independent holdout.

## Baseline and augmented corpus report

- Initial baseline: 5,402 train / 1,240 validation, using MASSIVE Korean plus 180/45
  MASSIVE-derived multi-label rows. Its production hybrid micro F1 is 0.898.
- Boundary-augmented training: 10,849 rows: MASSIVE Korean 5,217, HWU64 English 2,036,
  CLINC150 English/OOS 3,260, MASSIVE-derived multi-label 180, and Amadeus boundary
  generation 156 (69 capability-confusion, 36 reviewed weather-boundary, and 51 composed
  weather-boundary rows). Quarantined hand-authored data contributes 0%.
- Boundary-augmented validation: 2,714 rows: MASSIVE 1,201, HWU64 457, CLINC150 937,
  external-derived multi-label 45, and Amadeus-generated/reviewed 74.
- Separate external test: 4,112 rows: MASSIVE 1,381, HWU64 406, and CLINC150 2,325.
- Training labels: weather 978, music 1,714, PC 764, no-match 7,573, multi-label 180,
  and explicit hard-negative 340. Each capability pair has three generated confusion
  negatives; validation independently contains three per pair. Positive multi-label
  coverage is balanced at 60 train / 15 validation examples for each of weather+music,
  weather+PC, and music+PC.
- Quality exclusions: train 255 normalized duplicates, 6 conflicting-label rows, and 23
  near duplicates; validation 16/19/2; external test 35/0/5. Eighty-two train/validation
  collisions were excluded before boundary augmentation and 85 after it; 115
  fit/external-test collisions were excluded. Three train/Amadeus-holdout exact
  collisions were excluded. Final exact/normalized overlap is zero. Audit-only character
  trigram Jaccard found zero near overlaps at 0.85 (maximum train 0.80, validation 0.7143).
- Quarantined initial baseline: 55 train and 24 validation rows; retained for audit only.
- Production artifact remains 1,920,650 bytes with deterministic SHA-256
  `6d76206c0020b3175ecd4cba2a2735787674afaf9c23b22f169a02049d592b50`.
- The boundary-augmented research artifact is 1,864,038 bytes with SHA-256
  `9ea571ce3f75ab9d417e04100d75bb273b293c921c4f2fa18145134758b8d30c` and thresholds
  weather 0.33, music 0.65, PC 0.85. Two independent retraining runs produced the same hash.
- The capability-specific Weather predicate and media phrase patch were removed after
  ablation showed that they, rather than the classifier, accounted for the false-positive
  reduction. Production hybrid v1 therefore remains at micro F1 0.8982 on the untouched
  Amadeus holdout. Separate external-test standalone ML changes from micro P/R/F1
  0.901/0.861/0.880 and exact 0.960 to 0.905/0.858/0.881 and exact 0.960. The production
  v1 standalone baseline remains 0.916/0.444/0.598 and exact 0.897.

## Frozen multilingual sentence encoder experiment

The single sentence-level candidate is
[`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2),
an Apache-2.0 multilingual MiniLM Sentence Transformer supporting Korean among 50+
languages. It produces normalized 384-dimensional embeddings. The encoder is frozen;
three independent logistic regression heads learn `weather`, `music_control`, and
`pc_control`. Per-label F0.5 thresholds are selected on validation for conservative OOS
rejection. No capability-specific semantic rules are applied.

- Artifact: 23,441 bytes with deterministic SHA-256
  `bb3b63f18502c21857fb13bf0985eacb4f5fb52690ecba01f97e54b901e26f1e`;
  thresholds weather 0.36, music 0.52, PC 0.56. Two offline runs matched exactly.
- Validation standalone micro P/R/F1 is 0.880/0.657/0.752, exact 0.898. Weather is
  0.860/0.746/0.799. Multi-label exact is 0.022.
- Boundary Weather P/R/F1 is 0.800/1.000/0.889 with two false positives. The model keeps
  indirect clothing Weather but still routes an ordinary outdoor-study question as Weather.
- External-test standalone micro P/R/F1 is 0.870/0.718/0.787, exact 0.934. Weather is
  0.872/0.807/0.838. The TF-IDF research model remains stronger at 0.905/0.858/0.881,
  exact 0.960.
- Final one-time Amadeus holdout standalone sentence-model micro F1 is 0.414; the hybrid
  with the unchanged rule fast path and execution allow-list is 0.878 versus production
  hybrid v1 at 0.898. It is not promoted.

The frozen encoder improves semantic recall but does not provide sufficient OOS precision
or multi-label separation on the current corpus. SetFit-style contrastive encoder tuning is
a technically valid next experiment, but is not implemented in Issue #27 because the
minimal verified architecture did not justify expanding this work into a broader NLP
research project.
