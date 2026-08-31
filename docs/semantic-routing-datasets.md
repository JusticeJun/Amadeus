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
a technically valid final experiment.

## SetFit contrastive sentence encoder experiment

The final Issue #27 model experiment uses SetFit 1.1.3 with Transformers 4.x and the same
Apache-2.0 multilingual MiniLM encoder. Standard `CosineSimilarityLoss` contrastive tuning
runs for one epoch over 21,698 pairs (one iteration, batch 16, learning rate 2e-5). SetFit's
supported `one-vs-rest` multi-target strategy trains three logistic heads. Per-label F0.5
thresholds are selected only from validation: Weather 0.86, Music 0.81, and PC 0.91.
No handcrafted capability semantic filter is present.

- Validation standalone micro P/R/F1 is 0.940/0.799/0.864, exact 0.946. Per-label F1 is
  Weather 0.903, Music 0.862, and PC 0.805. Of 2,068 OOS rows, 2,033 are rejected and 35
  are false promotions.
- External official test micro P/R/F1 is 0.922/0.877/0.899, exact 0.967. Per-label F1 is
  Weather 0.929, Music 0.895, and PC 0.857.
- On the independent Weather boundary, Weather P/R/F1 is 1.000/1.000/1.000 with zero
  Weather false positives. Two negative rows receive Music predictions and one positive
  row also receives PC; these cannot execute through the unchanged ML allow-list.
- Independent multi-label micro P/R/F1 is 1.000/0.234/0.379 with exact 0.000. Pair exact
  match is zero for Weather+Music, Weather+PC, and Music+PC. This remains a decisive
  limitation despite improved single-capability separation.
- The final untouched Amadeus holdout standalone micro P/R/F1 is 0.742/0.605/0.667,
  exact 0.624. Hybrid rule fast path plus unchanged Weather-only ML promotion reaches
  0.880/0.963/0.920, exact 0.880, versus production TF-IDF v1 hybrid F1 0.898.
- Warm CPU latency is 41.9 ms mean and 47.5 ms p95. The saved model is 487,738,340 bytes,
  including a 6,608-byte classifier head, with directory fingerprint
  `39958d8cbc76726af8d47713510e336cdf84ff64d8e89d97a72a06120b145fa6`.
- Exact/normalized and audit-only near-overlap remain zero. Training, threshold selection,
  and configuration use only development train/validation; external and independent
  corpora are evaluation-only.

SetFit improves the semantic Weather boundary and overall hybrid score, but it does not
solve multi-label routing and is roughly 5x slower than the frozen encoder while producing
a much larger artifact. It is not promoted. Issue #27 stops model exploration here and
retains production TF-IDF v1, its documented lexical limitations, and the separate
Music/PC ML-only execution block.

## Controlled SetFit multi-label coverage ablation

A single follow-up ablation tests whether composition scarcity caused the SetFit
multi-label failure. The baseline 10,849-row train split and its 180 multi-label rows are
preserved. A separate research split adds 720 deterministic Korean clause compositions:
240 for each capability pair and 180 for each of explicit+explicit, implicit+explicit,
explicit+implicit, and implicit+implicit. The resulting train split has 11,569 rows and
900 multi-label rows, with 300 per pair. Parents come only from existing provenance-aware
single-label development rows; parent IDs, connector, order, and composition type are
recorded. Frozen manual regression utterances are excluded and are not generation seeds.
Validation and external files are byte-identical to the baseline.

All model settings remain fixed. Validation-selected thresholds change naturally from
Weather/Music/PC 0.86/0.81/0.91 to 0.95/0.80/0.74; no threshold is manually adjusted.
Augmented-minus-baseline results on identical slices are:

- Validation all: micro F1 +0.0291 and exact +0.0029; single-label F1 +0.0103. Multi-label
  recall/F1/exact improve by +0.3111/+0.3431/+0.0667. OOS false promotions increase
  from 35 to 42 (+7).
- External official test: micro precision/recall/F1 change by -0.0179/-0.0056/-0.0115
  and exact by -0.0029. Single-label F1 changes by -0.0067. OOS false promotions increase
  from 50 to 58 (+8).
- Independent holdout standalone: micro F1 +0.0273 but exact -0.0113. Single-label F1
  improves +0.0229. Multi-label recall/F1/exact improve by +0.1489/+0.1745/+0.0435,
  while OOS exact rejection drops 0.62 to 0.53 and false promotions rise 38 to 47 (+9).
- Pair exact remains 0 for Weather+Music and Weather+PC. Music+PC improves from 0 to 1
  on its single holdout case. Weather boundary metrics and production-policy hybrid
  metrics are unchanged; Weather itself remains P/R/F1 1.0/1.0/1.0 with zero Weather FP.
- Frozen manual accepted labels remain correct for all three single-intent cases. None of
  the three manual multi-label cases becomes exact: Weather+Music still accepts Music,
  Weather+PC changes from no labels to PC only, and Music+PC remains below both thresholds.
- Same-process warm CPU latency is effectively unchanged (8.41 ms baseline vs 8.05 ms
  augmented mean). Artifact size changes from 487,738,340 to 487,738,413 bytes; the head
  remains 6,608 bytes. Two augmented retraining runs produced the same canonical artifact
  SHA-256 `0b45259e379743f71f88810120045c92adfc2c63f10696f976ec7acaea4fdc87`.

The hypothesis is partially supported: additional composition coverage materially raises
multi-label recall but does not solve two of three pair types, and it weakens OOS rejection
and external precision/generalization. This is a Pareto trade-off, not a successful
replacement candidate. Both research artifacts are retained for user review; production
selection remains open and production TF-IDF/routing policy is unchanged.

## Conversationally balanced SetFit experiment

The final controlled corpus experiment starts from the 11,569-row positive-only research
split and adds reviewed Amadeus semantic families rather than another model or parameter
change. Each added row records orthogonal `interaction`, `request_form`, `routing_role`,
`domains`, `composition`, and `ambiguity` metadata. The taxonomy distinguishes ordinary
conversation, factual questions, emotional statements, observations, domain no-action,
lexical traps, conservative ambiguity, full multi-label, partial multi-label, and neither-
actionable composition. These fields describe the routing evidence without changing the
runtime capability vocabulary.

The train split adds 720 rows: 180 conversational no-tool rows, 240 matched single-domain
rows (60 Weather, 90 Music, and 90 PC), and 300 pairwise rows. Each capability pair has 28
full multi-label, 22 left-only, 22 right-only, 16 neither, and 12 ambiguous examples.
Validation adds 270 independently worded rows: 60 conversational, 90 matched single-domain,
and 120 pairwise. The resulting research corpus has 12,289 train and 2,984 validation rows.
The pairwise rows include integrated requests and matched partial/no-action counterexamples;
they do not consist solely of two positive parent clauses joined by a connector.

Quality gates found zero normalized duplicates or label conflicts in the additions, zero
character-trigram near duplicates at the audit-only 0.85 threshold within either split or
between train and validation, and zero exact overlap with the baseline train, validation,
or external partitions. Normalized exact and the same heuristic near-overlap checks against
the independent holdout are zero; maximum similarities are 0.4545 for train and 0.2273 for
validation. Frozen manual normalized overlap is zero. These checks support process and
provenance isolation, not a claim that semantic leakage is impossible.

The model configuration remains SetFit 1.1.3 with multilingual MiniLM, cosine similarity
loss, one epoch, batch 16, one iteration, learning rate 2e-5, seed 1729, and the supported
one-vs-rest logistic head. Validation F0.5 selection produces thresholds Weather 0.70,
Music 0.85, and PC 0.78. Compared with the original SetFit candidate:

- Original validation micro F1/exact is 0.864/0.946 with 35 no-match false promotions;
  balanced is 0.905/0.954 with 40.
- Original external micro precision/recall/F1/exact is 0.922/0.877/0.899/0.967 with 50
  no-match false promotions; balanced is 0.898/0.880/0.889/0.964 with 64.
- Original independent standalone F1/exact is 0.667/0.624 with 38 no-match false
  promotions; balanced is 0.743/0.669 with 35.
- Independent multi-label precision/recall/F1/exact improves from
  1.000/0.234/0.379/0.000 to 1.000/0.638/0.779/0.348. Weather+Music and Weather+PC exact
  become 0.286 and 0.429; the single Music+PC holdout case remains non-exact.
- Weather boundary classification remains P/R/F1 1.0/1.0/1.0 with zero Weather false
  positives. Overall boundary exact improves because unrelated secondary predictions are
  reduced.
- On the reviewed development slice used during threshold selection, balanced full-
  multi-label exact is 0.533, partial exact is 0.926, and neither-actionable exact is 1.0.
  These are development diagnostics, not independent generalization results.
- Five of six frozen manual cases are exact. The Music+PC case accepts PC only because its
  Music score 0.807 remains below the automatically selected 0.85 threshold.
- The artifact is 487,738,601 bytes with a 6,608-byte classifier head. Two runs produce
  the same canonical SHA-256 `dd18b54aef3458b6850ce232a722f078ca6796461da89d7ec8e4727c602a799a`;
  same-process warm CPU inference is approximately 7.7 ms mean and 8.3 ms p95.

The corpus-quality hypothesis is partially supported. Balanced semantic evidence greatly
improves compositional and independent-holdout behavior while recovering no-tool margin on
the known frozen outdoor example, but it does not preserve external rejection: external
false promotions rise by 14 and precision falls by about 0.024 from the original SetFit
candidate. The balanced artifact is therefore research evidence, not a production
replacement. Production TF-IDF and the Weather-only ML execution allow-list remain
unchanged. The separate fallback composition policy also remains unchanged: a rule match
prevents ML from supplementing a missing capability, while unconditional rule/ML union
would introduce unsafe secondary promotions.
