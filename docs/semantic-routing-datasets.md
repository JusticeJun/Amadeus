# Semantic Routing Dataset Research

The production label vocabulary remains Amadeus-owned (`weather`, `music_control`, and
`pc_control`). External intent names are evidence sources, not runtime labels. The fixed
corpora under `pc_bridge/evaluation/cases/` are independent holdout assets and are never
read by dataset preparation, training, validation, threshold selection, or augmentation.

## Dataset review

| Dataset | License | Language / scale | Structure | Decision |
|---|---|---|---|---|
| [MASSIVE 1.1](https://github.com/alexa/massive) | CC BY 4.0 | 52 languages including Korean; Korean has 16,521 utterances | 60 single intents, slots, official train/dev/test | Adopt Korean train/dev. Explicitly map seven intents; balance unmapped intents as no-match. Keep source test unused. |
| [CLINC150](https://archive.ics.uci.edu/dataset/570/clinc150) | CC BY 4.0 | English; 23,700 queries | 150 single intents plus OOS; train/validation/test | Partial candidate for English code-switch and OOS ablation. Not in the first corpus because MASSIVE already supplies assistant-domain Korean and broad negatives. |
| [HWU64 / NLU Evaluation Data](https://github.com/xliuhw/NLU-Evaluation-Data) | CC BY 4.0 | English; 25,716 annotated home-domain utterances | 64 intents, entities, published cross-validation | Reference/possible English supplement. Requires careful mapping and adds little Korean coverage. |
| [KoBlendX / KoMixX](https://github.com/HYU-NLP/BlendX) | GPL-2.0 repository contents | Korean translations of multi-intent ATIS, Banking77, and CLINC150 derivatives | Multi-label train/test variants; translation and generated blends | Reference and later multi-label ablation candidate. Do not mix into the first artifact because downstream source licensing and synthetic construction need a separate provenance review. |
| MixATIS / MixSNIPS | Common research benchmark, but redistributed dataset license is unclear | English | Synthetic concatenated multi-intent utterances | Reject for direct training until dataset-level redistribution and derivative licenses are explicit. |
| General Korean conversation corpora | Corpus-specific licenses and mostly non-intent labels | Korean conversational text | Dialogue, sentiment, or topic labels | Reference-only for style coverage. Do not invent capability labels for unrelated conversation. |

Primary sources: the MASSIVE official repository and paper, the UCI CLINC150 record,
the official NLU Evaluation Data repository, and the official BlendX repository/paper.

MASSIVE-derived rows are redistributed under CC BY 4.0. Attribute Jack FitzGerald et al.,
“MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51
Typologically-Diverse Languages” (2022), and Emanuele Bastianelli et al., “SLURP: A
Spoken Language Understanding Resource Package” (EMNLP 2020), whose English data was
used as MASSIVE seed data. The pinned upstream archive includes the full license text.

## MASSIVE mapping policy

`sources.json` is the reviewable mapping and license manifest. `weather_query` maps to
`weather`; `play_music` and `music_query` map to `music_control`; volume intents map to
`pc_control`. Other intents are sampled as no-match because forcing calendar, alarms,
IoT, transport, general QA, or other unsupported capabilities into an existing Tool is
unsafe. MASSIVE test remains unused rather than being merged with Amadeus evaluation.

The classifier still produces confidence for every capability. Runtime fallback promotion
is separately controlled by the capability catalog: the first artifact enables ML fallback
only for read-only weather. Side-effecting music and PC predictions remain measurable but
cannot reach Tool execution until independent evidence supports a safe policy.

The checked-in hand-authored corpus is retained as `amadeus-hand-authored-baseline-v1`,
but is quarantined wholesale from the final prepared corpus after an independent leakage
audit found normalized overlap with the existing holdout. Selecting individual safe rows
would tune data against holdout, so no row from that provenance is reused. Multi-label
coverage is instead composed deterministically from mapped MASSIVE rows and retains both
parent IDs. Prepared rows retain source, source split, source intent,
adaptation, and tags. Runtime artifacts contain only learned features and aggregate
provenance metadata.

## Reproduction and quality gates

Run `python tools/prepare_semantic_dataset.py`. The command downloads the pinned MASSIVE
1.1 archive, verifies both archive and Korean-member SHA-256 values, excludes its test
partition, applies the explicit mapping, balances no-match intents deterministically,
and writes prepared train/validation JSONL plus a report. It excludes label-conflicting
normalized duplicates and split overlap. The final quality gate compares only normalized
exact fingerprints against the independent Amadeus holdout and removes collisions; it
does not expose holdout labels, use holdout sentences as generation seeds, or make model,
mapping, representation, or threshold decisions from them.

Future capabilities repeat the same process: research a licensed source, add an explicit
intent mapping, supplement missing Korean and pairwise hard negatives, run quality gates,
retrain, select thresholds on validation, then evaluate once on the independent holdout.

## First prepared corpus and baseline report

- Training: 5,402 rows (5,222 direct MASSIVE Korean, 180 MASSIVE-derived multi-label).
- Validation: 1,240 rows (1,195 direct, 45 derived multi-label).
- Training labels: weather 645, music 890, PC 369, no-match 3,678; labels overlap on
  multi-label rows, so totals do not equal the row count.
- Quality exclusions: 239 normalized duplicates and 4 conflicting-label rows in train,
  16 duplicates in validation, 84 train/validation collisions, and 3 exact holdout
  collisions. Final normalized overlap is zero for train/validation and both versus holdout.
- Quarantined initial baseline: 55 train and 24 validation rows; retained for audit only.
- Artifact: 1,920,650 bytes, 25,461 retained features, deterministic SHA-256
  `6d76206c0020b3175ecd4cba2a2735787674afaf9c23b22f169a02049d592b50`.
- Validation-selected thresholds: weather 0.22, music 0.71, PC 0.64. Runtime risk policy
  independently blocks ML fallback promotion for music and PC.
- Fixed evaluation: rule micro P/R/F1 0.933/0.826/0.876; standalone ML
  0.716/0.576/0.639; production hybrid 0.864/0.935/0.898. Hybrid music is 1.0 F1 and PC
  is 0.992 F1 with zero false positives; weather recall improves from 0.650 to 0.900 while
  precision falls from 0.825 to 0.727. This weather false-positive tradeoff and weak
  standalone ML result are explicit limitations, not grounds for holdout tuning.
