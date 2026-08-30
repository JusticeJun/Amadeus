# Architecture

Amadeus는 자연어 상호작용, 외부 Tool, 캐릭터, 음성, 물리 하드웨어를 결합한 데스크톱 AI assistant다.

각 계층은 language understanding, planning, execution, conversation, hardware가 독립적으로 발전할 수 있도록 분리한다.

## System Overview

목표 구조:

User Input
  ↓
Semantic Understanding
  ↓
Structured Intent / Actions
  ↓
Planner (when required)
  ↓
Tool Execution
  ↓
Verified Results
  ↓
Conversation / Character
  ↓
TTS + Emotion
  ↓
Serial Protocol
  ↓
ESP32-S3 / Hardware

모든 요청이 모든 계층을 거칠 필요는 없다. 단순한 요청은 가장 짧고 결정적인 경로를 사용하고, 복잡한 semantic processing이나 planning은 필요한 경우에만 사용한다.

## Semantic Understanding

Semantic Understanding은 자연어를 검증 가능한 structured intent/action으로 변환한다.

다음 방식을 조합할 수 있다.

* deterministic rules — 단순하고 신뢰할 수 있는 fast path
* local ML — 저지연 semantic classification/interpretation
* LLM — 복잡하거나 모호한 자연어 해석

장기적으로는 local method가 확신할 수 있는 요청을 우선 처리하고 필요한 경우에만 LLM을 사용하는 hybrid 구조를 지향한다.

Semantic interpretation은 실행과 분리한다. LLM이나 classifier는 사용자의 의도를 해석할 수 있지만 외부 action의 성공 여부를 결정하지 않는다.

### Local ML semantic routing

Production routing preserves the stable `RoutingRequest -> RouteDecision -> ToolExecutor`
boundary. A deterministic rule router handles narrow, verified fast paths and conditional
planning guards. When it returns no match, a versioned local character n-gram TF-IDF,
one-vs-rest logistic model produces multi-label capability probabilities. Capability
definitions, side-effect risk, and ML-fallback eligibility are maintained independently
from concrete Tool classes.

Training is offline and deterministic. The provenance-aware train/validation corpus and
threshold selection are separate from the fixed evaluation corpus. Runtime loads a JSON
artifact and performs inference only. The first artifact promotes only read-only weather
fallback predictions; music and PC confidence remains evaluable but cannot cause a side
effect through ML fallback. Low-confidence predictions become an empty RouteDecision.
An optional future LLM semantic fallback may consume that no-match state, but LLM
provider/model routing remains a separate responsibility.

A research path can replace lexical TF-IDF features with a frozen multilingual sentence
encoder and a lightweight one-vs-rest linear classifier. Confidence thresholds are chosen
only from validation data, and no-match means that no capability clears its threshold.
This representation remains separate from execution policy: the capability catalog still
controls whether an accepted prediction may reach a side-effecting Tool. Research models
must outperform the production boundary without capability-specific semantic predicates
before promotion.

## LLM Routing

LLM을 사용하는 계층은 특정 provider나 model에 직접 결합하지 않는다.

LLM Router는 다음 요소를 기반으로 적절한 provider/model을 선택할 수 있다.

* task와 complexity
* 측정된 quality
* latency
* token/cost
* quota
* provider availability

Provider fallback은 timeout, rate limit, unavailable 등의 실패를 처리한다.

Semantic Routing은 필요한 capability를 결정하고, LLM Routing은 LLM이 필요한 작업에서 사용할 model/provider를 결정한다. 두 책임은 분리한다.

## Structured Actions and Planner

자연어 해석 결과는 외부 side effect 전에 validated structured action으로 변환한다.

Capability는 단일 action 또는 단순한 same-capability ordered sequence를 가질 수 있다.

Planner는 다음처럼 action 간 dependency가 필요한 경우에만 사용한다.

* cross-capability workflow
* conditional execution
* dependency ordering
* 복잡한 multi-step request

단순한 single-tool 요청이나 평면적인 same-tool sequence는 일반 Planner를 요구하지 않는다.

## Tools and Execution

외부 capability는 독립적인 Tool로 구현한다.

Tool은:

1. validated structured input을 받는다.
2. backend와 상호작용한다.
3. 가능한 경우 실제 결과를 검증한다.
4. compact structured result를 반환한다.

Tool은 Chris의 최종 대사를 생성하지 않는다.

LLM output은 외부 상태의 authoritative source가 아니다. Side effect는 가능한 경우 observable backend state로 검증하며, live data 역시 실제 Tool 결과 없이 조회된 사실처럼 표현하지 않는다.

새 capability는 가능한 한 기존 Tool과 독립적으로 추가한다.

세부 Tool 구조와 설정은 `tools.md`에서 관리한다.

## Music Known-Item Resolution

Music의 known-item 재생 경로는 다음 책임 경계를 따른다.

Natural language → MusicAction → Catalog/Library Retrieval → Track Resolver
→ optional Semantic Matcher → Resolved MusicItem → Playback → Verification

`MusicAction`은 사용자의 action과 title/artist surface를 보존한다. Retrieval은
Apple Music catalog와 개인 library의 실제 metadata만 후보로 제공한다. Track
Resolver는 정규화된 title과 artist evidence를 먼저 평가하며 search rank, 같은
artist, 또는 LLM이 생성한 query hint만으로 explicit-title 요청을 승인하지 않는다.

초기 catalog 검색이 비었을 때만 language/catalog query rewriter를 한 번 호출한다.
Rewriter는 structured artist/title surface를 입력받아 bounded structured variants를
생성하며, 모든 variant는 Apple Music native search를 다시 거친다. 실제 catalog
결과를 ID로 합친 하나의 candidate pool만 Track Resolver에 전달한다.

Lexical evidence만으로 결정할 수 없을 때만 Semantic Matcher가 이 실제 후보 집합에서
bounded candidate index를 선택할 수 있다. Provider 오류, 잘못된 index, ambiguity,
no-match는 side effect 없이 종료한다. Generated variant 자체는 resolution evidence가
아니며 resolver나 playback으로 직접 전달되지 않는다.

Resolution 이후에는 canonical `MusicItem`의 catalog id와 metadata만 playback
backend로 전달한다. Playback failure는 candidate resolution을 다시 실행하지 않으며,
성공은 별도의 player-state verification으로 확인한다. Transport, artist, personal
playlist action은 각각의 source of truth와 전용 경로를 사용한다.

## Conversation and Character

Conversation 계층은 다음을 결합한다.

* bounded conversation context
* verified Tool results
* character instructions

Character LLM은 이를 바탕으로 Chris의 최종 response와 emotion을 생성한다.

Character는 personality와 expression을 담당하지만 검증된 system state를 덮어쓰거나 실행되지 않은 action, 확인되지 않은 live data를 사실처럼 만들 수 없다.

## Speech and Hardware

PC bridge는 최종 response를 TTS로 변환하고 character state/emotion을 serial protocol로 ESP32에 전달한다.

ESP32는 display animation, 향후 servo 등 실시간 physical presentation을 담당한다.

하드웨어 세부사항은 `hardware.md`, `wiring.md`, `serial-protocol.md`에서 관리한다.

## Evaluation

Interpretation, execution, response validation은 가능한 한 독립적으로 평가한다.

주요 평가 계층:

* semantic routing / interpretation
* structured action execution with fake backends
* execution and data truthfulness
* representative E2E production validation

고정 holdout은 semantic architecture나 model 비교를 위해 보존한다.

필요한 경우 accuracy뿐 아니라 latency, resource/token usage, external LLM dependency도 함께 측정한다.

## Design Principles

* Probabilistic interpretation과 deterministic execution을 분리한다.
* 외부 side effect와 live data는 가능한 경우 검증한다.
* 단순한 요청에는 단순하고 빠른 경로를 사용한다.
* 비싼 semantic processing은 필요한 경우에만 사용한다.
* Tool은 capability별로 독립적으로 유지한다.
* 특정 LLM provider/model에 강하게 결합하지 않는다.
* Planner는 실제 dependency가 필요해질 때 확장한다.
* Rule, local ML, LLM을 교체하거나 조합할 수 있도록 stable boundary를 유지한다.
* 현재 프로젝트 규모보다 앞선 과도한 abstraction을 만들지 않는다.
* 실제 실패 사례를 regression/evaluation 자산으로 활용한다.

## Related Documentation

* `roadmap.md` — 진행 상황과 향후 milestone
* `tools.md` — Tool 및 외부 capability
* `hardware.md` — 검증된 hardware 구성
* `wiring.md` — hardware 연결
* `serial-protocol.md` — PC ↔ ESP32 통신
* `troubleshooting.md` — 알려진 개발 및 hardware 문제
