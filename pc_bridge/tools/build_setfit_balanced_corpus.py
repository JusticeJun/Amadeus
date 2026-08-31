"""Build the final conversationally balanced SetFit research corpus."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import unicodedata


BRIDGE_ROOT = Path(__file__).resolve().parents[1]
BASE_DATA = BRIDGE_ROOT / "research_data" / "semantic-routing-setfit-multilabel"
DEFAULT_OUTPUT = BRIDGE_ROOT / "research_data" / "semantic-routing-setfit-balanced"
SOURCE = "amadeus-reviewed-semantic-routing-v1"
CAPABILITIES = ("weather", "music_control", "pc_control")
PAIR_SPECS = (
    ("weather", "music_control"),
    ("weather", "pc_control"),
    ("music_control", "pc_control"),
)
CONVERSATION_COUNTS = {"train": 180, "validation": 60}
SINGLE_COUNTS = {
    "train": {"weather": 60, "music_control": 90, "pc_control": 90},
    "validation": {"weather": 24, "music_control": 33, "pc_control": 33},
}
PAIR_COUNTS = {
    "train": {
        "full_multilabel": 28,
        "left_only": 22,
        "right_only": 22,
        "neither": 16,
        "ambiguous": 12,
    },
    "validation": {
        "full_multilabel": 10,
        "left_only": 9,
        "right_only": 9,
        "neither": 7,
        "ambiguous": 5,
    },
}


def normalize(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def load_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_row(
    *,
    split: str,
    family: str,
    index: int,
    text: str,
    capabilities: tuple[str, ...] = (),
    interaction: str,
    request_form: str,
    routing_role: str,
    domains: tuple[str, ...] = (),
    composition: str = "single",
    ambiguity: str = "clear",
) -> dict[str, object]:
    tags = ["reviewed", "korean", "semantic_slice", interaction, request_form, routing_role]
    if routing_role in {"domain_no_action", "lexical_trap", "partial_multilabel", "neither"}:
        tags.append("hard_negative")
    if composition != "single":
        tags.append(composition)
    return {
        "id": f"amadeus-balanced-{split}-{family}-{index:04d}",
        "text": text,
        "capabilities": sorted(capabilities),
        "source": SOURCE,
        "source_split": split,
        "source_intent": family,
        "adaptation": "reviewed-semantic-family-generation",
        "semantic": {
            "interaction": interaction,
            "request_form": request_form,
            "routing_role": routing_role,
            "domains": list(domains),
            "composition": composition,
            "ambiguity": ambiguity,
        },
        "generation": {"family": family, "review_status": "reviewed-pattern-family"},
        "tags": sorted(set(tags)),
    }


CONVERSATION_BANKS = {
    "train": {
        "conversational": (
            "오늘은 그냥 네 이야기를 듣고 싶어", "아까보다 마음이 조금 편해졌어", "요즘 시간이 참 빨리 가는 것 같아",
            "너랑 이야기하면 생각이 정리돼", "오늘 있었던 일을 천천히 말해 볼게", "잠깐 쉬면서 수다나 떨자",
            "별일 없는 하루도 나쁘지 않네", "문득 예전 생각이 났어", "지금 이 순간은 꽤 평화롭다",
            "네가 옆에 있다고 생각하니 든든해", "조금 심심한데 같이 얘기할래", "오늘 하루는 어땠을 것 같아",
            "괜히 웃음이 나는 날이야", "아무 말이나 편하게 해 줘", "그냥 잠깐 같이 있어 줘",
        ),
        "emotional_statement": (
            "오늘은 기분이 많이 가라앉아 있어", "생각보다 일이 잘 풀려서 신나", "괜히 마음이 조급해지네",
            "요즘 작은 일에도 쉽게 지치는 것 같아", "칭찬을 들어서 하루 종일 기분이 좋았어", "조금 외롭다는 생각이 들어",
            "해야 할 일이 많아서 부담스러워", "오랜만에 마음껏 웃었더니 개운해", "실수한 장면이 계속 떠올라 속상해",
            "기대하던 일이 있어서 설레", "오늘은 유난히 자신감이 없어", "그래도 끝까지 해낸 내가 대견해",
            "별 이유 없이 짜증이 나는 날도 있지", "마음이 복잡해서 잠깐 멈추고 싶어", "누군가 내 편이라는 느낌이 필요해",
        ),
        "observation": (
            "창가에 앉으니 시간이 느리게 흐르는 것 같아", "길가의 나무 색이 어제와 달라졌어", "요즘 동네에 사람이 부쩍 많아졌네",
            "책상 위가 생각보다 많이 어질러져 있어", "따뜻한 차 향이 방 안에 오래 남아", "오늘 고양이가 유난히 조용하더라",
            "저녁이 되니 거리 분위기가 완전히 달라졌어", "새로 산 의자가 생각보다 편안해", "요즘 말할 때 자꾸 같은 표현을 쓰게 돼",
            "집에 오면 시간이 더 빨리 가는 느낌이야", "화분에 새잎이 하나 올라왔어", "아침보다 지금 집중이 더 잘돼",
            "요즘은 긴 글보다 짧은 글이 눈에 들어와", "방 안 조명을 바꾸니 분위기가 부드러워졌어", "오늘은 평소보다 사람들이 친절해 보였어",
        ),
        "factual_question": (
            "달의 뒷면은 왜 지구에서 보이지 않아", "고래는 잠을 잘 때 어떻게 숨을 쉬어", "한글은 어떤 원리로 만들어졌어",
            "사람이 하품을 하는 이유는 뭐야", "커피에서 카페인은 어떻게 추출해", "윤년이 생기는 이유를 설명해 줘",
            "빛의 속도는 왜 일정하다고 해", "고대 도시는 주로 강 근처에 생겼어", "나무의 나이테는 어떻게 만들어져",
            "철새는 먼 길을 어떻게 찾아가", "사막의 밤은 왜 그렇게 추워", "종이는 처음 어디에서 만들어졌어",
            "별빛을 보면 과거를 보는 셈이야", "사람마다 목소리가 다른 이유는 뭐야", "발효와 부패는 어떻게 구분해",
        ),
        "short_reaction": (
            "그건 정말 의외네", "응 그 말은 이해했어", "잠깐만 생각해 볼게", "아 그렇구나", "그럴 수도 있겠다",
            "왠지 마음에 드는데", "그 부분은 조금 어렵네", "좋아 계속 말해 줘", "음 아직 잘 모르겠어", "생각보다 재미있다",
            "그건 나중에 다시 얘기하자", "맞아 나도 그렇게 느꼈어", "조금 더 들어 보고 싶어", "이번에는 네 말이 맞는 것 같아", "그 얘기는 여기까지 하자",
        ),
        "domain_no_action": (
            "비 오는 장면이 나오는 영화가 인상적이었어", "예전 겨울은 지금보다 더 길게 느껴졌어", "날씨 이야기는 처음 만난 사람과 하기 편하지",
            "요즘 재즈의 역사에 관한 책을 읽고 있어", "그 가수의 목소리는 들을수록 독특해", "친구와 좋아하는 앨범에 대해 오래 이야기했어",
            "새 컴퓨터 디자인은 꽤 단정해 보이더라", "스피커 음량 표시 방식이 제품마다 다르네", "키보드 소리에 민감한 사람도 많더라",
            "비라는 단어가 들어간 노래 제목이 참 많아", "바람 소리를 녹음한 음악이 신기했어", "컴퓨터가 등장하는 옛날 영화를 봤어",
            "눈 오는 날을 그린 그림이 기억에 남아", "조용한 음악을 좋아하는 취향은 그대로야", "오디오 장비 이야기는 들을수록 복잡해",
        ),
    },
    "validation": {
        "conversational": ("오늘은 네 의견도 궁금해", "잠깐 아무 주제로나 이야기하자", "요즘 내가 좀 달라진 것 같지", "별것 아닌 일도 말하고 싶어", "오늘은 조용히 대화하고 싶다"),
        "emotional_statement": ("괜히 마음이 허전해", "드디어 끝내서 정말 후련하다", "오늘은 사소한 일에도 예민해", "기대와 걱정이 동시에 들어", "생각보다 많이 행복한 하루였어"),
        "observation": ("창문에 비친 방이 낯설게 보여", "요즘 해가 길어진 느낌이야", "책상 위치를 바꾸니 집중이 잘돼", "동네 가게 간판이 새로 바뀌었네", "밤이 되면 생각이 많아져"),
        "factual_question": ("문어의 피는 왜 파란색이야", "무지개는 왜 둥글게 생겨", "도시는 왜 열섬 현상이 생겨", "소리는 진공에서 전달되지 않아", "나침반은 어떻게 북쪽을 찾아"),
        "short_reaction": ("그건 처음 알았네", "응 계속해 봐", "아직은 판단하기 어렵다", "그 말도 일리가 있어", "이번 얘기는 흥미롭네"),
        "domain_no_action": ("비를 소재로 한 소설을 읽었어", "그 밴드의 초기 음악이 더 좋아", "컴퓨터 팬 소리가 소재인 영상이 웃겼어", "친구와 눈 오는 여행을 추억했어", "스피커 디자인에 관한 글을 봤어"),
    },
}


def conversational_rows(split: str) -> list[dict[str, object]]:
    banks = CONVERSATION_BANKS[split]
    target = CONVERSATION_COUNTS[split]
    endings = ("", " 그래서 네 생각이 궁금해") if split == "train" else ("", " 너는 어떻게 생각해")
    candidates: list[tuple[str, str]] = []
    for ending in endings:
        for interaction, texts in banks.items():
            for text in texts:
                candidates.append((interaction, text + ending))
    result = []
    for index, (interaction, text) in enumerate(candidates[:target]):
        domains = tuple(label for label, words in {
            "weather": ("비", "눈", "날씨", "바람"),
            "music_control": ("음악", "노래", "앨범", "가수", "밴드"),
            "pc_control": ("컴퓨터", "스피커", "음량", "키보드", "오디오"),
        }.items() if any(word in text for word in words))
        result.append(make_row(
            split=split, family=f"conversation-{interaction}", index=index, text=text,
            interaction=interaction, request_form="no_request", routing_role="domain_no_action" if domains else "no_tool",
            domains=domains,
        ))
    if len(result) != target:
        raise RuntimeError(f"conversation bank produced {len(result)} of {target} rows")
    return result


SINGLE_BANKS = {
    "weather": {
        "explicit": ("지금 기온 알려 줘", "오늘 비가 오는지 확인해 줘", "내일 날씨를 알려 줘", "현재 바깥 습도가 어때", "이번 주말 예보를 확인해 줘"),
        "implicit": ("우산을 챙겨야 할지 봐 줘", "겉옷이 필요한 날인지 확인해 줘", "빨래를 밖에 널어도 될까", "산책하기 괜찮은 조건인지 봐 줘", "창문을 열어 둘 만한 날씨인지 알려 줘"),
        "domain_no_action": ("비 오는 날의 냄새가 좋아", "지난겨울에는 눈이 자주 왔어", "날씨 이야기는 무난한 대화 주제야", "기후에 관한 다큐멘터리를 봤어", "바람을 소재로 한 시가 기억에 남아"),
        "factual_question": ("구름은 어떻게 만들어져", "번개가 칠 때 천둥이 늦게 들리는 이유는 뭐야", "태풍의 눈은 왜 조용해", "일기예보는 어떤 자료로 만들어", "습도가 높으면 왜 더 덥게 느껴져"),
        "lexical_trap": ("비라는 제목의 소설을 찾고 있어", "눈이 큰 캐릭터가 귀여워", "기온이라는 이름의 주인공이 나와", "바람이라는 밴드 이름이 멋져", "우산 모양 조명이 독특해"),
        "ambiguous": ("오늘은 밖이 좀 궁금하네", "우산 생각이 나네", "겉옷을 볼 때가 됐나", "산책을 나갈까 고민 중이야", "창밖이 평소와 달라 보여"),
    },
    "music_control": {
        "explicit": ("재즈 음악을 재생해 줘", "지금 노래를 잠시 멈춰 줘", "현재 재생을 다시 시작해 줘", "아이유 노래를 틀어 줘", "재생 중인 곡 제목을 알려 줘"),
        "implicit": ("집중할 때 듣기 좋은 곡을 골라 틀어 줘", "분위기를 조금 밝게 바꿔 줄 음악 부탁해", "잠들기 전에 들을 잔잔한 곡을 재생해 줘", "운동할 때 어울리는 비트로 이어 줘", "지금 기분에 맞는 노래 하나 부탁해"),
        "domain_no_action": ("요즘 재즈를 자주 듣고 있어", "그 가수의 새 앨범이 인상적이더라", "친구와 플레이리스트 취향을 비교했어", "노래 가사를 곱씹어 보는 편이야", "라이브 공연의 분위기가 그리워"),
        "factual_question": ("재즈와 블루스는 어떻게 달라", "바로크 음악의 특징은 뭐야", "그 가수는 언제 데뷔했어", "앨범과 싱글의 차이를 설명해 줘", "절대음감은 어떻게 생기는 거야"),
        "lexical_trap": ("노래 잘 부르는 연습법을 알려 줘", "재생 에너지의 장점을 설명해 줘", "플레이리스트라는 단어의 뜻이 뭐야", "앨범 표지를 직접 만들고 싶어", "음악이라는 제목의 책을 읽었어"),
        "ambiguous": ("뭔가 듣고 싶은 기분이네", "요즘 그 노래가 자꾸 생각나", "조용한 곡이면 좋을 텐데", "이어폰을 괜히 챙겼나", "지금 분위기가 조금 허전해"),
    },
    "pc_control": {
        "explicit": ("컴퓨터 소리를 줄여 줘", "음량을 조금 높여 줘", "지금 소리를 음소거해 줘", "스피커 볼륨을 절반으로 맞춰 줘", "컴퓨터 음소거를 해제해 줘"),
        "implicit": ("소리가 너무 크니 조용하게 해 줘", "잘 안 들리니까 조금 키워 줘", "통화 중이니 아무 소리도 안 나게 해 줘", "옆방에 들리지 않을 정도로 낮춰 줘", "다시 들을 수 있게 소리를 켜 줘"),
        "domain_no_action": ("새 컴퓨터는 팬 소리가 조용하더라", "스피커 음질을 비교하는 글을 읽었어", "나는 보통 낮은 음량으로 듣는 편이야", "키보드 소리가 큰 제품도 있더라", "오디오 설정 화면이 복잡해 보여"),
        "factual_question": ("컴퓨터에서 소리는 어떻게 처리돼", "데시벨은 어떤 단위야", "스피커와 헤드폰은 구조가 어떻게 달라", "음소거와 볼륨 0은 같은 상태야", "운영체제는 장치 음량을 어떻게 관리해"),
        "lexical_trap": ("이 글의 목소리를 조금 낮춰 쓰고 싶어", "회의에서 내 발언 비중을 줄여야겠어", "볼륨이라는 만화책을 찾고 있어", "컴퓨터라는 제목의 영화를 봤어", "스피커 역할을 맡은 사람은 누구야"),
        "ambiguous": ("소리가 조금 신경 쓰이네", "오늘은 컴퓨터가 유난히 크게 느껴져", "스피커 쪽을 한번 봐야 하나", "조용했으면 좋겠다는 생각이 들어", "볼륨이 평소와 다른 것 같아"),
    },
}


def single_rows(split: str) -> list[dict[str, object]]:
    result = []
    suffixes = ("", " 부탁할게", " 지금 확인해 줄래") if split == "train" else (" 가능하면 부탁해", " 해 줄 수 있을까")
    for capability, target in SINGLE_COUNTS[split].items():
        roles = tuple(SINGLE_BANKS[capability])
        base, remainder = divmod(target, len(roles))
        candidates = []
        for role_index, role in enumerate(roles):
            texts = SINGLE_BANKS[capability][role]
            count = base + int(role_index < remainder)
            for item_index in range(count):
                text = texts[item_index % len(texts)] + suffixes[item_index // len(texts)]
                candidates.append((role, text))
        for index, (role, text) in enumerate(candidates):
            positive = role in {"explicit", "implicit"}
            result.append(make_row(
                split=split, family=f"single-{capability}-{role}", index=index, text=text,
                capabilities=(capability,) if positive else (),
                interaction="tool_request" if positive else ("factual_question" if role == "factual_question" else "discussion"),
                request_form=role if positive else "no_request",
                routing_role="positive" if positive else role,
                domains=(capability,), ambiguity="conservative" if role == "ambiguous" else "clear",
            ))
    return result


ACTION_BANKS = {
    "weather": ("현재 날씨를 확인해", "우산이 필요한지 봐", "겉옷이 필요할지 알려 줘", "밖에 나가기 괜찮은지 확인해"),
    "music_control": ("잔잔한 음악을 틀어 줘", "다음 노래로 넘겨 줘", "기분 좋은 곡을 재생해 줘", "지금 곡을 잠시 멈춰 줘"),
    "pc_control": ("컴퓨터 소리를 줄여 줘", "음량을 조금 올려 줘", "소리를 음소거해 줘", "다시 소리가 나게 해 줘"),
}
DISCUSSION_BANKS = {
    "weather": ("비 오는 영화 이야기도 하자", "지난겨울 눈이 많았다는 얘기가 떠올라", "날씨라는 말은 대화에 자주 나오지", "바람을 소재로 한 시가 좋더라"),
    "music_control": ("요즘 좋아하는 앨범 이야기도 하고 싶어", "그 가수 목소리가 독특하다는 생각이 들어", "재즈 역사에 관한 책을 읽고 있어", "노래 가사를 해석하는 건 재미있어"),
    "pc_control": ("새 컴퓨터 디자인 이야기도 해 보자", "스피커 음질 차이가 흥미롭더라", "볼륨이라는 만화를 기억해", "키보드 소리에 대한 글을 읽었어"),
}


def pair_text(pair: tuple[str, str], role: str, index: int, split: str) -> str:
    left, right = pair
    la = ACTION_BANKS[left][index % 4]
    ra = ACTION_BANKS[right][(index * 3 + (1 if split == "validation" else 0)) % 4]
    ld = DISCUSSION_BANKS[left][(index * 2 + (1 if split == "validation" else 0)) % 4]
    rd = DISCUSSION_BANKS[right][(index * 3 + 2) % 4]
    moments = (
        "오늘 아침에", "점심 무렵에", "오후 일정 전에", "저녁이 되기 전에", "잠깐 쉬는 동안",
        "일을 시작하기 전에", "집을 나서기 전에", "약속을 준비하면서",
    ) if split == "train" else (
        "이른 아침에", "점심 약속 전에", "오후 일을 시작하며", "저녁 약속에 맞춰", "잠시 여유가 있을 때",
        "준비를 마치기 전에", "외출하기 직전에", "하루를 정리하면서",
    )
    purposes = ("바로", "차분하게", "간단히", "잊지 않게", "한꺼번에")
    context = f"{moments[index % len(moments)]} {purposes[(index // len(moments)) % len(purposes)]}"
    domain_names = {"weather": "날씨", "music_control": "음악", "pc_control": "컴퓨터 소리"}
    if split == "validation":
        if role == "full_multilabel":
            forms = (f"{context} {la}. 그것까지 확인되면 {ra}", f"{context} {ra}. 같은 흐름으로 {la} 줘")
        elif role == "left_only":
            forms = (f"{rd}는 나중에 더 얘기하고, {context} {la} 줘", f"{context} 필요한 건 {la}는 거야. 참고로 {rd}")
        elif role == "right_only":
            forms = (f"{ld}는 그냥 내 생각이고, {context} {ra}", f"{context} 부탁은 {ra}는 거야. {ld}")
        elif role == "neither":
            forms = (f"{context} {ld}. 부탁하는 건 아니고 {rd}", f"{context} {rd}는 흥미롭고, {ld}")
        else:
            forms = (
                f"{context} {domain_names[left]}하고 {domain_names[right]} 둘 다 문득 떠오르네",
                f"{context} {domain_names[right]}도 {domain_names[left]}도 오늘따라 신경이 쓰여",
            )
        return forms[index % len(forms)]
    if role == "full_multilabel":
        forms = (
            f"{context} {la} 주고 이어서 {ra}",
            f"{context} {ra}고 나서 {la} 줘",
            f"{context} {la}면서 {ra}",
            f"{context} 상황에 맞춰 {la}고 동시에 {ra}",
        )
    elif role == "left_only":
        forms = (f"{context} {la} 줘, 그리고 {rd}", f"{rd}. 그건 그렇고 {context} {la} 줘", f"{context} {la} 주면 좋겠어. {rd}")
    elif role == "right_only":
        forms = (f"{ld}. 그리고 {context} {ra}", f"{context} {ra}고 {ld}", f"{ld}지만 {context} {ra}")
    elif role == "neither":
        forms = (f"{context} 문득 {ld}, {rd}", f"{rd}. {context} 그러고 보니 {ld}", f"{context} {ld}고 {rd}")
    else:
        forms = (
            f"{context} {domain_names[left]}도 {domain_names[right]}도 조금 신경 쓰이네",
            f"{context} {domain_names[left]}와 {domain_names[right]} 생각이 같이 나",
            f"{context} {domain_names[left]} 쪽도 보고 {domain_names[right]} 쪽도 생각해 봐야 하나",
        )
    return forms[index % len(forms)]


def pair_rows(split: str) -> list[dict[str, object]]:
    result = []
    for pair_index, pair in enumerate(PAIR_SPECS):
        row_index = 0
        for role, count in PAIR_COUNTS[split].items():
            for index in range(count):
                if role == "full_multilabel":
                    labels = pair
                    routing_role = role
                elif role == "left_only":
                    labels = (pair[0],)
                    routing_role = "partial_multilabel"
                elif role == "right_only":
                    labels = (pair[1],)
                    routing_role = "partial_multilabel"
                else:
                    labels = ()
                    routing_role = role
                result.append(make_row(
                    split=split, family=f"pair-{pair_index + 1}-{role}", index=row_index,
                    text=pair_text(pair, role, index + pair_index * 5, split), capabilities=labels,
                    interaction="tool_request" if labels else "discussion",
                    request_form="mixed" if role not in {"neither", "ambiguous"} else "no_request",
                    routing_role=routing_role, domains=pair, composition=role,
                    ambiguity="conservative" if role == "ambiguous" else "clear",
                ))
                row_index += 1
    return result


def build(split: str) -> list[dict[str, object]]:
    rows = conversational_rows(split) + single_rows(split) + pair_rows(split)
    keys = [normalize(str(row["text"])) for row in rows]
    if len(keys) != len(set(keys)):
        duplicates = [key for key, count in Counter(keys).items() if count > 1]
        raise ValueError(f"generated normalized duplicates: {duplicates[:3]}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    train = load_rows(BASE_DATA / "train.jsonl")
    validation = load_rows(BASE_DATA / "validation.jsonl")
    added_train = build("train")
    added_validation = build("validation")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "train.jsonl", train + added_train)
    write_rows(args.output_dir / "validation.jsonl", validation + added_validation)
    shutil.copyfile(BASE_DATA / "external_test.jsonl", args.output_dir / "external_test.jsonl")
    added = added_train + added_validation
    report = {
        "source": SOURCE,
        "base_train_rows": len(train),
        "base_validation_rows": len(validation),
        "added_train_rows": len(added_train),
        "added_validation_rows": len(added_validation),
        "balanced_train_rows": len(train) + len(added_train),
        "balanced_validation_rows": len(validation) + len(added_validation),
        "split_interactions": Counter(f"{row['source_split']}:{row['semantic']['interaction']}" for row in added),
        "split_routing_roles": Counter(f"{row['source_split']}:{row['semantic']['routing_role']}" for row in added),
        "split_compositions": Counter(f"{row['source_split']}:{row['semantic']['composition']}" for row in added),
        "label_sets": Counter("+".join(row["capabilities"]) or "no_match" for row in added),
        "metadata_schema": {
            "interaction": "conversational, emotional_statement, observation, factual_question, discussion, or tool_request",
            "request_form": "explicit, implicit, mixed, ambiguous, or no_request",
            "routing_role": "positive, no_tool, domain_no_action, lexical_trap, full/partial_multilabel, neither, or ambiguous",
            "domains": "capability vocabulary present in the utterance",
            "composition": "single or pairwise routing truth family",
            "ambiguity": "clear or conservative",
        },
        "external_test_unchanged": (BASE_DATA / "external_test.jsonl").read_bytes()
        == (args.output_dir / "external_test.jsonl").read_bytes(),
    }
    (args.output_dir / "corpus_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
