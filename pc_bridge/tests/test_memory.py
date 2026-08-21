from app.memory import ConversationMemory


def test_memory_is_bounded_and_summarized() -> None:
    memory = ConversationMemory(max_recent_messages=4)
    for index in range(4):
        memory.add_turn(f"사용자 질문 {index}", f"크리스 답변 {index}")
    messages = memory.messages()
    assert messages[0].role == "system"
    assert len([item for item in messages if item.role != "system"]) == 4


def test_default_memory_keeps_only_four_recent_turns() -> None:
    memory = ConversationMemory()
    for index in range(6):
        memory.add_turn(f"질문 {index}", f"답변 {index}")
    recent = [item for item in memory.messages() if item.role != "system"]
    assert len(recent) == 8


def test_memory_preserves_facts_and_drops_low_information_chatter() -> None:
    memory = ConversationMemory(max_recent_messages=2, max_summary_chars=240)
    memory.add_turn("안녕", "안녕")
    memory.add_turn("내 이름은 의준이라고 해", "기억할게")
    memory.add_turn("오늘은 코딩을 공부하고 있어", "어떤 부분을 보고 있어?")

    summary = memory.messages()[0].content
    assert "내 이름은 의준이라고 해" in summary
    assert "사용자: 안녕" not in summary


def test_summary_respects_character_budget() -> None:
    memory = ConversationMemory(max_recent_messages=2, max_summary_chars=180)
    for index in range(12):
        memory.add_turn(
            f"오늘 진행하는 프로젝트의 세부 작업 {index}를 기록해줘",
            f"세부 작업 {index}의 진행 상황을 기억할게",
        )
    summary = memory.messages()[0].content
    assert len(summary) <= 180

