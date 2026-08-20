from app.memory import ConversationMemory


def test_memory_is_bounded_and_summarized() -> None:
    memory = ConversationMemory(max_recent_messages=4)
    for index in range(4):
        memory.add_turn(f"u{index}", f"a{index}")
    messages = memory.messages()
    assert messages[0].role == "system"
    assert len([item for item in messages if item.role != "system"]) == 4

