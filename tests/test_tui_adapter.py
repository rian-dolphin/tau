from tau_agent import (
    AgentEndEvent,
    AgentStartEvent,
    AgentToolResult,
    AssistantMessage,
    ErrorEvent,
    MessageDeltaEvent,
    MessageEndEvent,
    MessageStartEvent,
    ToolCall,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    UserMessage,
)
from tau_coding.tui import TuiEventAdapter, TuiState
from tau_coding.tui.state import format_tool_result_block


def test_tui_adapter_tracks_running_state() -> None:
    state = TuiState()
    adapter = TuiEventAdapter(state)

    adapter.apply(AgentStartEvent())
    assert state.running is True

    adapter.apply(AgentEndEvent())
    assert state.running is False


def test_tui_adapter_builds_assistant_items_from_streamed_messages() -> None:
    state = TuiState()
    adapter = TuiEventAdapter(state)

    adapter.apply(MessageStartEvent())
    adapter.apply(MessageDeltaEvent(delta="Hel"))
    adapter.apply(MessageDeltaEvent(delta="lo"))
    assert state.assistant_buffer == "Hello"
    assert state.items == []

    adapter.apply(MessageEndEvent(message=AssistantMessage(content="Hello")))

    assert state.assistant_buffer == ""
    assert [(item.role, item.text) for item in state.items] == [("assistant", "Hello")]


def test_tui_adapter_builds_user_items_from_streamed_messages() -> None:
    state = TuiState()
    adapter = TuiEventAdapter(state)

    adapter.apply(MessageStartEvent(message_role="user"))
    adapter.apply(MessageEndEvent(message=UserMessage(content="Hello Tau")))

    assert state.assistant_buffer == ""
    assert [(item.role, item.text) for item in state.items] == [("user", "Hello Tau")]


def test_tui_adapter_flushes_assistant_buffer_before_tool_events() -> None:
    state = TuiState()
    adapter = TuiEventAdapter(state)

    adapter.apply(MessageDeltaEvent(delta="Before tool"))
    adapter.apply(
        ToolExecutionStartEvent(
            tool_call=ToolCall(id="call-1", name="read", arguments={"path": "README.md"})
        )
    )

    assert state.assistant_buffer == ""
    assert state.items[0].role == "assistant"
    assert state.items[0].text == "Before tool"
    assert state.items[1].role == "tool"
    assert "→ read" in state.items[1].text


def test_tui_adapter_records_tool_updates_and_results() -> None:
    state = TuiState()
    adapter = TuiEventAdapter(state)

    adapter.apply(ToolExecutionUpdateEvent(tool_call_id="call-1", message="reading"))
    adapter.apply(
        ToolExecutionEndEvent(
            result=AgentToolResult(tool_call_id="call-1", name="read", ok=True, content="done")
        )
    )
    adapter.apply(
        ToolExecutionEndEvent(
            result=AgentToolResult(
                tool_call_id="call-2",
                name="bash",
                ok=False,
                content="failed",
            )
        )
    )

    assert [(item.role, item.text, item.tool_result_text) for item in state.items] == [
        ("tool", "… reading", None),
        ("tool", "✓ read", "✓ read\ndone"),
        ("tool", "✗ bash", "✗ bash\nfailed"),
    ]


def test_tool_result_blocks_preview_long_content() -> None:
    content = "\n".join(f"line {index}" for index in range(1, 12))

    block = format_tool_result_block(name="read", ok=True, content=content)

    assert "line 1" in block
    assert "line 8" in block
    assert "line 9" not in block
    assert "3 more lines" in block


def test_tui_adapter_renders_live_edit_patch() -> None:
    state = TuiState()
    adapter = TuiEventAdapter(state)

    adapter.apply(
        ToolExecutionEndEvent(
            result=AgentToolResult(
                tool_call_id="call-1",
                name="edit",
                ok=True,
                content="Successfully replaced 1 block.",
                data={"patch": "--- a.py\n+++ a.py\n@@\n-old\n+new"},
            )
        )
    )

    assert [(item.role, item.text, item.tool_result_text) for item in state.items] == [
        (
            "tool",
            "✓ edit",
            "✓ edit\n"
            "Successfully replaced 1 block.\n"
            "\n"
            "Patch:\n"
            "--- a.py\n"
            "+++ a.py\n"
            "@@\n"
            "-old\n"
            "+new",
        )
    ]


def test_tui_adapter_records_errors_and_stops_on_non_recoverable_error() -> None:
    state = TuiState(running=True, assistant_buffer="partial")
    adapter = TuiEventAdapter(state)

    adapter.apply(ErrorEvent(message="provider failed", recoverable=False))

    assert state.running is False
    assert state.error == "provider failed"
    assert [(item.role, item.text) for item in state.items] == [
        ("assistant", "partial"),
        ("error", "Error: provider failed"),
    ]
