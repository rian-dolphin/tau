from collections.abc import AsyncIterator

import pytest

from tau_agent import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    AssistantMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from tau_coding.tui.app import TauTuiApp


class FakeSession:
    def __init__(self, messages=(), events=()) -> None:
        self.messages = tuple(messages)
        self.events = tuple(events)

    async def prompt(self, text: str) -> AsyncIterator[AgentEvent]:
        for event in self.events:
            yield event


def test_tui_app_loads_restored_messages_into_display_state() -> None:
    app = TauTuiApp(
        FakeSession(
            messages=[
                UserMessage(content="Read the file"),
                AssistantMessage(
                    content="I'll inspect it.",
                    tool_calls=[
                        ToolCall(id="call-1", name="read", arguments={"path": "README.md"})
                    ],
                ),
                ToolResultMessage(
                    tool_call_id="call-1",
                    name="read",
                    content="README contents",
                    ok=True,
                ),
            ]
        )
    )

    assert [(item.role, item.text) for item in app.state.items] == [
        ("user", "Read the file"),
        ("assistant", "I'll inspect it."),
        ("tool", "→ read {'path': 'README.md'}"),
        ("tool", "✓ read\nREADME contents"),
    ]


@pytest.mark.anyio
async def test_tui_prompt_worker_refreshes_directly() -> None:
    app = TauTuiApp(FakeSession(events=[AgentStartEvent(), AgentEndEvent()]))
    refreshes = 0

    def fake_refresh() -> None:
        nonlocal refreshes
        refreshes += 1

    app._refresh = fake_refresh  # type: ignore[method-assign]

    await app._run_prompt("hello")

    assert refreshes == 2
    assert app.state.running is False
