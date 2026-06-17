from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from rich.console import Console

from tau_agent import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    AssistantMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from tau_coding.skills import Skill
from tau_coding.tools import create_coding_tools
from tau_coding.tui.app import TauTuiApp
from tau_coding.tui.widgets import render_session_sidebar


class FakeSession:
    def __init__(self, messages=(), events=()) -> None:
        self.messages = tuple(messages)
        self.events = tuple(events)
        self.cwd = Path("/workspace/project")
        self.model = "fake-model"
        self.tools = tuple(create_coding_tools(cwd=self.cwd))
        self.skills = (Skill(name="review", path=self.cwd / "review.md", content="Review code"),)
        self.prompt_templates = ()

    async def prompt(self, text: str) -> AsyncIterator[AgentEvent]:
        for event in self.events:
            yield event


def test_session_sidebar_renders_session_metadata() -> None:
    console = Console(record=True, width=80)

    console.print(render_session_sidebar(FakeSession()))

    output = console.export_text()
    assert "session" in output
    assert "fake-model" in output
    assert "tools" in output
    assert "read" in output
    assert "skills" in output
    assert "review" in output


@pytest.mark.anyio
async def test_tui_app_mounts_sidebar_and_transcript() -> None:
    app = TauTuiApp(FakeSession())

    async with app.run_test():
        assert app.query_one("#sidebar") is not None
        assert app.query_one("#transcript") is not None
        assert app.query_one("#prompt") is not None


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
