"""Small Textual widgets for Tau's interactive TUI."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.widgets import RichLog, Static

from tau_agent.tools import AgentTool
from tau_coding.prompt_templates import PromptTemplate
from tau_coding.skills import Skill
from tau_coding.tui.state import ChatItem, TuiState

_ROLE_LABELS = {
    "user": "you",
    "assistant": "assistant",
    "tool": "tool",
    "error": "error",
    "status": "status",
}

_ROLE_STYLES = {
    "user": "bold bright_cyan",
    "assistant": "bold bright_green",
    "tool": "bright_yellow",
    "error": "bold bright_red",
    "status": "dim",
}


class SessionSummarySource(Protocol):
    """Session attributes displayed by the sidebar."""

    @property
    def cwd(self) -> Path: ...

    @property
    def model(self) -> str: ...

    @property
    def tools(self) -> Sequence[AgentTool]: ...

    @property
    def skills(self) -> Sequence[Skill]: ...

    @property
    def prompt_templates(self) -> Sequence[PromptTemplate]: ...


class SessionSidebar(Static):
    """Compact sidebar with current session metadata."""

    def update_from_session(self, session: SessionSummarySource) -> None:
        """Redraw the sidebar from current session metadata."""
        self.update(render_session_sidebar(session))


class TranscriptView(RichLog):
    """Scrollable transcript view backed by ``TuiState``."""

    def update_from_state(self, state: TuiState) -> None:
        """Redraw the transcript from display state."""
        self.clear()
        for item in state.items:
            self.write(render_chat_item(item))
        if state.assistant_buffer:
            self.write(render_chat_item(ChatItem(role="assistant", text=state.assistant_buffer)))


def render_session_sidebar(session: SessionSummarySource) -> RenderableType:
    """Render a dark, minimalist summary of the active coding session."""
    metadata = Table.grid(padding=(0, 1))
    metadata.add_column(style="bright_black", no_wrap=True)
    metadata.add_column(style="white")
    metadata.add_row("model", session.model)
    metadata.add_row("cwd", _short_path(session.cwd))
    metadata.add_row("tools", str(len(session.tools)))
    metadata.add_row("skills", str(len(session.skills)))

    tools = _bullet_list([tool.name for tool in session.tools], empty="No tools")
    skills = _bullet_list([skill.name for skill in session.skills], empty="No skills loaded yet")
    prompts = _bullet_list(
        [template.name for template in session.prompt_templates],
        empty="No prompt templates",
    )

    return Group(
        Panel(metadata, title="session", border_style="bright_black", padding=(0, 1)),
        Panel(tools, title="tools", border_style="cyan", padding=(0, 1)),
        Panel(skills, title="skills", border_style="green", padding=(0, 1)),
        Panel(prompts, title="prompts", border_style="magenta", padding=(0, 1)),
    )


def render_chat_item(item: ChatItem) -> Text:
    """Render a chat item as Rich text."""
    label = _ROLE_LABELS[item.role]
    style = _ROLE_STYLES[item.role]
    text = Text()
    text.append(f"{label}: ", style=style)
    text.append(item.text, style="white")
    return text


def _bullet_list(items: Sequence[str], *, empty: str) -> Text:
    text = Text()
    if not items:
        text.append(empty, style="bright_black")
        return text

    for index, item in enumerate(items):
        if index:
            text.append("\n")
        text.append("• ", style="bright_black")
        text.append(item, style="white")
    return text


def _short_path(path: Path) -> str:
    home = Path.home()
    try:
        return f"~/{path.relative_to(home)}"
    except ValueError:
        return str(path)
