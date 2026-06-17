"""Textual TUI frontend for Tau coding sessions."""

from tau_coding.tui.adapter import TuiEventAdapter
from tau_coding.tui.app import TauTuiApp, run_tui_app
from tau_coding.tui.state import ChatItem, TuiState
from tau_coding.tui.widgets import (
    SessionSidebar,
    TranscriptView,
    render_chat_item,
    render_session_sidebar,
)

__all__ = [
    "ChatItem",
    "TauTuiApp",
    "SessionSidebar",
    "TranscriptView",
    "TuiEventAdapter",
    "TuiState",
    "render_chat_item",
    "render_session_sidebar",
    "run_tui_app",
]
