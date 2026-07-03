"""Extension-facing API types and hook payloads."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from tau_agent.tools import AgentTool, AgentToolResult
from tau_agent.types import JSONValue

if TYPE_CHECKING:
    from tau_coding.extensions.runtime import ExtensionRuntime

AGENT_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "agent_start",
        "agent_end",
        "turn_start",
        "turn_end",
        "retry",
        "queue_update",
        "message_start",
        "message_delta",
        "thinking_delta",
        "message_end",
        "tool_execution_start",
        "tool_execution_update",
        "tool_execution_end",
        "error",
    }
)
AGENT_EVENT_WILDCARD = "agent_event"

LIFECYCLE_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "session_start",
        "session_shutdown",
        "input",
        "tool_call",
        "tool_result",
    }
)

SessionLifecycleReason = Literal["startup", "reload", "new", "resume", "branch", "quit"]
DeliverAs = Literal["steer", "follow_up"]
NotifyLevel = Literal["info", "warning", "error"]


class ExtensionError(RuntimeError):
    """Raised when an extension misuses the API (e.g. actions before binding)."""


@dataclass(frozen=True, slots=True)
class SessionStartEvent:
    """Payload for the `session_start` lifecycle event."""

    reason: SessionLifecycleReason


@dataclass(frozen=True, slots=True)
class SessionShutdownEvent:
    """Payload for the `session_shutdown` lifecycle event."""

    reason: SessionLifecycleReason


@dataclass(frozen=True, slots=True)
class InputEvent:
    """Payload for the `input` hook: user prompt text after expansion."""

    text: str


@dataclass(frozen=True, slots=True)
class InputHookResult:
    """Result of an `input` hook handler.

    `action="continue"` leaves the text unchanged, `"transform"` replaces it
    with `text` (transforms chain across handlers), and `"handled"` consumes
    the input entirely, optionally showing `message` to the user.
    """

    action: Literal["continue", "transform", "handled"] = "continue"
    text: str | None = None
    message: str | None = None


@dataclass(frozen=True, slots=True)
class ToolCallHookEvent:
    """Payload for the `tool_call` hook, before a tool executes.

    Carries no tool-call id: the hook runs inside the tool executor seam,
    which the agent loop invokes without the id. Use the observation events
    (`tool_execution_start`/`tool_execution_end`) for id correlation.
    """

    tool_name: str
    arguments: Mapping[str, JSONValue]


@dataclass(frozen=True, slots=True)
class ToolCallHookResult:
    """Result of a `tool_call` hook handler.

    Set `block=True` (with an optional `reason`) to prevent execution, or
    return replacement `arguments` to rewrite the call. Blocking wins over
    argument rewrites and short-circuits remaining handlers.
    """

    block: bool = False
    reason: str | None = None
    arguments: Mapping[str, JSONValue] | None = None


@dataclass(frozen=True, slots=True)
class ToolResultHookEvent:
    """Payload for the `tool_result` hook, after a tool executes."""

    tool_name: str
    arguments: Mapping[str, JSONValue]
    result: AgentToolResult


@dataclass(frozen=True, slots=True)
class ToolResultHookResult:
    """Result of a `tool_result` hook handler; set fields to override."""

    content: str | None = None
    ok: bool | None = None
    details: dict[str, JSONValue] | None = None


ExtensionHandler = Callable[[object], object | Awaitable[object]]
# Command handlers are sync-only: the slash-command path (CommandRegistry ->
# CodingSession.handle_command -> TUI submit) is synchronous end to end.
ExtensionCommandHandler = Callable[["str", "ExtensionCommandContext"], "str | None"]


@dataclass(frozen=True, slots=True)
class ExtensionRuntimeDiagnostic:
    """A runtime failure raised by an extension handler."""

    extension: str
    event: str
    message: str


class UiBridge(Protocol):
    """Host-provided UI capabilities available to extensions."""

    @property
    def has_ui(self) -> bool:
        """Return whether an interactive UI is attached."""
        ...

    def notify(self, message: str, level: NotifyLevel = "info") -> None:
        """Show a notification to the user (no-op without a UI)."""
        ...


class NullUiBridge:
    """UI bridge used when no interactive frontend is attached."""

    @property
    def has_ui(self) -> bool:
        """Return False: print mode has no interactive UI."""
        return False

    def notify(self, message: str, level: NotifyLevel = "info") -> None:
        """Ignore notifications without a UI."""


@dataclass(frozen=True, slots=True)
class ExtensionCommandContext:
    """Context passed to extension slash-command handlers."""

    name: str
    args: str
    api: ExtensionAPI


class ExtensionContext:
    """Read-only session context exposed to extensions."""

    def __init__(self, runtime: ExtensionRuntime) -> None:
        self._runtime = runtime

    @property
    def cwd(self) -> Path:
        """Return the session working directory."""
        return self._runtime.session_view.cwd

    @property
    def model(self) -> str:
        """Return the active model name."""
        return self._runtime.session_view.model

    @property
    def provider_name(self) -> str:
        """Return the active provider name."""
        return self._runtime.session_view.provider_name

    @property
    def session_id(self) -> str | None:
        """Return the current session id, if the session is indexed."""
        return self._runtime.session_view.session_id

    @property
    def system_prompt(self) -> str:
        """Return the active system prompt."""
        return self._runtime.session_view.system_prompt

    @property
    def is_running(self) -> bool:
        """Return whether an agent run is currently active."""
        return self._runtime.session_view.is_running

    @property
    def has_ui(self) -> bool:
        """Return whether an interactive UI is attached."""
        return self._runtime.ui.has_ui


class ExtensionAPI:
    """The object handed to each extension's `setup(tau)` entry point."""

    def __init__(self, runtime: ExtensionRuntime, extension_name: str) -> None:
        self._runtime = runtime
        self._extension_name = extension_name
        self._context = ExtensionContext(runtime)

    @property
    def name(self) -> str:
        """Return this extension's name."""
        return self._extension_name

    @property
    def context(self) -> ExtensionContext:
        """Return read-only session context."""
        return self._context

    def register_tool(self, tool: AgentTool) -> None:
        """Register an agent tool (first registration per name wins)."""
        self._runtime.register_tool(self._extension_name, tool)

    def register_command(
        self,
        name: str,
        handler: ExtensionCommandHandler,
        *,
        description: str = "",
        usage: str | None = None,
        aliases: tuple[str, ...] = (),
    ) -> None:
        """Register a slash command backed by this extension."""
        self._runtime.register_command(
            self._extension_name,
            name,
            handler,
            description=description,
            usage=usage,
            aliases=aliases,
        )

    def on(
        self,
        event: str,
        handler: ExtensionHandler | None = None,
    ) -> Callable[[ExtensionHandler], ExtensionHandler] | ExtensionHandler:
        """Subscribe to an event, directly or as a decorator."""
        if handler is not None:
            self._runtime.subscribe(self._extension_name, event, handler)
            return handler

        def decorator(decorated: ExtensionHandler) -> ExtensionHandler:
            self._runtime.subscribe(self._extension_name, event, decorated)
            return decorated

        return decorator

    def send_user_message(
        self,
        content: str,
        *,
        deliver_as: DeliverAs = "follow_up",
    ) -> None:
        """Queue a user message for the active or next agent run."""
        self._runtime.send_user_message(content, deliver_as=deliver_as)

    async def append_entry(self, namespace: str, data: dict[str, JSONValue]) -> None:
        """Persist extension-owned data to the session as a custom entry."""
        await self._runtime.append_custom_entry(namespace, data)

    def notify(self, message: str, level: NotifyLevel = "info") -> None:
        """Show a notification in the UI, if one is attached."""
        self._runtime.ui.notify(message, level)


@dataclass(slots=True)
class RegisteredExtension:
    """Book-keeping for one loaded extension inside the runtime."""

    name: str
    path: Path
    api: ExtensionAPI
    handlers: dict[str, list[ExtensionHandler]] = field(default_factory=dict)
