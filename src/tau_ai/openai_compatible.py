"""OpenAI-compatible chat completions provider."""

from collections.abc import AsyncIterator, Mapping
from json import JSONDecodeError, dumps, loads
from typing import Any

import httpx

from tau_agent.messages import (
    AgentMessage,
    AssistantMessage,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from tau_agent.tools import AgentTool, ToolCall
from tau_agent.types import JSONValue
from tau_ai.env import OpenAICompatibleConfig
from tau_ai.events import (
    ProviderErrorEvent,
    ProviderEvent,
    ProviderResponseEndEvent,
    ProviderResponseStartEvent,
    ProviderTextDeltaEvent,
    ProviderThinkingDeltaEvent,
    ProviderToolCallEvent,
)
from tau_ai.provider import CancellationToken
from tau_ai.retry import provider_retry_event, retry_delay_seconds, wait_for_retry


class OpenAICompatibleProvider:
    """Provider adapter for OpenAI-compatible `/chat/completions` APIs."""

    def __init__(
        self,
        config: OpenAICompatibleConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._client = client
        self._owns_client = client is None

    async def aclose(self) -> None:
        """Close the underlying HTTP client if this provider created it."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        signal: CancellationToken | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        """Stream one chat completion response as provider-neutral events."""

        async def iterator() -> AsyncIterator[ProviderEvent]:
            client = self._get_client()
            payload = _build_chat_payload(
                model=model,
                system=system,
                messages=messages,
                tools=tools,
                reasoning_effort=self._config.reasoning_effort,
                reasoning_effort_parameter=self._config.reasoning_effort_parameter,
                supports_usage_in_streaming=self._config.supports_usage_in_streaming,
            )
            headers = {
                **(dict(self._config.headers or {})),
                "Authorization": f"Bearer {self._config.api_key}",
            }
            url = f"{self._config.base_url.rstrip('/')}/chat/completions"

            attempt = 0
            while True:
                emitted_content = False
                try:
                    async with client.stream(
                        "POST", url, json=payload, headers=headers
                    ) as response:
                        if response.status_code >= 400:
                            body = await response.aread()
                            if self._should_retry(attempt, status_code=response.status_code):
                                delay = retry_delay_seconds(
                                    attempt,
                                    max_delay_seconds=self._config.max_retry_delay_seconds,
                                )
                                yield provider_retry_event(
                                    attempt=attempt,
                                    max_retries=self._config.max_retries,
                                    delay_seconds=delay,
                                    reason=f"HTTP {response.status_code}",
                                    data={
                                        "status_code": response.status_code,
                                        "body": body.decode(errors="replace"),
                                    },
                                )
                                attempt += 1
                                if not await wait_for_retry(delay, signal=signal):
                                    return
                                continue
                            yield ProviderErrorEvent(
                                message=(
                                    "Provider request failed with status "
                                    f"{response.status_code}"
                                ),
                                data={
                                    "body": body.decode(errors="replace"),
                                    "attempts": attempt + 1,
                                },
                            )
                            return

                        yield ProviderResponseStartEvent(model=model)
                        content_parts: list[str] = []
                        tool_call_builders: dict[int, _ToolCallBuilder] = {}
                        finish_reason: str | None = None
                        usage: Usage | None = None

                        async for line in response.aiter_lines():
                            if signal is not None and signal.is_cancelled():
                                return

                            event = _parse_sse_line(line)
                            if event is None:
                                continue
                            if event == "[DONE]":
                                break

                            chunk = _loads_object(event)
                            if chunk is None:
                                yield ProviderErrorEvent(
                                    message="Provider returned invalid JSON chunk"
                                )
                                return

                            # The final usage chunk (from stream_options) carries
                            # usage at the top level and often has empty choices.
                            chunk_usage = chunk.get("usage")
                            if isinstance(chunk_usage, Mapping):
                                usage = _parse_chunk_usage(chunk_usage)

                            choice = _first_choice(chunk)
                            if choice is None:
                                continue

                            # Fallback: some providers (e.g. Moonshot) attach
                            # usage to the choice instead of the chunk. Matches
                            # Pi's per-chunk `!chunk.usage` guard: the fallback
                            # applies whenever this chunk lacks top-level usage.
                            choice_usage = choice.get("usage")
                            if not isinstance(chunk_usage, Mapping) and isinstance(
                                choice_usage, Mapping
                            ):
                                usage = _parse_chunk_usage(choice_usage)

                            finish_reason = choice.get("finish_reason") or finish_reason
                            delta = choice.get("delta")
                            if not isinstance(delta, Mapping):
                                continue

                            content = delta.get("content")
                            if isinstance(content, str) and content:
                                emitted_content = True
                                content_parts.append(content)
                                yield ProviderTextDeltaEvent(delta=content)

                            thinking = _thinking_delta_text(delta)
                            if thinking:
                                emitted_content = True
                                yield ProviderThinkingDeltaEvent(delta=thinking)

                            for tool_call_delta in _tool_call_deltas(delta):
                                emitted_content = True
                                index = int(tool_call_delta.get("index", 0))
                                builder = tool_call_builders.setdefault(
                                    index, _ToolCallBuilder()
                                )
                                builder.add_delta(tool_call_delta)

                        tool_calls = [
                            builder.build(index)
                            for index, builder in sorted(tool_call_builders.items())
                        ]
                        for tool_call in tool_calls:
                            yield ProviderToolCallEvent(tool_call=tool_call)

                        message = AssistantMessage(
                            content="".join(content_parts),
                            tool_calls=tool_calls,
                            usage=usage,
                        )
                        yield ProviderResponseEndEvent(
                            message=message, finish_reason=finish_reason
                        )
                        return
                except httpx.HTTPError as exc:
                    if not emitted_content and self._should_retry(attempt):
                        delay = retry_delay_seconds(
                            attempt,
                            max_delay_seconds=self._config.max_retry_delay_seconds,
                        )
                        yield provider_retry_event(
                            attempt=attempt,
                            max_retries=self._config.max_retries,
                            delay_seconds=delay,
                            reason="network error",
                            data={
                                "error": str(exc),
                                "error_type": type(exc).__name__,
                            },
                        )
                        attempt += 1
                        if not await wait_for_retry(delay, signal=signal):
                            return
                        continue
                    yield ProviderErrorEvent(
                        message=str(exc),
                        data={"attempts": attempt + 1},
                    )
                    return

        return iterator()

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._config.timeout_seconds)
        return self._client

    def _should_retry(self, attempt: int, *, status_code: int | None = None) -> bool:
        if attempt >= self._config.max_retries:
            return False
        return status_code is None or _is_transient_status(status_code)


class _ToolCallBuilder:
    def __init__(self) -> None:
        self.id = ""
        self.name = ""
        self.arguments_parts: list[str] = []

    def add_delta(self, delta: Mapping[str, Any]) -> None:
        call_id = delta.get("id")
        if isinstance(call_id, str):
            self.id = call_id

        function = delta.get("function")
        if not isinstance(function, Mapping):
            return

        name = function.get("name")
        if isinstance(name, str):
            self.name = name

        arguments = function.get("arguments")
        if isinstance(arguments, str):
            self.arguments_parts.append(arguments)

    def build(self, index: int) -> ToolCall:
        arguments_text = "".join(self.arguments_parts)
        arguments = _loads_object(arguments_text) if arguments_text else {}
        if arguments is None:
            arguments = {"_raw_arguments": arguments_text}

        return ToolCall(
            id=self.id or f"tool-call-{index}",
            name=self.name,
            arguments=arguments,
        )


def _build_chat_payload(
    *,
    model: str,
    system: str,
    messages: list[AgentMessage],
    tools: list[AgentTool],
    reasoning_effort: str | None = None,
    reasoning_effort_parameter: str = "reasoning_effort",
    supports_usage_in_streaming: bool = True,
) -> dict[str, JSONValue]:
    payload: dict[str, JSONValue] = {
        "model": model,
        "stream": True,
        "messages": [
            _system_message(system),
            *[_message_to_openai(message) for message in messages],
        ],
    }
    if supports_usage_in_streaming:
        # Ask OpenAI-compatible providers to report billed token usage in the
        # final streaming chunk (mirrors Pi's include_usage default).
        payload["stream_options"] = {"include_usage": True}
    if reasoning_effort is not None:
        if reasoning_effort_parameter == "reasoning.effort":
            payload["reasoning"] = {"effort": reasoning_effort}
        else:
            payload["reasoning_effort"] = reasoning_effort
    if tools:
        payload["tools"] = [_tool_to_openai(tool) for tool in tools]
    return payload


def _system_message(system: str) -> dict[str, JSONValue]:
    return {"role": "system", "content": system}


def _message_to_openai(message: AgentMessage) -> dict[str, JSONValue]:
    if isinstance(message, UserMessage):
        return {"role": "user", "content": message.content}

    if isinstance(message, AssistantMessage):
        item: dict[str, JSONValue] = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            item["tool_calls"] = [
                _tool_call_to_openai(tool_call) for tool_call in message.tool_calls
            ]
        return item

    if isinstance(message, ToolResultMessage):
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "name": message.name,
            "content": message.content,
        }


def _tool_to_openai(tool: AgentTool) -> dict[str, JSONValue]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": dict(tool.input_schema),
        },
    }


def _tool_call_to_openai(tool_call: ToolCall) -> dict[str, JSONValue]:
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.name,
            "arguments": dumps(tool_call.arguments),
        },
    }


def _parse_sse_line(line: str) -> str | None:
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None
    return line.removeprefix("data:").strip()


def _loads_object(value: str) -> dict[str, JSONValue] | None:
    try:
        loaded = loads(value)
    except JSONDecodeError:
        return None
    if isinstance(loaded, dict):
        return loaded
    return None


def _first_choice(chunk: Mapping[str, Any]) -> Mapping[str, Any] | None:
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return None
    return choice


def _int_or_zero(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _parse_chunk_usage(raw: Mapping[str, Any]) -> Usage:
    """Parse an OpenAI-compatible ``usage`` payload into a Usage.

    Ports Pi's openai-completions.ts parseChunkUsage: ``cached_tokens`` are
    cache reads, writes are subtracted from the prompt to leave the fresh input,
    and ``completion_tokens`` already includes reasoning tokens. Cost is left
    unset (None) because Tau has no per-model pricing table.
    """
    prompt_tokens = _int_or_zero(raw.get("prompt_tokens"))
    prompt_details = raw.get("prompt_tokens_details")
    cached_tokens: int | None = None
    cache_write = 0
    if isinstance(prompt_details, Mapping):
        cached_tokens = _int_or_none(prompt_details.get("cached_tokens"))
        cache_write = _int_or_zero(prompt_details.get("cache_write_tokens"))
    # Nullish fallback, matching Pi's `cached_tokens ?? prompt_cache_hit_tokens
    # ?? 0` (DeepSeek reports cache hits in prompt_cache_hit_tokens): a reported
    # 0 does not fall through.
    if cached_tokens is None:
        cached_tokens = _int_or_none(raw.get("prompt_cache_hit_tokens"))
    cache_read = cached_tokens or 0
    fresh_input = max(0, prompt_tokens - cache_read - cache_write)
    output = _int_or_zero(raw.get("completion_tokens"))
    reasoning = None
    completion_details = raw.get("completion_tokens_details")
    if isinstance(completion_details, Mapping):
        reasoning = _int_or_zero(completion_details.get("reasoning_tokens"))
    return Usage(
        input=fresh_input,
        output=output,
        cache_read=cache_read,
        cache_write=cache_write,
        reasoning=reasoning,
        total_tokens=fresh_input + output + cache_read + cache_write,
    )


def _tool_call_deltas(delta: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    tool_calls = delta.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    return [tool_call for tool_call in tool_calls if isinstance(tool_call, Mapping)]


def _thinking_delta_text(delta: Mapping[str, Any]) -> str:
    for field_name in ("reasoning_content", "reasoning", "thinking"):
        value = delta.get(field_name)
        if isinstance(value, str) and value:
            return value
    return ""


def _is_transient_status(status_code: int) -> bool:
    return status_code in {408, 409, 425, 429} or status_code >= 500
