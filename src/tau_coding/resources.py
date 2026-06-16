"""Markdown resource path and frontmatter helpers."""

from dataclasses import dataclass, field
from pathlib import Path

from tau_agent.types import JSONValue


class ResourceError(ValueError):
    """Raised when Tau resources are invalid or cannot be expanded."""


@dataclass(frozen=True, slots=True)
class TauResourcePaths:
    """Filesystem locations for Tau markdown resources."""

    root: Path = field(default_factory=lambda: Path.home() / ".tau")

    @property
    def skills_dir(self) -> Path:
        """Return the skills directory."""
        return self.root / "skills"

    @property
    def prompts_dir(self) -> Path:
        """Return the prompt templates directory."""
        return self.root / "prompts"


def parse_markdown_resource(text: str) -> tuple[dict[str, str], str]:
    """Parse minimal YAML-like frontmatter from a markdown resource.

    Only simple `key: value` pairs are supported. This keeps resource parsing
    dependency-free and avoids evaluating arbitrary code.
    """
    if not text.startswith("---\n"):
        return {}, text

    end = text.find("\n---", 4)
    if end == -1:
        return {}, text

    raw_frontmatter = text[4:end]
    body = text[end + len("\n---") :]
    if body.startswith("\n"):
        body = body[1:]

    metadata: dict[str, str] = {}
    for line in raw_frontmatter.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition(":")
        if not separator:
            continue
        metadata[key.strip()] = value.strip().strip("\"'")
    return metadata, body


def derive_description(content: str) -> str | None:
    """Derive a short description from markdown content."""
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or None
        return stripped
    return None


def metadata_to_json(metadata: dict[str, str]) -> dict[str, JSONValue]:
    """Convert string metadata into JSON-like values."""
    return dict(metadata)
