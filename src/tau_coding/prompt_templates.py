"""Markdown prompt template loading and rendering."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from tau_coding.resources import (
    ResourceError,
    TauResourcePaths,
    derive_description,
    parse_markdown_resource,
)

_TEMPLATE_VARIABLE_RE = re.compile(r"{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}}")


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """A markdown prompt template resource."""

    name: str
    path: Path
    content: str
    description: str | None = None


def load_prompt_templates(paths: TauResourcePaths | None = None) -> list[PromptTemplate]:
    """Load markdown prompt templates from `paths.prompts_dir`."""
    resource_paths = paths or TauResourcePaths()
    prompts_dir = resource_paths.prompts_dir
    if not prompts_dir.exists():
        return []

    templates: list[PromptTemplate] = []
    seen: set[str] = set()
    for path in sorted(prompts_dir.glob("*.md"), key=lambda item: item.name):
        name = path.stem
        if name in seen:
            raise ResourceError(f"Duplicate prompt template name: {name}")
        seen.add(name)
        templates.append(_load_prompt_template(name, path))
    return templates


def render_prompt_template(template: PromptTemplate, variables: Mapping[str, str]) -> str:
    """Render a prompt template using `{{ variable }}` placeholders."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        value = variables.get(name)
        if value is None:
            raise ResourceError(f"Missing prompt template variable: {name}")
        return value

    return _TEMPLATE_VARIABLE_RE.sub(replace, template.content)


def _load_prompt_template(name: str, path: Path) -> PromptTemplate:
    raw = path.read_text(encoding="utf-8")
    metadata, content = parse_markdown_resource(raw)
    description = metadata.get("description") or derive_description(content)
    return PromptTemplate(name=name, path=path, content=content, description=description)
