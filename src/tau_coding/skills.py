"""Markdown skill loading and expansion."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tau_coding.resources import (
    ResourceError,
    TauResourcePaths,
    derive_description,
    parse_markdown_resource,
)


@dataclass(frozen=True, slots=True)
class Skill:
    """A markdown skill resource."""

    name: str
    path: Path
    content: str
    description: str | None = None


def load_skills(paths: TauResourcePaths | None = None) -> list[Skill]:
    """Load markdown skills from `paths.skills_dir` in deterministic order."""
    resource_paths = paths or TauResourcePaths()
    skills_dir = resource_paths.skills_dir
    if not skills_dir.exists():
        return []

    skills: list[Skill] = []
    seen: set[str] = set()

    for path in sorted(skills_dir.iterdir(), key=lambda item: item.name):
        skill_path: Path | None = None
        name = path.stem
        if path.is_dir():
            skill_path = path / "SKILL.md"
            name = path.name
            if not skill_path.exists():
                continue
        elif path.is_file() and path.suffix.lower() == ".md":
            skill_path = path
        else:
            continue

        if name in seen:
            raise ResourceError(f"Duplicate skill name: {name}")
        seen.add(name)
        skills.append(_load_skill(name, skill_path))

    return sorted(skills, key=lambda skill: skill.name)


def expand_skill_command(text: str, skills: Sequence[Skill]) -> str | None:
    """Expand `/skill:name` prompt text, or return None for non-skill text."""
    stripped = text.strip()
    if not stripped.startswith("/skill:"):
        return None

    command, separator, request = stripped.partition(" ")
    name = command.removeprefix("/skill:").strip()
    if not name:
        raise ResourceError("Skill command must include a skill name")

    skill_by_name = {skill.name: skill for skill in skills}
    skill = skill_by_name.get(name)
    if skill is None:
        raise ResourceError(f"Unknown skill: {name}")

    sections = [
        "Use the following skill instructions:",
        f'<skill name="{skill.name}">\n{skill.content.strip()}\n</skill>',
    ]
    if separator and request.strip():
        sections.append(f"User request:\n{request.strip()}")
    return "\n\n".join(sections)


def build_skill_index(skills: Sequence[Skill]) -> str:
    """Build a concise index of available skills for future system prompt assembly."""
    if not skills:
        return "Available skills: none"
    lines = ["Available skills:"]
    for skill in sorted(skills, key=lambda item: item.name):
        description = skill.description or "No description"
        lines.append(f"- {skill.name}: {description}")
    return "\n".join(lines)


def _load_skill(name: str, path: Path) -> Skill:
    raw = path.read_text(encoding="utf-8")
    metadata, content = parse_markdown_resource(raw)
    description = metadata.get("description") or derive_description(content)
    return Skill(name=name, path=path, content=content, description=description)
