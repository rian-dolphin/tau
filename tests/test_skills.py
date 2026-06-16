from pathlib import Path

import pytest

from tau_coding import TauResourcePaths, build_skill_index, expand_skill_command, load_skills
from tau_coding.resources import ResourceError


def test_load_skills_missing_directory_returns_empty(tmp_path: Path) -> None:
    assert load_skills(TauResourcePaths(root=tmp_path)) == []


def test_load_skills_from_directory_and_file(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    (skills_dir / "python-testing").mkdir(parents=True)
    (skills_dir / "python-testing" / "SKILL.md").write_text(
        "---\ndescription: Test Python code\n---\n# Python Testing\nUse pytest.",
        encoding="utf-8",
    )
    (skills_dir / "git-review.md").write_text("# Git Review\nReview diffs.", encoding="utf-8")

    skills = load_skills(TauResourcePaths(root=tmp_path))

    assert [skill.name for skill in skills] == ["git-review", "python-testing"]
    assert skills[0].description == "Git Review"
    assert skills[1].description == "Test Python code"


def test_load_skills_rejects_duplicate_names(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    (skills_dir / "dup").mkdir(parents=True)
    (skills_dir / "dup" / "SKILL.md").write_text("# Directory skill", encoding="utf-8")
    (skills_dir / "dup.md").write_text("# File skill", encoding="utf-8")

    with pytest.raises(ResourceError, match="Duplicate skill name"):
        load_skills(TauResourcePaths(root=tmp_path))


def test_expand_skill_command_includes_skill_and_user_request(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "testing.md").write_text("# Testing\nRun pytest.", encoding="utf-8")
    skills = load_skills(TauResourcePaths(root=tmp_path))

    expanded = expand_skill_command("/skill:testing add parser tests", skills)

    assert expanded is not None
    assert '<skill name="testing">' in expanded
    assert "Run pytest." in expanded
    assert "User request:\nadd parser tests" in expanded


def test_expand_skill_command_returns_none_for_normal_prompt(tmp_path: Path) -> None:
    assert expand_skill_command("hello", load_skills(TauResourcePaths(root=tmp_path))) is None


def test_expand_skill_command_rejects_unknown_skill() -> None:
    with pytest.raises(ResourceError, match="Unknown skill"):
        expand_skill_command("/skill:missing", [])


def test_build_skill_index(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "testing.md").write_text(
        "---\ndescription: Test things\n---\nBody",
        encoding="utf-8",
    )

    assert build_skill_index(load_skills(TauResourcePaths(root=tmp_path))) == (
        "Available skills:\n- testing: Test things"
    )
