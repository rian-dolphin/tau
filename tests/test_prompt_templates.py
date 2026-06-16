from pathlib import Path

import pytest

from tau_coding import (
    TauResourcePaths,
    load_prompt_templates,
    render_prompt_template,
)
from tau_coding.prompt_templates import PromptTemplate
from tau_coding.resources import ResourceError


def test_load_prompt_templates_missing_directory_returns_empty(tmp_path: Path) -> None:
    assert load_prompt_templates(TauResourcePaths(root=tmp_path)) == []


def test_load_prompt_templates_from_markdown_files(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "review.md").write_text(
        "---\ndescription: Review code\n---\nReview {{ topic }}.",
        encoding="utf-8",
    )

    templates = load_prompt_templates(TauResourcePaths(root=tmp_path))

    assert len(templates) == 1
    assert templates[0].name == "review"
    assert templates[0].description == "Review code"


def test_render_prompt_template_replaces_variables() -> None:
    template = PromptTemplate(
        name="review",
        path=Path("review.md"),
        content="Review {{ topic }} for {{ focus }}.",
    )

    assert render_prompt_template(template, {"topic": "auth", "focus": "security"}) == (
        "Review auth for security."
    )


def test_render_prompt_template_rejects_missing_variables() -> None:
    template = PromptTemplate(name="review", path=Path("review.md"), content="Review {{ topic }}.")

    with pytest.raises(ResourceError, match="Missing prompt template variable"):
        render_prompt_template(template, {})
