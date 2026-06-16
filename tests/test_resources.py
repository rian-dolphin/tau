from pathlib import Path

from tau_coding import TauResourcePaths
from tau_coding.resources import derive_description, parse_markdown_resource


def test_resource_paths_use_tau_subdirectories(tmp_path: Path) -> None:
    paths = TauResourcePaths(root=tmp_path)

    assert paths.skills_dir == tmp_path / "skills"
    assert paths.prompts_dir == tmp_path / "prompts"


def test_parse_frontmatter_description() -> None:
    metadata, body = parse_markdown_resource(
        "---\ndescription: Write tests\n---\n# Testing\nUse pytest."
    )

    assert metadata == {"description": "Write tests"}
    assert body == "# Testing\nUse pytest."


def test_derive_description_uses_first_heading_or_paragraph() -> None:
    assert derive_description("\n# Title\nBody") == "Title"
    assert derive_description("\nFirst paragraph\nMore") == "First paragraph"
