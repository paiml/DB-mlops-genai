"""Tests for repository structure validation."""

from pathlib import Path


def test_course3_demos_exist():
    """Course 3 demos directory should exist."""
    demos = Path("demos/course3")
    assert demos.exists(), "demos/course3 directory must exist"


def test_course4_demos_exist():
    """Course 4 demos directory should exist."""
    demos = Path("demos/course4")
    assert demos.exists(), "demos/course4 directory must exist"


def test_course3_labs_exist():
    """Course 3 labs directory should exist."""
    labs = Path("labs/course3")
    assert labs.exists(), "labs/course3 directory must exist"


def test_course4_labs_exist():
    """Course 4 labs directory should exist."""
    labs = Path("labs/course4")
    assert labs.exists(), "labs/course4 directory must exist"


def test_docs_outline_exists():
    """Course outline should exist."""
    outline = Path("docs/outline.md")
    assert outline.exists(), "docs/outline.md must exist"


def test_readme_has_required_sections():
    """README should have required sections."""
    readme = Path("README.md").read_text()
    required_sections = ["Installation", "Usage", "Contributing"]
    for section in required_sections:
        assert f"## {section}" in readme, f"README missing ## {section} section"
