"""
README.md template processor for cutde.

This module automatically generates the README.md file from a template, ensuring
dynamic content stays synchronized with the project configuration.

Usage:
    python docs/tmpl_readme.py

The script reads from:
- docs/README-tmpl.md: The template file
- pyproject.toml: Project configuration for version requirements

And writes to:
- README.md: The generated README file
"""

import re
import tomllib
from pathlib import Path

from mako.template import Template


def get_python_version_requirement() -> str:
    """Extract Python version requirement from pyproject.toml"""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        raise FileNotFoundError("pyproject.toml not found")

    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    requires_python = data["project"]["requires-python"]
    # Extract the minimum version from ">=3.8" format
    match = re.search(r">=(\d+\.\d+)", requires_python)
    if not match:
        raise ValueError(
            f"Could not parse Python version requirement: {requires_python}"
        )

    return match.group(1)


def main() -> None:
    """Generate README.md from template."""
    python_version = get_python_version_requirement()

    template_path = Path("docs/README-tmpl.md")
    tmpl_txt = template_path.read_text()

    # Handle Mako template syntax
    tmpl_txt = tmpl_txt.replace("##", "${'##'}")
    tmpl = Template(tmpl_txt)

    result = tmpl.render(python_version=python_version)
    assert isinstance(result, str)

    Path("README.md").write_text(result)


if __name__ == "__main__":
    main()
