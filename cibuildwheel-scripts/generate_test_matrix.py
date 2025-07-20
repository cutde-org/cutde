#!/usr/bin/env python3
"""
Generate test matrix for cutde wheel testing based on the original matrix configuration.
Filters platforms based on event type and run-on-pr flags.
"""

import argparse
import json

import yaml  # type: ignore

# Original matrix configuration with run-on-pr flags added
MATRIX_CONFIG = """
include:
  - os: ubuntu-latest
    platform: ubuntu-latest
    python-cmd: python
    setup-python: true
    python-filter: ""
    run-on-pr: true

  - os: ubuntu-24.04-arm
    platform: ubuntu-24.04-arm
    python-cmd: python
    setup-python: true
    python-filter: ""
    run-on-pr: false

  - os: windows-latest
    platform: windows-latest
    python-cmd: python
    setup-python: true
    python-filter: ""
    run-on-pr: true

  - os: macos-13
    platform: macos-13
    python-cmd: python
    setup-python: true
    python-filter: ""
    run-on-pr: false

  - os: macos-14
    platform: macos-14
    python-cmd: python
    setup-python: true
    python-filter: ""
    run-on-pr: true

  - os: ubuntu-latest
    platform: alpine-musllinux-cp310
    container: python:3.10-alpine
    python-cmd: python
    setup-python: false
    python-filter: "-cp310-"
    run-on-pr: false

  - os: ubuntu-latest
    platform: alpine-musllinux-cp311
    container: python:3.11-alpine
    python-cmd: python
    setup-python: false
    python-filter: "-cp311-"
    run-on-pr: false

  - os: ubuntu-latest
    platform: alpine-musllinux-cp312
    container: python:3.12-alpine
    python-cmd: python
    setup-python: false
    python-filter: "-cp312-"
    run-on-pr: false

  - os: ubuntu-latest
    platform: alpine-musllinux-cp313
    container: python:3.13-alpine
    python-cmd: python
    setup-python: false
    python-filter: "-cp313-"
    run-on-pr: true
"""


def generate_matrix(event_type: str) -> dict:
    """Generate matrix based on event type and run-on-pr flags."""
    # Parse the original matrix
    matrix_data = yaml.safe_load(MATRIX_CONFIG)

    if event_type == "pull_request":
        # Filter to only platforms with run-on-pr: true
        filtered_platforms = [
            platform
            for platform in matrix_data["include"]
            if platform.get("run-on-pr", False)
        ]
    else:
        # Use all platforms for main/release builds
        filtered_platforms = matrix_data["include"]

    return {"include": filtered_platforms}


def main():
    parser = argparse.ArgumentParser(description="Generate test matrix for cutde")
    parser.add_argument(
        "--event-type",
        choices=["pull_request", "push", "release"],
        required=True,
        help="GitHub event type",
    )

    args = parser.parse_args()

    # Generate and output matrix
    matrix = generate_matrix(args.event_type)
    print(json.dumps(matrix, separators=(",", ":")))


if __name__ == "__main__":
    main()
