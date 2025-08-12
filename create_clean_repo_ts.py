#!/usr/bin/env python3
"""
create_clean_repo.py

Creates a timestamped “clean” folder next to your project, containing only:
  - All Python scripts (*.py) except this helper script
  - HTML files (*.html)
  - Jupyter notebooks (*.ipynb)
  - requirements.txt, train.jsonl, val.jsonl
  - metadata/ and examples/ directories

Then initializes a fresh Git repo there and force-pushes to your GitHub remote.
"""

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# File patterns and directories to include
INCLUDE_GLOBS = [
    "*.py",             # all Python scripts
    "*.html",           # HTML files
    "*.ipynb",          # notebooks
    "requirements.txt",
    "train.jsonl",
    "val.jsonl",
]
INCLUDE_DIRS = [
    "metadata",
    "examples",
]

def run(cmd, cwd):
    """Run a subprocess command in cwd, exit on error."""
    print(f"> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(
        description="Build & push a clean code-only snapshot to GitHub."
    )
    parser.add_argument(
        "--clean-name", required=True,
        help="Base name for the clean snapshot folder"
    )
    parser.add_argument(
        "--remote", required=True,
        help="Git remote URL (e.g. https://github.com/You/your-repo.git)"
    )
    parser.add_argument(
        "--branch", default="master",
        help="Branch name to force-push (default: master)"
    )
    args = parser.parse_args()

    src = Path.cwd()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    clean = src.parent / f"{args.clean_name}-{timestamp}"
    clean.mkdir()
    print(f"→ Created clean folder: {clean}")

    # Copy matching files
    helper = Path(__file__).name
    for pattern in INCLUDE_GLOBS:
        for f in src.glob(pattern):
            if f.name == helper:
                continue
            shutil.copy2(f, clean / f.name)
            print(f"  • Copied file: {f.name}")

    # Copy matching directories
    for d in INCLUDE_DIRS:
        srcd = src / d
        if srcd.is_dir():
            shutil.copytree(srcd, clean / d)
            print(f"  • Copied directory: {d}")

    # Initialize Git and push
    run(["git", "init"], cwd=clean)
    run(["git", "remote", "add", "origin", args.remote], cwd=clean)
    run(["git", "add", "."], cwd=clean)
    run(["git", "commit", "-m", "Clean code-only snapshot"], cwd=clean)
    run(["git", "branch", "-M", args.branch], cwd=clean)
    run(["git", "push", "-u", "origin", args.branch, "--force"], cwd=clean)

    print("\n✅ Clean snapshot pushed successfully!")

if __name__ == "__main__":
    main()
