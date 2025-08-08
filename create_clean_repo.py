#!/usr/bin/env python3
"""
create_clean_repo_ts.py

Creates a timestamped clean, code-only snapshot to push to GitHub,
avoiding any need to delete old folders.
"""

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Patterns and directories to include
INCLUDE_GLOBS = ["*.py", "*.html", "*.ipynb", "requirements.txt", "train.jsonl", "val.jsonl"]
INCLUDE_DIRS = ["metadata", "examples"]

def run(cmd, cwd):
    print(f"> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        sys.exit(result.returncode)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clean-name", required=True, help="Base name for clean folder")
    p.add_argument("--remote",     required=True, help="Git remote URL")
    p.add_argument("--branch",     default="master", help="Branch to push")
    args = p.parse_args()

    src = Path.cwd()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    clean = src.parent / f"{args.clean_name}-{timestamp}"

    clean.mkdir()
    print(f"Created new clean folder: {clean}")

    # Copy files
    for pattern in INCLUDE_GLOBS:
        for f in src.glob(pattern):
            shutil.copy2(f, clean / f.name)
            print(f"Copied: {f.name}")

    # Copy directories
    for d in INCLUDE_DIRS:
        srcd = src / d
        if srcd.is_dir():
            shutil.copytree(srcd, clean / d)
            print(f"Copied dir: {d}")

    # Initialize & push Git
    run(["git", "init"], cwd=clean)
    run(["git", "remote", "add", "origin", args.remote], cwd=clean)
    run(["git", "add", "."], cwd=clean)
    run(["git", "commit", "-m", "Clean code-only snapshot"], cwd=clean)
    run(["git", "branch", "-M", args.branch], cwd=clean)
    run(["git", "push", "-u", "origin", args.branch, "--force"], cwd=clean)

    print("✅ Pushed clean snapshot!")

if __name__ == "__main__":
    main()
