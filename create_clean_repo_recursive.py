#!/usr/bin/env python3
# Build a timestamped clean snapshot recursively (exclude heavy stuff) and force-push.

import argparse, shutil, subprocess, sys
from datetime import datetime
from pathlib import Path

# Folders to exclude anywhere in the tree
EXCLUDE_DIRS = {
    "venv", "__pycache__", ".git", ".ipynb_checkpoints",
    "data", "qlora-neo1.3b-lungcancer"
}

# File extensions to exclude (big/artifacts)
EXCLUDE_EXTS = {".tsv", ".gz", ".joblib", ".pkl", ".bin", ".png"}

# What we DO include
INCLUDE_EXTS = {".py", ".html", ".ipynb", ".md", ".txt", ".json"}
INCLUDE_BASENAMES = {"requirements.txt", "train.jsonl", "val.jsonl"}

def run(cmd, cwd):
    print("> " + " ".join(cmd))
    if subprocess.run(cmd, cwd=cwd).returncode != 0:
        sys.exit(1)

def should_copy(f: Path, root: Path) -> bool:
    # Skip excluded dirs anywhere in the path
    for part in f.relative_to(root).parts:
        if part in EXCLUDE_DIRS:
            return False
    # Always include specific basenames
    if f.name in INCLUDE_BASENAMES:
        return True
    # Skip excluded extensions
    if f.suffix in EXCLUDE_EXTS:
        return False
    # Only include allowed extensions
    return f.suffix in INCLUDE_EXTS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-name", required=True)
    ap.add_argument("--remote", required=True)
    ap.add_argument("--branch", default="master")
    args = ap.parse_args()

    root = Path.cwd()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    clean = root.parent / f"{args.clean_name}-{ts}"
    clean.mkdir()
    print(f" Created {clean}")

    # Recursively copy, preserving structure
    for f in root.rglob("*"):
        if f.is_file() and should_copy(f, root):
            rel = f.relative_to(root)
            dst = clean / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            print(f"   {rel}")

    # Init new git repo & push
    run(["git", "init"], cwd=str(clean))
    run(["git", "remote", "add", "origin", args.remote], cwd=str(clean))
    run(["git", "add", "."], cwd=str(clean))
    run(["git", "commit", "-m", "Clean recursive code-only snapshot"], cwd=str(clean))
    run(["git", "branch", "-M", args.branch], cwd=str(clean))
    run(["git", "push", "-u", "origin", args.branch, "--force"], cwd=str(clean))
    print("\n Pushed clean recursive snapshot!")

if __name__ == "__main__":
    main()
