#!/usr/bin/env python3
import argparse, shutil, subprocess, sys
from datetime import datetime
from pathlib import Path

EXCLUDE_DIRS = {"venv","__pycache__", ".git",".ipynb_checkpoints","data","qlora-neo1.3b-lungcancer"}
EXCLUDE_EXTS = {".tsv",".gz",".joblib",".pkl",".bin",".png",".pt",".pth"}
INCLUDE_EXTS = {".py",".html",".ipynb",".md",".txt",".json",".css",".js"}
INCLUDE_BASENAMES = {"requirements.txt","train.jsonl","val.jsonl","README.md","README",".gitignore"}

def run(cmd, cwd):
    print("> "+" ".join(cmd))
    if subprocess.run(cmd, cwd=cwd).returncode != 0: sys.exit(1)

def should_copy(f: Path, root: Path) -> bool:
    for part in f.relative_to(root).parts:
        if part in EXCLUDE_DIRS: return False
    if f.name in INCLUDE_BASENAMES: return True
    if f.suffix in EXCLUDE_EXTS: return False
    return f.suffix in INCLUDE_EXTS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--clean-name", required=True)
    ap.add_argument("--remote", required=True)
    ap.add_argument("--branch", default="master")
    args = ap.parse_args()

    src = Path(args.source_dir).resolve()
    if not src.is_dir(): print(f"Source dir does not exist: {src}"); sys.exit(1)

    ts = __import__("datetime").datetime.now().strftime("%Y%m%d-%H%M%S")
    clean = src.parent / f"{args.clean_name}-{ts}"
    clean.mkdir()
    print(f" Created {clean}")

    copied = 0
    for f in src.rglob("*"):
        if f.is_file() and should_copy(f, src):
            rel = f.relative_to(src)
            dst = clean / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            print(f"   {rel}")
            copied += 1
    if copied == 0:
        print("Warning: nothing copied. Check --source-dir."); sys.exit(1)

    run(["git","init"], cwd=str(clean))
    run(["git","remote","add","origin", args.remote], cwd=str(clean))
    run(["git","add","."], cwd=str(clean))
    run(["git","commit","-m","Clean recursive code-only snapshot"], cwd=str(clean))
    run(["git","branch","-M", args.branch], cwd=str(clean))
    run(["git","push","-u","origin", args.branch, "--force"], cwd=str(clean))
    print("\n Pushed clean snapshot!")

if __name__ == "__main__": main()
