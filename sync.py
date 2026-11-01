#!/usr/bin/env python3
"""
Cross-platform Git synchronizer for Streamflow_Project.

Run from anywhere inside the repository:
    python3 sync.py

Sync policy:
- Include: sync.py, README.md, .gitignore, scripts/, docs/,
  data/processed/, and models/
- Exclude: data/raw/, .agents/, caches, OS metadata, and log files
- Use GitHub's origin/main as the shared source of truth
- Commit local in-scope changes, rebase onto origin/main, then push
- If Git reports a merge conflict, stop and leave it for manual resolution
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


BRANCH = "main"
REMOTE = "origin"

# These paths are intentionally synchronized.
SYNC_PATHS = [
    "sync.py",
    "README.md",
    ".gitignore",
    "scripts",
    "docs",
    "data/processed",
    "models",
]

# Patterns appended to .gitignore so large/raw and machine-specific files
# are not accidentally included. The negation rules reopen data/processed
# even if an older .gitignore contains a broad "data/" rule.
GITIGNORE_RULES = """
# --- Added by sync.py ---
# Keep raw input datasets local; synchronize processed outputs.
!data/
data/raw/
!data/processed/
!models/

# Machine-specific and temporary files
.DS_Store
**/.DS_Store
__pycache__/
**/__pycache__/
*.py[cod]
*.log
"""


class SyncError(Exception):
    pass


def run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if check and result.returncode != 0:
        raise SyncError(result.stdout.strip() or f"Command failed: {' '.join(args)}")
    return result


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(["git", *args], check=check)


def say(message: str = "") -> None:
    print(message, flush=True)


def find_repo_root() -> Path:
    result = git("rev-parse", "--show-toplevel", check=False)
    if result.returncode != 0:
        raise SyncError(
            "This folder is not inside a Git repository.\n"
            "Put sync.py in the project root and run it from that folder."
        )
    return Path(result.stdout.strip()).resolve()


def ensure_gitignore(root: Path) -> bool:
    path = root / ".gitignore"
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if "# --- Added by sync.py ---" in old:
        return False
    separator = "" if not old or old.endswith("\n") else "\n"
    path.write_text(old + separator + GITIGNORE_RULES.lstrip("\n"), encoding="utf-8")
    return True


def check_branch() -> None:
    current = git("branch", "--show-current").stdout.strip()
    if current != BRANCH:
        raise SyncError(
            f"Current branch is '{current or '(detached HEAD)'}', not '{BRANCH}'.\n"
            f"Switch to the project branch first: git switch {BRANCH}"
        )


def check_no_operation_in_progress() -> None:
    git_dir = Path(git("rev-parse", "--git-dir").stdout.strip())
    if not git_dir.is_absolute():
        git_dir = (Path.cwd() / git_dir).resolve()
    markers = ["rebase-merge", "rebase-apply", "MERGE_HEAD", "CHERRY_PICK_HEAD"]
    active = [name for name in markers if (git_dir / name).exists()]
    if active:
        raise SyncError(
            "Git has an unfinished operation: " + ", ".join(active) + ".\n"
            "Resolve the conflict manually, then run the appropriate command:\n"
            "  git add <resolved-files>\n"
            "  git rebase --continue     (if a rebase is in progress)\n"
            "After it finishes, run python3 sync.py again."
        )


def ensure_remote() -> None:
    remotes = git("remote").stdout.split()
    if REMOTE not in remotes:
        raise SyncError(
            "No 'origin' remote is configured.\n"
            "Add it with:\n"
            "  git remote add origin https://github.com/abhitejj/Streamflow_Project.git"
        )
    url = git("remote", "get-url", REMOTE).stdout.strip()
    if "github.com/abhitejj/Streamflow_Project" not in url:
        say(f"Notice: origin currently points to: {url}")


def ensure_upstream() -> None:
    # Set tracking only when origin/main exists and the local branch has a commit.
    remote_ref = git("show-ref", "--verify", "--quiet", f"refs/remotes/{REMOTE}/{BRANCH}", check=False)
    if remote_ref.returncode != 0:
        raise SyncError(
            f"{REMOTE}/{BRANCH} was not found. Run 'git fetch {REMOTE}' and check the remote."
        )
    local_head = git("rev-parse", "--verify", "HEAD", check=False)
    if local_head.returncode != 0:
        raise SyncError(
            "This local branch has no commit yet. Connect it to the existing GitHub history "
            "before using sync.py:\n"
            "  git reset --mixed origin/main\n"
            "  git branch --set-upstream-to=origin/main main"
        )
    git("branch", "--set-upstream-to", f"{REMOTE}/{BRANCH}", BRANCH)


def path_is_in_scope(name: str) -> bool:
    normalized = name.replace("\\", "/")
    return (
        normalized == "sync.py"
        or normalized == "README.md"
        or normalized == ".gitignore"
        or normalized.startswith("scripts/")
        or normalized.startswith("docs/")
        or normalized.startswith("data/processed/")
        or normalized.startswith("models/")
    )


def reject_staged_changes_outside_scope() -> None:
    names = git("diff", "--cached", "--name-only", "-z").stdout.split("\0")
    outside = [name for name in names if name and not path_is_in_scope(name)]
    if outside:
        listing = "\n".join(f"  {name}" for name in outside)
        raise SyncError(
            "There are already staged changes outside the sync scope. "
            "I will not include them in a commit:\n"
            f"{listing}\n"
            "Commit or unstage those files separately, then rerun sync.py."
        )


def stage_sync_paths(root: Path) -> None:
    # Add only explicitly approved paths. Never stage the whole repository.
    for item in SYNC_PATHS:
        path = root / item
        # For tracked paths that have been deleted, git add -A still needs
        # the pathspec, so include it if it exists or is known to Git.
        tracked = git("ls-files", "--error-unmatch", item, check=False).returncode == 0
        if path.exists() or tracked:
            git("add", "-A", "--", item)

    # Remove any accidentally staged machine-specific/cache/log files that
    # are within an otherwise synchronized directory.
    staged = git("diff", "--cached", "--name-only", "-z").stdout.split("\0")
    for name in [n for n in staged if n]:
        base = Path(name).name
        if (
            base == ".DS_Store"
            or base.endswith(".log")
            or "__pycache__" in Path(name).parts
            or base.endswith((".pyc", ".pyo"))
        ):
            git("reset", "-q", "HEAD", "--", name, check=False)


def check_github_file_sizes() -> None:
    # GitHub blocks ordinary Git files larger than 100 MiB. Check staged
    # files before committing so an oversized model does not create a
    # commit that cannot be pushed.
    names = git("diff", "--cached", "--name-only", "-z").stdout.split("\0")
    oversized: list[tuple[str, int]] = []
    limit = 100 * 1024 * 1024
    for name in [n for n in names if n]:
        candidate = Path(name)
        if candidate.is_file():
            size = candidate.stat().st_size
            if size >= limit:
                oversized.append((name, size))
    if oversized:
        details = "\n".join(
            f"  {name}: {size / (1024 * 1024):.1f} MiB"
            for name, size in oversized
        )
        raise SyncError(
            "These staged files are at least 100 MiB and cannot be pushed "
            "using ordinary GitHub Git storage:\n"
            f"{details}\n"
            "Use Git LFS for these model files, or exclude them from Git."
        )



def commit_local_changes() -> bool:
    reject_staged_changes_outside_scope()
    check_github_file_sizes()
    staged = git("diff", "--cached", "--quiet", check=False)
    if staged.returncode == 0:
        return False
    say("Committing your local project changes...")
    result = git("commit", "-m", f"Sync from {os.uname().nodename}")
    say(result.stdout.strip())
    return True


def main() -> int:
    say("=" * 62)
    say("Streamflow Project synchronizer")
    say("=" * 62)

    root = find_repo_root()
    os.chdir(root)
    say(f"Project: {root}")

    check_branch()
    check_no_operation_in_progress()
    ensure_remote()

    if ensure_gitignore(root):
        say("Updated .gitignore with sync safety rules.")

    # Ensure the ignore-file update itself is included in the next sync.
    git("add", "--", ".gitignore")
    stage_sync_paths(root)
    reject_staged_changes_outside_scope()

    commit_local_changes()

    say(f"\nFetching {REMOTE}/{BRANCH}...")
    say(git("fetch", REMOTE).stdout.strip())

    ensure_upstream()

    say("\nRebasing local work onto the latest GitHub version...")
    rebase = git("rebase", f"{REMOTE}/{BRANCH}", check=False)
    if rebase.returncode != 0:
        say(rebase.stdout.strip())
        say(
            "\nSYNC PAUSED: Git found a conflict or could not complete the rebase.\n"
            "Your local changes and Git's conflict information have been preserved.\n\n"
            "To resolve manually:\n"
            "  1. Run: git status\n"
            "  2. Open each conflicted file and resolve the conflict markers.\n"
            "  3. Run: git add <resolved-files>\n"
            "  4. Run: git rebase --continue\n"
            "  5. Run: python3 sync.py\n"
            "\nDo not use 'git reset --hard' or 'git rebase --abort' unless "
            "you deliberately want to discard/restart the operation."
        )
        return 2

    say("\nPushing synchronized changes to GitHub...")
    push = git("push", REMOTE, BRANCH, check=False)
    if push.returncode != 0:
        say(push.stdout.strip())
        say(
            "\nPush did not complete. Your local commit is preserved.\n"
            "Check your connection/permissions, then run python3 sync.py again."
        )
        return 3

    say(push.stdout.strip() or "Push complete.")
    say("\nChecking final status...")
    status = git("status", "--short", "--branch")
    say(status.stdout.strip())

    counts = git("rev-list", "--left-right", "--count", f"{REMOTE}/{BRANCH}...HEAD").stdout.strip()
    if counts == "0\t0":
        say("\nSUCCESS: This machine and GitHub are synchronized.")
    else:
        say("\nSync finished, but Git reports remaining divergence. Review git status.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SyncError as exc:
        say(f"\nSYNC STOPPED:\n{exc}")
        raise SystemExit(1)
    except KeyboardInterrupt:
        say("\nInterrupted. Any completed Git operations are preserved.")
        raise SystemExit(130)
