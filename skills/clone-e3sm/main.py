#!/usr/bin/env python3
"""clone-e3sm skill

Clones git@github.com:E3SM-Project/E3SM.git into a local ``project``
directory (or any caller-specified parent directory).  The clone is done
with ``--recurse-submodules`` so all external libraries bundled as git
submodules are fetched automatically.  The parent directory is created
automatically when it does not yet exist.

Expected input (JSON via STDIN or first CLI argument)::

    {
        "target_dir": "project",   // optional, default "project"
        "branch":     "master",    // optional
        "depth":      1            // optional – shallow clone
    }

Output (JSON)::

    {
        "clone_path": "/abs/path/to/project/E3SM",
        "status":     "success" | "skipped",
        "message":    "..."
    }
"""

import json
import os
import subprocess
import sys
from pathlib import Path

E3SM_REPO = "git@github.com:E3SM-Project/E3SM.git"
REPO_DIR_NAME = "E3SM"


def _load_input() -> dict:
    """Return the parsed input payload from STDIN or argv."""
    if not sys.stdin.isatty():
        raw = sys.stdin.read().strip()
        if raw:
            return json.loads(raw)
    if len(sys.argv) > 1:
        try:
            return json.loads(sys.argv[1])
        except json.JSONDecodeError:
            pass
    return {}


def _check_git_lfs() -> None:
    """Raise a clear error if git-lfs is not installed."""
    result = subprocess.run(
        ["git", "lfs", "version"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "git-lfs is not installed but is required by E3SM submodules (e.g. fates). "
            "Install it first:\n"
            "  macOS:  brew install git-lfs && git lfs install\n"
            "  Linux:  apt install git-lfs  && git lfs install"
        )


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a command, streaming output to stderr, and return the result."""
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
    )


def main() -> None:
    payload = _load_input()

    target_dir: str = payload.get("target_dir") or "project"
    branch: str | None = payload.get("branch")
    depth: int | None = payload.get("depth")

    # Resolve target_dir relative to cwd
    parent = Path(target_dir).expanduser()
    if not parent.is_absolute():
        parent = Path.cwd() / parent

    clone_path = parent / REPO_DIR_NAME

    # ------------------------------------------------------------------
    # Pre-flight: git-lfs must be present (required by fates submodule)
    # ------------------------------------------------------------------
    _check_git_lfs()

    # ------------------------------------------------------------------
    # Create the parent directory if needed
    # ------------------------------------------------------------------
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {parent}", file=sys.stderr)
    else:
        print(f"Directory already exists: {parent}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Skip if already cloned
    # ------------------------------------------------------------------
    if clone_path.exists():
        result = {
            "clone_path": str(clone_path),
            "status": "skipped",
            "message": (
                f"{clone_path} already exists – skipping clone. "
                "Run 'git pull' inside that directory to update."
            ),
        }
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # ------------------------------------------------------------------
    # Build the git clone command
    # ------------------------------------------------------------------
    cmd = ["git", "clone", "--recurse-submodules", E3SM_REPO, REPO_DIR_NAME]

    if branch:
        cmd += ["--branch", branch]

    if depth is not None:
        cmd += ["--depth", str(int(depth))]
        # Shallow clones need this so submodule histories are also limited
        cmd += ["--shallow-submodules"]

    print(f"Running: {' '.join(cmd)}", file=sys.stderr)

    try:
        _run(cmd, cwd=parent)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"git clone failed with exit code {exc.returncode}. "
            "Make sure your SSH key is configured for GitHub."
        ) from exc

    result = {
        "clone_path": str(clone_path),
        "status": "success",
        "message": f"E3SM cloned successfully to {clone_path}.",
    }
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
