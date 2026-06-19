"""Per-run metadata writer."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GitInfo:
    commit: str | None
    dirty: bool | None


def collect_git_info(repo_root: Path) -> GitInfo:
    git = shutil.which("git")
    if git is None:
        return GitInfo(None, None)
    rev = subprocess.run([git, "-C", str(repo_root), "rev-parse", "--short=7", "HEAD"], capture_output=True, text=True)
    if rev.returncode != 0:
        return GitInfo("nogit", None)
    status = subprocess.run([git, "-C", str(repo_root), "status", "--porcelain"], capture_output=True, text=True)
    return GitInfo(rev.stdout.strip() or "nogit", bool(status.stdout.strip()) if status.returncode == 0 else None)


def collect_package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for name in ("pandas", "numpy", "pyarrow", "pydantic", "pyyaml", "matplotlib"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "unknown"
    return out


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_dataframe(df) -> str:
    import io

    buf = io.BytesIO()
    df.to_parquet(buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def build_metadata(
    *,
    run_id: str,
    config_snapshot_relpath: str,
    input_data_hashes: dict[str, str],
    repo_root: Path,
) -> dict[str, Any]:
    git = collect_git_info(repo_root)
    return {
        "run_id": run_id,
        "config_snapshot_path": config_snapshot_relpath,
        "python_version": sys.version.split()[0],
        "package_versions": collect_package_versions(),
        "git_commit": git.commit,
        "git_dirty": git.dirty,
        "input_data_hashes": dict(input_data_hashes),
    }


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

