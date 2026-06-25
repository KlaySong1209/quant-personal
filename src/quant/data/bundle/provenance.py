"""Append-only audit log for bundle operations.

One JSONL line per fetch/ingest/migrate event. Existing lines are NEVER
modified — corruption (or attempted modification) is treated as a bug.

This is the project's "诚实优先" principle applied to the data layer:
when the user (or a future you) asks "where did this 2026-06-15 row come
from?" the answer must always be answerable from disk.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

ProvenanceOp = Literal["fetch", "ingest", "migrate", "create", "delete"]
ProvenanceStatus = Literal["ok", "partial", "failed"]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ProvenanceRecord:
    """One audit-log line.

    *details* captures op-specific context (file paths, row counts, errors).
    Keep it small and JSON-serializable.
    """

    op: ProvenanceOp
    status: ProvenanceStatus
    ts: str = field(default_factory=_utcnow_iso)
    source: str | None = None
    bundle: str | None = None
    raw_path: str | None = None
    symbols: list[str] | None = None
    rows: int | None = None
    error: str | None = None
    details: dict[str, Any] | None = None

    def to_jsonl(self) -> str:
        d = {k: v for k, v in asdict(self).items() if v is not None}
        return json.dumps(d, ensure_ascii=False, sort_keys=True)


class ProvenanceLog:
    """File-backed append-only log.

    Reads return records oldest-first. Writes are atomic per-line (open in
    append mode and write a single line; no need for fsync — provenance is
    best-effort durable, the bundle files are the source of truth).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, record: ProvenanceRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(record.to_jsonl())
            f.write("\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"corrupt provenance line in {self.path}: {exc}"
                ) from exc
        return out

    def tail(self, n: int = 20) -> list[dict[str, Any]]:
        rows = self.read_all()
        return rows[-n:] if n > 0 else rows
