"""Bundle manifest model: JSON-schema-validated, language-neutral on disk,
Pydantic-frozen in memory.

The schema (``manifest.schema.json``) ships next to this file inside the
``quant.data.bundle`` package. We load + cache it once and validate on every
read/write; the in-memory representation is a Pydantic model that matches the
schema field-for-field.

Mirroring the project's existing config style ([src/quant/config/schema.py]):
- ``frozen=True`` — manifests are values, mutate via ``model_copy``
- ``extra="forbid"`` — unknown fields fail loudly (Pydantic) and at the
  schema layer (jsonschema)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import jsonschema
from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"
SUPPORTED_MARKETS = frozenset({"a_share_cn"})

# Schema is a source-code artifact, not user data: it lives in the package.
SCHEMA_PATH = Path(__file__).resolve().with_name("manifest.schema.json")


@lru_cache(maxsize=1)
def _schema() -> dict[str, Any]:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"bundle manifest schema not found: {SCHEMA_PATH}")
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _validate(data: dict[str, Any]) -> None:
    """Raise jsonschema.ValidationError if *data* doesn't match the on-disk schema."""
    jsonschema.validate(instance=data, schema=_schema())


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DateRange(_Frozen):
    first: str  # ISO date, e.g. "2020-01-01"
    last: str   # ISO date


class AdjustmentMeta(_Frozen):
    convention: Literal["forward", "backward", "none"]
    method: Literal[
        "provided_adjusted_close",
        "provided_adjustment_factor",
        "built_from_dividends_splits",
        "raw_unadjusted",
        "mootdx_qfq",
        "mootdx_hfq",
    ]


class CalendarMeta(_Frozen):
    source: Literal["synthetic", "file", "mootdx"]
    exchange: str


class FreshnessMeta(_Frozen):
    expected_through: str  # ISO date
    actual_through: str    # ISO date
    status: Literal["fresh", "stale", "no_data"]


class BundleManifest(_Frozen):
    name: str
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    market: Literal["a_share_cn"] = "a_share_cn"
    created_at: str  # ISO datetime
    updated_at: str  # ISO datetime
    symbols: list[str] = Field(min_length=1)
    date_range: DateRange
    source_chain: list[str] = Field(min_length=1)
    adjustment: AdjustmentMeta
    row_count: int = Field(ge=0)
    calendar: CalendarMeta
    freshness: FreshnessMeta

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BundleManifest":
        """Validate against the JSON schema and parse into a frozen Pydantic model.

        Two-layer validation: jsonschema first (language-neutral contract),
        then Pydantic (typing + enum coercion). Either one rejects ⇒ fail-fast.
        """
        _validate(data)
        return cls.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        out = self.model_dump(mode="python")
        _validate(out)
        return out

    @classmethod
    def load(cls, path: str | Path) -> "BundleManifest":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"manifest not found: {p}")
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"manifest root must be an object: {p}")
        return cls.from_dict(raw)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        return target


# ---------------------------------------------------------------------------
# Catalog model (data/bundles/catalog.json)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CatalogEntry:
    name: str
    created_at: str  # ISO datetime
    path: str        # directory name relative to data/bundles/

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "created_at": self.created_at, "path": self.path}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CatalogEntry":
        return cls(
            name=str(data["name"]),
            created_at=str(data["created_at"]),
            path=str(data["path"]),
        )
