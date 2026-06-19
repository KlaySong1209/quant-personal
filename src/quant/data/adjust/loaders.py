"""Configurable column-mapping helpers for local vendor exports."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd


def mapping_to_dict(mapping: Any) -> dict[str, str | None]:
    if is_dataclass(mapping):
        return asdict(mapping)
    if isinstance(mapping, dict):
        return dict(mapping)
    raise TypeError("column mapping must be a dataclass or dict")


def read_mapped_csv(
    path: str | Path,
    mapping: Any,
    *,
    required: tuple[str, ...],
    optional: tuple[str, ...] = (),
) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str)
    mp = mapping_to_dict(mapping)
    out: dict[str, pd.Series] = {}
    missing: list[str] = []
    for canonical in required:
        vendor = mp.get(canonical)
        if not vendor or vendor not in raw.columns:
            missing.append(f"{canonical}->{vendor}")
            continue
        out[canonical] = raw[vendor]
    if missing:
        raise ValueError(f"missing mapped column(s): {missing}")
    for canonical in optional:
        vendor = mp.get(canonical)
        if vendor and vendor in raw.columns:
            out[canonical] = raw[vendor]
    return pd.DataFrame(out)

