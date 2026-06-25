"""BundleCatalog: index of all bundles under ``data/bundles/``.

Persists to ``data/bundles/catalog.json``. The catalog is purely an index —
truth lives in each bundle's ``manifest.json``. The catalog can be rebuilt
from disk at any time via :meth:`BundleCatalog.rescan`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from quant.data.bundle.manifest import CatalogEntry
from quant.data.bundle.store import BundleStore

CATALOG_FILENAME = "catalog.json"
CATALOG_SCHEMA_VERSION = "1.0"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CatalogError(Exception):
    pass


@dataclass
class BundleCatalog:
    root: Path
    entries: list[CatalogEntry]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, root: str | Path) -> "BundleCatalog":
        """Load existing catalog, or return an empty one anchored at *root*."""
        root_path = Path(root)
        catalog_path = root_path / CATALOG_FILENAME
        if not catalog_path.exists():
            return cls(root=root_path, entries=[])
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise CatalogError(f"catalog must be a JSON object: {catalog_path}")
        version = raw.get("schema_version")
        if version != CATALOG_SCHEMA_VERSION:
            raise CatalogError(
                f"unsupported catalog schema_version {version!r}; expected {CATALOG_SCHEMA_VERSION}"
            )
        entries = [CatalogEntry.from_dict(e) for e in raw.get("bundles", [])]
        return cls(root=root_path, entries=entries)

    @classmethod
    def rescan(cls, root: str | Path) -> "BundleCatalog":
        """Rebuild the catalog by scanning *root* for valid bundle directories.

        Useful when the catalog file is missing/stale but the bundle
        directories themselves are intact.
        """
        root_path = Path(root)
        entries: list[CatalogEntry] = []
        if root_path.exists():
            for child in sorted(root_path.iterdir()):
                if not child.is_dir():
                    continue
                store = BundleStore(child)
                if not store.exists():
                    continue
                try:
                    m = store.manifest()
                except Exception:
                    continue
                entries.append(
                    CatalogEntry(name=m.name, created_at=m.created_at, path=child.name)
                )
        catalog = cls(root=root_path, entries=entries)
        catalog.save()
        return catalog

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[CatalogEntry]:
        return iter(self.entries)

    def names(self) -> list[str]:
        return [e.name for e in self.entries]

    def get(self, name: str) -> CatalogEntry | None:
        for entry in self.entries:
            if entry.name == name:
                return entry
        return None

    def path_for(self, name: str) -> Path:
        entry = self.get(name)
        if entry is None:
            raise CatalogError(f"unknown bundle: {name!r}; have {self.names()}")
        return self.root / entry.path

    def store_for(self, name: str) -> BundleStore:
        return BundleStore(self.path_for(name))

    def register(self, *, name: str, path: str | None = None, created_at: str | None = None) -> CatalogEntry:
        if self.get(name) is not None:
            raise CatalogError(f"bundle already registered: {name!r}")
        entry = CatalogEntry(
            name=name,
            created_at=created_at or _utcnow_iso(),
            path=path or name,
        )
        self.entries.append(entry)
        self.save()
        return entry

    def deregister(self, name: str) -> CatalogEntry:
        entry = self.get(name)
        if entry is None:
            raise CatalogError(f"unknown bundle: {name!r}")
        self.entries = [e for e in self.entries if e.name != name]
        self.save()
        return entry

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / CATALOG_FILENAME
        data = {
            "schema_version": CATALOG_SCHEMA_VERSION,
            "bundles": [e.to_dict() for e in self.entries],
        }
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        return path
