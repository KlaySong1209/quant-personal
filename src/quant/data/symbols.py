"""Canonical symbol space for the A-share market.

A canonical symbol is ``<EXCHANGE><RAW_CODE>``, e.g. ``SH600519`` / ``SZ000001``.
All bundle artifacts store symbols in canonical form; per-fetcher transforms
convert at the boundary (mootdx / tencent / eastmoney / etc.).

A small ``SYNTH`` exchange is supported so existing placeholder-symbol flows
(``AAA``/``BBB``/``CCC``) keep working without retrofitting the test suite.
Real A-share rules (6→SH, 0/3→SZ) are applied only when the input matches
a 6-digit numeric A-share code without a prefix.

Northbound/BJ exchange is intentionally out of scope (the project targets
3-5 SH/SZ tickers per [docs/architecture.md](docs/architecture.md)).
"""

from __future__ import annotations

import re
from enum import Enum

A_SHARE_RAW_RE = re.compile(r"^\d{6}$")
CANONICAL_RE = re.compile(r"^(SH|SZ|SYNTH)([A-Z0-9]+)$")


class Exchange(str, Enum):
    SSE = "SH"        # Shanghai Stock Exchange (600/601/603/688/605)
    SZSE = "SZ"       # Shenzhen Stock Exchange (000/001/002/003/300/301)
    SYNTH = "SYNTH"   # Synthetic / placeholder symbols (AAA, BBB, CCC, ...)


class SymbolError(ValueError):
    """Raised when a symbol cannot be parsed or transformed."""


def parse_symbol(s: str) -> tuple[Exchange, str]:
    """Return ``(exchange, raw_code)`` for *s*.

    Accepted forms:
      - ``SH600519`` / ``SZ000001`` — canonical, returned as-is
      - ``SYNTHAAA``                — canonical synthetic
      - ``600519``                  — bare A-share digits, exchange inferred
      - ``AAA`` / ``BBB``           — bare alpha-only, treated as SYNTH
      - ``sh600519`` / ``sz000001`` — case-insensitive canonical

    Raises :class:`SymbolError` on anything else (mixed prefixes, BJ codes, …).
    """
    if not isinstance(s, str) or not s:
        raise SymbolError(f"symbol must be a non-empty string: {s!r}")
    upper = s.strip().upper()

    m = CANONICAL_RE.match(upper)
    if m:
        prefix, raw = m.group(1), m.group(2)
        if prefix == "SH":
            return Exchange.SSE, raw
        if prefix == "SZ":
            return Exchange.SZSE, raw
        return Exchange.SYNTH, raw

    if A_SHARE_RAW_RE.match(upper):
        first = upper[0]
        if first == "6":
            return Exchange.SSE, upper
        if first in {"0", "3"}:
            return Exchange.SZSE, upper
        # 4/8/9 etc. belong to BJ or are otherwise out of scope.
        raise SymbolError(
            f"cannot infer exchange for A-share code {upper!r}; "
            f"only 6xxxxx (SH) and 0xxxxx/3xxxxx (SZ) are supported. "
            f"Use an explicit prefix like 'SH{upper}' or 'SZ{upper}' if needed."
        )

    if upper.isalpha():
        return Exchange.SYNTH, upper

    raise SymbolError(
        f"unrecognized symbol {s!r}; expected canonical (SH600519), "
        f"6-digit A-share code (600519), or alpha synthetic (AAA)"
    )


def normalize(s: str) -> str:
    """Return the canonical form of *s* (``SH600519``, ``SZ000001``, ``SYNTHAAA``)."""
    exchange, raw = parse_symbol(s)
    return f"{exchange.value}{raw}"


def normalize_many(symbols: list[str]) -> list[str]:
    """Normalize a list, preserving order, rejecting duplicates after normalize."""
    out: list[str] = []
    seen: set[str] = set()
    for s in symbols:
        canonical = normalize(s)
        if canonical in seen:
            raise SymbolError(f"duplicate symbol after normalization: {canonical}")
        seen.add(canonical)
        out.append(canonical)
    return out


def is_canonical(s: str) -> bool:
    """True iff *s* is already a canonical ``SH``/``SZ``/``SYNTH`` symbol."""
    return bool(CANONICAL_RE.match(s)) if isinstance(s, str) else False


def to_mootdx(s: str) -> tuple[int, str]:
    """Convert canonical to mootdx ``(market, code)`` tuple.

    mootdx market codes: 0=SZSE, 1=SSE. SYNTH is not supported.
    """
    exchange, raw = parse_symbol(s)
    if exchange == Exchange.SSE:
        return 1, raw
    if exchange == Exchange.SZSE:
        return 0, raw
    raise SymbolError(f"mootdx does not support {exchange.value} symbol {s!r}")


def to_tencent(s: str) -> str:
    """Convert canonical to tencent api code: ``sh600519`` / ``sz000001``."""
    exchange, raw = parse_symbol(s)
    if exchange == Exchange.SSE:
        return f"sh{raw}"
    if exchange == Exchange.SZSE:
        return f"sz{raw}"
    raise SymbolError(f"tencent does not support {exchange.value} symbol {s!r}")


def to_eastmoney(s: str) -> str:
    """Convert canonical to eastmoney api code: ``1.600519`` (SH) / ``0.000001`` (SZ)."""
    exchange, raw = parse_symbol(s)
    if exchange == Exchange.SSE:
        return f"1.{raw}"
    if exchange == Exchange.SZSE:
        return f"0.{raw}"
    raise SymbolError(f"eastmoney does not support {exchange.value} symbol {s!r}")
