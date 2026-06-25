"""Thin wrapper around ``mootdx.quotes.Quotes`` with explicit fallbacks.

Why this exists (see simonlin1212/a-stock-data V3.2.4 #26):
- ``Quotes.factory()`` is documented to "select the fastest server", but on
  some fresh installs the empty ``BESTIP.HQ`` value crashes the call.
  Existing users whose ``~/.mootdx/config.json`` already has IPs never hit it.
- The fix is **not** to pin ``mootdx<0.11`` — that crashes on import under
  newer numpy/pandas. The fix is a three-level fallback in user-land code:
    1. bare ``factory()`` (lets mootdx do its BESTIP speed test)
    2. explicit ``server=(ip, port)`` from a baked-in working list
    3. raise a clear error pointing the user at network / IP geolocation

We also centralise warning suppression and logger level so callers don't have
to repeat the boilerplate every fetch.
"""

from __future__ import annotations

import logging
import socket
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

# A handful of mootdx-bundled servers known to respond at the time this list
# was authored. mootdx itself ships a longer list and refreshes it via its
# config file; this is only the manual fallback when BESTIP fails or is empty.
# Format: (host, port). All TDX std-market servers listen on TCP/7709.
_FALLBACK_SERVERS: tuple[tuple[str, int], ...] = (
    ("119.147.212.81", 7709),   # 招商证券深圳行情
    ("47.103.86.229", 7709),    # 阿里云华东1
    ("106.14.95.149", 7709),    # 阿里云华东2
    ("123.125.108.14", 7709),   # 北京
    ("110.40.193.81", 7709),    # 腾讯云
)


class TdxClientError(Exception):
    """Raised when no mootdx route is reachable."""


@dataclass(frozen=True)
class TdxRoute:
    """How a working ``Quotes`` instance was obtained — for provenance logging."""

    method: str            # "bestip" | "explicit" | "cached"
    server: str | None     # "host:port" when method=="explicit", else None


@contextmanager
def _silence_mootdx() -> Iterator[None]:
    """Mute mootdx's noisy WARNING logs + DeprecationWarning chatter."""
    prev_level = logging.getLogger("mootdx").level
    logging.getLogger("mootdx").setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            yield
        finally:
            logging.getLogger("mootdx").setLevel(prev_level)


def _probe_tcp(host: str, port: int, timeout_s: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def open_client(*, probe_timeout_s: float = 1.5):
    """Return a working mootdx ``Quotes`` client and the route taken.

    Three-level fallback:
      1. ``Quotes.factory(market='std')`` — lets mootdx run BESTIP speed test.
      2. Explicit ``server=(ip,port)`` from a baked-in TCP-probed shortlist.
      3. Raise :class:`TdxClientError` listing what was tried.

    Returns ``(client, route)``. The route is informational only — it's logged
    into provenance so a future "why didn't fetch work yesterday" answer is
    on disk.
    """
    # Import inside the function so unit tests can mock mootdx without
    # paying the import cost on every import of this module.
    from mootdx.quotes import Quotes

    # Level 1: bare factory (the official happy path).
    with _silence_mootdx():
        try:
            client = Quotes.factory(market="std")
            return client, TdxRoute(method="bestip", server=None)
        except (ValueError, IndexError, KeyError) as exc:
            level_1_error = exc
        except Exception as exc:  # noqa: BLE001 — mootdx raises many shapes
            level_1_error = exc

    # Level 2: explicit server, TCP-probed first.
    errors: list[str] = [f"bestip: {type(level_1_error).__name__}: {level_1_error}"]
    for host, port in _FALLBACK_SERVERS:
        if not _probe_tcp(host, port, timeout_s=probe_timeout_s):
            errors.append(f"{host}:{port}: tcp unreachable")
            continue
        with _silence_mootdx():
            try:
                client = Quotes.factory(market="std", server=(host, port))
                return client, TdxRoute(method="explicit", server=f"{host}:{port}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{host}:{port}: {type(exc).__name__}: {exc}")

    # Level 3: nothing worked.
    raise TdxClientError(
        "could not open a mootdx (TDX) client; attempts:\n  - "
        + "\n  - ".join(errors)
        + "\nLikely causes: non-CN IP (mootdx requires mainland routing), "
        "firewall blocking TCP/7709, or all listed servers offline."
    )
