"""Freshness judgement: comparing a bundle's actual coverage against what we'd
*expect* given a reference date and a trading calendar.

This is deliberately permissive about the calendar — the early bundles will
have a synthetic business-day calendar (since fetcher integration is later
phases). When that's the case we fall back to ``pandas.bdate_range`` for the
expected-through date.

A bundle is:
  - ``fresh``   — actual_through >= expected_through
  - ``stale``   — actual_through  < expected_through
  - ``no_data`` — bundle has zero rows (or date_range absent)
"""

from __future__ import annotations

from datetime import date
from typing import Literal

import pandas as pd

from quant.data.adjust.calendar import TradingCalendar
from quant.data.bundle.manifest import BundleManifest, FreshnessMeta

FreshnessStatus = Literal["fresh", "stale", "no_data"]


def expected_through(
    *,
    as_of: pd.Timestamp | None = None,
    calendar: TradingCalendar | None = None,
) -> pd.Timestamp:
    """Return the latest trading day <= as_of.

    If *calendar* is None, falls back to a business-day approximation (same
    fallback the rest of the project uses; see :class:`TradingCalendar.synthetic`).
    """
    if as_of is None:
        as_of = pd.Timestamp.now(tz="UTC")
    else:
        as_of = pd.Timestamp(as_of)
        if as_of.tzinfo is None:
            as_of = as_of.tz_localize("UTC")
        else:
            as_of = as_of.tz_convert("UTC")
    as_of_date = as_of.normalize()

    if calendar is not None:
        sessions = pd.DatetimeIndex(calendar.sessions).normalize()
        on_or_before = sessions[sessions <= as_of_date]
        if len(on_or_before) == 0:
            # Calendar starts after as_of — degenerate but possible during testing.
            return sessions[0]
        return on_or_before[-1]

    # Synthetic business-day fallback (consistent with TradingCalendar.synthetic).
    bdays = pd.bdate_range(end=as_of_date, periods=1, tz="UTC")
    return bdays[-1]


def judge(
    manifest: BundleManifest,
    *,
    as_of: pd.Timestamp | None = None,
    calendar: TradingCalendar | None = None,
) -> FreshnessMeta:
    """Compute a :class:`FreshnessMeta` for *manifest* relative to *as_of*.

    The returned meta can replace ``manifest.freshness`` via ``model_copy``.
    """
    if manifest.row_count == 0:
        # Take last-known dates as both sides so the meta is still well-formed.
        last = manifest.date_range.last
        return FreshnessMeta(
            expected_through=last,
            actual_through=last,
            status="no_data",
        )

    expected_ts = expected_through(as_of=as_of, calendar=calendar)
    actual_ts = pd.Timestamp(manifest.date_range.last)
    if actual_ts.tzinfo is None:
        actual_ts = actual_ts.tz_localize("UTC")

    status: FreshnessStatus = "fresh" if actual_ts >= expected_ts else "stale"
    return FreshnessMeta(
        expected_through=expected_ts.date().isoformat(),
        actual_through=actual_ts.date().isoformat(),
        status=status,
    )
