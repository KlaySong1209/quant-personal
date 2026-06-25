"""Daily OHLCV data quality checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

PRICE_COLUMNS = ("open", "high", "low", "close")


@dataclass(frozen=True)
class QualityIssue:
    code: str
    severity: str
    message: str
    symbol: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def errors(self) -> list[QualityIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[QualityIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "issues": [
                {
                    "code": issue.code,
                    "severity": issue.severity,
                    "message": issue.message,
                    "symbol": issue.symbol,
                    "detail": issue.detail,
                }
                for issue in self.issues
            ],
        }


def _normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        return df
    if "timestamp" in df.columns:
        return df.rename(columns={"timestamp": "date"})
    raise ValueError("daily frame must have a 'date' or 'timestamp' column")


def _to_calendar_index(calendar: Any) -> pd.DatetimeIndex:
    if calendar is None:
        return pd.DatetimeIndex([])
    if isinstance(calendar, pd.DatetimeIndex):
        idx = calendar
    elif hasattr(calendar, "sessions"):
        idx = pd.DatetimeIndex(calendar.sessions)
    elif isinstance(calendar, (list, tuple, np.ndarray, pd.Series)):
        idx = pd.DatetimeIndex(calendar)
    else:
        raise TypeError(f"unsupported calendar type: {type(calendar)!r}")
    return pd.DatetimeIndex(idx).tz_localize(None).normalize().drop_duplicates().sort_values()


def run_quality_checks(
    ohlcv: pd.DataFrame,
    *,
    symbols: list[str] | None = None,
    calendar: Any = None,
    max_abs_log_return: float = 0.22,
    stale_after_days: int | None = None,
    as_of: pd.Timestamp | None = None,
    production_data: bool = False,
) -> QualityReport:
    report = QualityReport()
    if ohlcv is None or ohlcv.empty:
        report.issues.append(QualityIssue("empty_data", "error", "no rows supplied to quality checks"))
        return report

    df = _normalize_date_column(ohlcv).copy()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
    df["symbol"] = df["symbol"].astype(str)

    for col in PRICE_COLUMNS:
        if col not in df.columns:
            report.issues.append(QualityIssue("missing_column", "error", f"required price column '{col}' is absent"))
    if not report.ok:
        return report

    present_symbols = set(df["symbol"].unique())
    if symbols is not None:
        for sym in symbols:
            if str(sym) not in present_symbols:
                report.issues.append(
                    QualityIssue(
                        "missing_symbol",
                        "error",
                        f"requested symbol '{sym}' is absent from the data",
                        symbol=str(sym),
                        detail={"symbol": str(sym)},
                    )
                )

    dup_mask = df.groupby("symbol")["date"].transform(lambda s: s.duplicated(keep=False))
    dups = df.loc[dup_mask, ["symbol", "date"]]
    for (sym, date), _grp in dups.groupby(["symbol", "date"]):
        report.issues.append(
            QualityIssue(
                "duplicate_date",
                "error",
                f"symbol '{sym}' has a duplicate date: {pd.Timestamp(date).date()}",
                symbol=str(sym),
                detail={"date": str(pd.Timestamp(date).date())},
            )
        )

    for col in PRICE_COLUMNS:
        bad = df[df[col] <= 0]
        for _, row in bad.iterrows():
            report.issues.append(
                QualityIssue(
                    "zero_or_negative_price",
                    "error",
                    f"symbol '{row['symbol']}' has {col}={row[col]} on {row['date'].date()}",
                    symbol=str(row["symbol"]),
                    detail={"column": col, "value": float(row[col]), "date": str(row["date"].date())},
                )
            )

    high_lt = df["high"] < df[["open", "close"]].max(axis=1)
    low_gt = df["low"] > df[["open", "close"]].min(axis=1)
    low_gt_high = df["low"] > df["high"]
    for _, row in df[high_lt | low_gt | low_gt_high].iterrows():
        report.issues.append(
            QualityIssue(
                "ohlc_inconsistent",
                "error",
                (
                    f"symbol '{row['symbol']}' OHLC inconsistent on {row['date'].date()}: "
                    f"open={row['open']} high={row['high']} low={row['low']} close={row['close']}"
                ),
                symbol=str(row["symbol"]),
                detail={
                    "date": str(row["date"].date()),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                },
            )
        )

    for sym, sub in df.sort_values(["symbol", "date"]).groupby("symbol"):
        closes = sub["close"].astype(float)
        valid = closes.gt(0) & np.isfinite(closes.to_numpy())
        log_ret = pd.Series(np.nan, index=closes.index)
        log_ret.loc[valid] = np.log(closes.loc[valid]).diff()
        flagged = log_ret[log_ret.abs() > max_abs_log_return]
        for idx, val in flagged.items():
            row = sub.loc[idx]
            report.issues.append(
                QualityIssue(
                    "abnormal_return",
                    "warning",
                    f"symbol '{sym}' abnormal return {float(val):+.4f} (log) on {row['date'].date()}: close={row['close']}",
                    symbol=str(sym),
                    detail={"date": str(row["date"].date()), "log_return": float(val), "close": float(row["close"])},
                )
            )

    if stale_after_days is not None:
        ref = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now("UTC").normalize()
        ref = ref.tz_localize(None) if ref.tzinfo is not None else ref
        ref = ref.normalize()
        for sym, sub in df.groupby("symbol"):
            latest = sub["date"].max()
            age_days = (ref - latest).days
            if age_days > stale_after_days:
                report.issues.append(
                    QualityIssue(
                        "stale_data",
                        "warning",
                        f"symbol '{sym}' latest data is {age_days} days old (as of {ref.date()})",
                        symbol=str(sym),
                        detail={"latest_date": str(latest.date()), "age_days": int(age_days), "as_of": str(ref.date())},
                    )
                )

    cal_idx = _to_calendar_index(calendar) if calendar is not None else pd.DatetimeIndex([])
    if production_data and cal_idx.empty:
        report.issues.append(
            QualityIssue(
                "missing_calendar",
                "error",
                "production_data requires a real trading calendar to detect gaps",
            )
        )
    if not cal_idx.empty:
        for sym, sub in df.groupby("symbol"):
            dates = pd.DatetimeIndex(sub["date"]).drop_duplicates().sort_values()
            expected = cal_idx[(cal_idx >= dates.min()) & (cal_idx <= dates.max())]
            missing = expected.difference(dates)
            if len(missing) > 0:
                report.issues.append(
                    QualityIssue(
                        "trading_day_gap",
                        "warning",
                        f"symbol '{sym}' is missing {len(missing)} trading session(s)",
                        symbol=str(sym),
                        detail={"missing_dates": [str(d.date()) for d in missing[:20]], "count": int(len(missing))},
                    )
                )

    return report
