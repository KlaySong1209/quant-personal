"""Pydantic schema for YAML configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RunConfig(_Frozen):
    name: str
    seed: int = 42


class SyntheticParams(_Frozen):
    initial_price: float = Field(gt=0)
    annual_drift: float = 0.05
    annual_vol: float = Field(ge=0, default=0.20)


class DataValidationConfig(_Frozen):
    max_missing_ratio: float = Field(ge=0, le=1, default=0.01)


class CorporateActionsConfig(_Frozen):
    enabled: bool = False
    path: str | None = None
    column_map: dict[str, str] = Field(default_factory=dict)
    max_unadjusted_log_return: float = Field(gt=0, default=0.25)
    adjustment_convention: Literal["forward", "backward", "none"] | None = None
    has_adjustment_factor: bool | None = None
    dividend_tax_treatment: Literal["pre_tax", "post_tax"] | None = None


class UniverseConfig(_Frozen):
    source: Literal["all", "file", "synthetic_delisting"] = "all"
    path: str | None = None
    column_map: dict[str, str] = Field(default_factory=dict)
    delisted_symbol: str | None = None
    delist_date: str | None = None


class CalendarConfig(_Frozen):
    source: Literal["synthetic", "file"] = "synthetic"
    mode: Literal["synthetic", "file"] = "synthetic"
    exchange: str = "SYNTH"
    path: str | None = None
    file: str | None = None
    column_map: dict[str, str] = Field(default_factory=dict)
    column_mapping: dict[str, str] = Field(default_factory=dict)
    date_format: str | None = None
    timezone: str = "UTC"


class DataConfig(_Frozen):
    source: Literal["synthetic", "csv", "local_file"]
    production_data: bool = False
    csv_path: str = "data/example"
    local_path: str | None = None
    column_mapping_path: str | None = None
    processed_output_dir: str = "data/processed"
    symbols: list[str]
    start: str
    end: str
    synthetic: SyntheticParams
    validation: DataValidationConfig
    corporate_actions: CorporateActionsConfig = Field(default_factory=CorporateActionsConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    calendar: CalendarConfig = Field(default_factory=CalendarConfig)

    @field_validator("symbols")
    @classmethod
    def _nonempty_symbols(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("symbols must be a non-empty list")
        if len(set(v)) != len(v):
            raise ValueError("symbols must be unique")
        return v

    @model_validator(mode="after")
    def _production_data_requires_real_inputs(self) -> "DataConfig":
        if self.production_data and self.source == "synthetic":
            raise ValueError("production_data=true forbids synthetic data source")
        cal_mode = self.calendar.mode or self.calendar.source
        if self.production_data and cal_mode == "synthetic":
            raise ValueError("production_data=true forbids synthetic calendar")
        return self


class FuturesSyntheticParams(_Frozen):
    n_contracts: int = Field(ge=2, default=4)
    days_per_contract: int = Field(ge=10, default=60)
    overlap_days: int = Field(ge=1, default=10)
    initial_underlying_price: float = Field(gt=0, default=500.0)
    annual_drift: float = 0.02
    annual_vol: float = Field(ge=0, default=0.15)
    basis_per_contract: float = Field(ge=0, default=2.0)


class RollConfig(_Frozen):
    method: Literal["back_adjusted"] = "back_adjusted"
    rule: Literal["calendar", "volume_crossover"] = "calendar"
    calendar_days_before_expiry: int = Field(ge=0, default=5)
    max_implausible_log_return: float = Field(gt=0, default=0.10)


class FuturesDataConfig(_Frozen):
    source: Literal["synthetic_futures", "local_file_futures"]
    symbol: str
    continuous_symbol: str
    start: str
    path: str | None = None
    column_map: dict[str, str] = Field(default_factory=dict)
    synthetic: FuturesSyntheticParams
    roll: RollConfig
    validation: DataValidationConfig


class StrategyConfig(_Frozen):
    name: Literal["placeholder", "ts_trend"]
    params: dict[str, float | int | str]


class PortfolioConfig(_Frozen):
    initial_equity: float = Field(gt=0)
    rebalance: Literal["daily"] = "daily"


class CostsConfig(_Frozen):
    bps: float = Field(ge=0)
    slippage_bps: float = Field(ge=0)
    spread_bps: float = Field(ge=0, default=0.0)
    allow_zero_cost_for_tests: bool = False


class FuturesCostsConfig(_Frozen):
    commission_bps: float = Field(ge=0, default=0.5)
    slippage_bps: float = Field(ge=0, default=1.0)
    spread_bps: float = Field(ge=0, default=1.0)
    allow_zero_cost_for_tests: bool = False


class RiskConfigModel(_Frozen):
    max_symbol_weight: float = Field(gt=0, le=1)
    max_gross_leverage: float = Field(gt=0)
    reject_nan: bool = True


class FuturesRiskCfgModel(_Frozen):
    max_symbol_notional: float = Field(gt=0)
    max_gross_notional: float = Field(gt=0)
    max_margin_use: float = Field(gt=0, le=1.0)
    reject_nan: bool = True


class PaperBrokerConfig(_Frozen):
    starting_cash: float = Field(gt=0)
    allow_short: bool = False
    allow_margin: bool = False
    max_gross_leverage: float = Field(gt=0, default=1.0)


class ExecutionConfig(_Frozen):
    broker: Literal["paper"]
    paper: PaperBrokerConfig


class AppConfig(_Frozen):
    run: RunConfig
    data: DataConfig
    strategy: StrategyConfig
    portfolio: PortfolioConfig
    costs: CostsConfig
    risk: RiskConfigModel
    execution: ExecutionConfig


class InstrumentsConfig(_Frozen):
    specs: list[str]


class FuturesAppConfig(_Frozen):
    mode: Literal["futures"]
    run: RunConfig
    data: FuturesDataConfig
    instruments: InstrumentsConfig
    strategy: StrategyConfig
    portfolio: PortfolioConfig
    costs: FuturesCostsConfig
    risk: FuturesRiskCfgModel
