"""Broker adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BrokerAdapter(ABC):
    @abstractmethod
    def submit_target(self, timestamp: pd.Timestamp, target_shares: dict[str, float]) -> None:
        raise NotImplementedError

    @abstractmethod
    def mark_to_market(self, timestamp: pd.Timestamp) -> float:
        raise NotImplementedError

