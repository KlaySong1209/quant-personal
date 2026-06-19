"""Strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    @abstractmethod
    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

