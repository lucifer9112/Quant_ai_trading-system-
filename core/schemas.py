from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AssetMetadata:
    symbol: str
    sector: str = "UNKNOWN"
    exchange: str = "NSE"
    asset_class: str = "equity"


@dataclass
class FeatureRow:
    date: Any
    symbol: str
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeSignal:
    date: Any
    symbol: str
    signal: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    weight: float = 0.0


@dataclass
class PortfolioSnapshot:
    date: Any
    equity: float
    cash: float
    drawdown: float
    gross_exposure: float
