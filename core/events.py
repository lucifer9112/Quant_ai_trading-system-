from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MarketEvent:
    date: Any
    symbol: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SignalEvent:
    date: Any
    symbol: str
    signal: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderEvent:
    date: Any
    symbol: str
    target_weight: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FillEvent:
    date: Any
    symbol: str
    quantity: float
    price: float
    transaction_cost: float
