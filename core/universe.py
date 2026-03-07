from dataclasses import dataclass
from pathlib import Path

import yaml

from .schemas import AssetMetadata


@dataclass(frozen=True)
class UniverseDefinition:
    name: str
    assets: tuple[AssetMetadata, ...]
    start_date: str = "2015-01-01"
    end_date: str | None = None

    def symbols(self):

        return [asset.symbol for asset in self.assets]

    def sector_map(self):

        return {asset.symbol: asset.sector for asset in self.assets}


class UniverseManager:

    def __init__(self, base_dir="."):

        self.base_dir = Path(base_dir)

    def from_yaml(self, path):

        universe_path = Path(path)

        if not universe_path.is_absolute():
            universe_path = (self.base_dir / universe_path).resolve()

        with open(universe_path, "r", encoding="utf-8") as handle:
            mapping = yaml.safe_load(handle) or {}

        return self.from_mapping(mapping)

    def from_mapping(self, mapping):

        universe_config = mapping.get("universe", mapping) or {}

        if universe_config.get("path"):
            return self.from_yaml(universe_config["path"])

        assets = universe_config.get("assets")

        if assets:
            asset_definitions = tuple(
                AssetMetadata(
                    symbol=asset["symbol"],
                    sector=asset.get("sector", "UNKNOWN"),
                    exchange=asset.get("exchange", "NSE"),
                    asset_class=asset.get("asset_class", "equity"),
                )
                for asset in assets
            )
            return UniverseDefinition(
                name=universe_config.get("name", "custom_universe"),
                assets=asset_definitions,
                start_date=universe_config.get("start_date", "2015-01-01"),
                end_date=universe_config.get("end_date"),
            )

        symbols = universe_config.get("symbols")
        if symbols:
            sector_map = universe_config.get("sectors", {})
            asset_definitions = tuple(
                AssetMetadata(symbol=symbol, sector=sector_map.get(symbol, "UNKNOWN"))
                for symbol in symbols
            )
            return UniverseDefinition(
                name=universe_config.get("name", "configured_universe"),
                assets=asset_definitions,
                start_date=universe_config.get("start_date", "2015-01-01"),
                end_date=universe_config.get("end_date"),
            )

        if mapping.get("symbol"):
            return UniverseDefinition(
                name="single_symbol",
                assets=(AssetMetadata(symbol=mapping["symbol"]),),
                start_date=mapping.get("data", {}).get("start_date", "2015-01-01"),
                end_date=mapping.get("data", {}).get("end_date"),
            )

        raise ValueError("Universe configuration must define `assets`, `symbols`, or `symbol`.")
