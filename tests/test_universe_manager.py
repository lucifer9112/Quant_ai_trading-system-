from core.universe import UniverseManager


def test_universe_manager_builds_from_symbols_and_sectors():

    manager = UniverseManager()
    universe = manager.from_mapping({
        "universe": {
            "name": "demo",
            "symbols": ["RELIANCE", "TCS"],
            "sectors": {"RELIANCE": "Energy", "TCS": "Information Technology"},
            "start_date": "2020-01-01",
        }
    })

    assert universe.name == "demo"
    assert universe.symbols() == ["RELIANCE", "TCS"]
    assert universe.sector_map()["TCS"] == "Information Technology"


def test_universe_manager_reads_yaml_definition(tmp_path):

    universe_file = tmp_path / "universe.yaml"
    universe_file.write_text(
        "name: custom\n"
        "assets:\n"
        "  - symbol: INFY\n"
        "    sector: Information Technology\n",
        encoding="utf-8",
    )

    manager = UniverseManager(base_dir=tmp_path)
    universe = manager.from_yaml(universe_file)

    assert universe.name == "custom"
    assert universe.symbols() == ["INFY"]
