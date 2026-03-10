from main import build_cli_parser


def test_cli_defaults_to_legacy_mode():

    args = build_cli_parser().parse_args([])

    assert args.mode == "legacy"
    assert args.data is None


def test_cli_accepts_complete_mode_with_data_argument():

    args = build_cli_parser().parse_args(["--mode", "complete", "--data", "market.csv"])

    assert args.mode == "complete"
    assert args.data == "market.csv"
