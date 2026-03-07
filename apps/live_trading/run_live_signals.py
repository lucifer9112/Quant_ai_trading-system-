import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.live_trading.live_signal_engine import LiveSignalEngine
from data_pipeline.market_data.realtime_stream import RealtimeStreamer


def parse_args():

    parser = argparse.ArgumentParser(description="Generate near real-time trading signals from streaming bars.")
    parser.add_argument("--symbol", default="RELIANCE", help="Ticker symbol without .NS suffix.")
    parser.add_argument("--interval", type=int, default=10, help="Polling interval in seconds.")
    parser.add_argument("--limit", type=int, default=5, help="Maximum bars to consume before exit.")

    return parser.parse_args()


def main():

    args = parse_args()
    streamer = RealtimeStreamer(args.symbol)
    engine = LiveSignalEngine()

    for bar in streamer.stream_quotes(interval=args.interval, limit=args.limit):
        signal = engine.on_bar(bar)

        if signal is None:
            continue

        print(signal)


if __name__ == "__main__":
    main()
