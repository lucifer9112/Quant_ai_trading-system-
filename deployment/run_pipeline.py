from data_pipeline.market_data.nse_downloader import NSEDownloader
from feature_engineering.feature_pipeline import FeaturePipeline
from strategy_engine.strategy_scoring import StrategyScoring
from decision_engine.signal_generator import SignalGenerator


class TradingPipeline:

    def run(self):

        downloader = NSEDownloader("RELIANCE")

        df = downloader.download()

        df = FeaturePipeline().run(df)

        df = StrategyScoring().compute_score(df)

        df = SignalGenerator().generate(df)

        return df


if __name__ == "__main__":

    pipeline = TradingPipeline()

    data = pipeline.run()

    print(data.tail())