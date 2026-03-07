from features.market_microstructure.advanced_features import MarketMicrostructureFeatures
from features.regime.regime_aware_features import RegimeAwareFeatures
from features.registry import FeatureRegistry
from features.technical.advanced_features import AdvancedTechnicalFeatures


class ResearchFeaturePipeline:

    def __init__(self, base_pipeline=None, registry=None):

        if base_pipeline is None:
            from feature_engineering.feature_pipeline import FeaturePipeline

            base_pipeline = FeaturePipeline()

        self.base_pipeline = base_pipeline
        self.registry = registry or self._build_registry()

    def _build_registry(self):

        registry = FeatureRegistry()
        registry.register("advanced_technical", AdvancedTechnicalFeatures().add)
        registry.register("market_microstructure", MarketMicrostructureFeatures().add)
        registry.register("regime_aware", RegimeAwareFeatures().add)

        return registry

    def run(self, df):

        df = self.base_pipeline.run(df.copy())
        df = self.registry.apply(df)

        return df
