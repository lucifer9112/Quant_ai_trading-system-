"""Feature selection, pruning, and explainability helpers."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler

from explainability.shap_explainer import ShapExplainer

logger = logging.getLogger(__name__)


class FeatureAnalyzer:
    """Analyze feature quality, prune redundancy, and compute importance."""

    def __init__(
        self,
        correlation_threshold: float = 0.95,
        variance_threshold: float = 0.01,
        method: str = "combined",
    ):
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.method = method
        self.shap_explainer = ShapExplainer()

    def analyze_features(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> Dict:
        results = {
            "total_features": X.shape[1],
            "low_variance": self._find_low_variance_features(X),
            "high_correlation": self._find_correlated_features(X),
            "missing_values": self._analyze_missing_values(X),
            "outliers": self._detect_outliers(X),
        }

        if y is not None:
            results["feature_importance"] = self._compute_feature_importance(X, y)
            results["feature_stability"] = self._compute_feature_stability(X, y)

        return results

    def prune_correlated_features(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        *,
        max_correlation: Optional[float] = None,
        protected_features: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        threshold = max_correlation or self.correlation_threshold
        protected = set(protected_features or [])
        numeric = X.select_dtypes(include=[np.number]).copy()

        if numeric.empty:
            return {
                "selected_features": list(X.columns),
                "removed_features": [],
                "correlation_pairs": [],
                "pruned_frame": X.copy(),
            }

        importance_rank = self._importance_rank(numeric, y)
        corr_matrix = numeric.corr().abs()
        removed = set()
        decisions = []

        for i, feature in enumerate(corr_matrix.columns):
            if feature in removed:
                continue
            for peer in corr_matrix.columns[i + 1:]:
                if peer in removed:
                    continue
                correlation = corr_matrix.loc[feature, peer]
                if pd.isna(correlation) or correlation <= threshold:
                    continue

                keep = self._pick_feature_to_keep(
                    feature,
                    peer,
                    importance_rank=importance_rank,
                    protected=protected,
                )
                drop = peer if keep == feature else feature
                removed.add(drop)
                decisions.append(
                    {
                        "keep": keep,
                        "drop": drop,
                        "correlation": float(correlation),
                    }
                )

        selected = [column for column in X.columns if column not in removed]
        logger.info("Pruned %d correlated features; kept %d", len(removed), len(selected))
        return {
            "selected_features": selected,
            "removed_features": sorted(removed),
            "correlation_pairs": decisions,
            "pruned_frame": X[selected].copy(),
        }

    def explain_with_shap(self, model, X: pd.DataFrame):
        return self.shap_explainer.explain(model, X)

    def compute_shap_importance(self, model, X: pd.DataFrame) -> pd.DataFrame:
        return self.explain_with_shap(model, X).feature_importance

    def select_features(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        n_features: Optional[int] = None,
        max_correlation: float = 0.95,
    ) -> List[str]:
        if n_features is None:
            n_features = min(50, max(1, X.shape[1] // 2))

        selected = [col for col in X.columns if col not in self._find_low_variance_features(X)]
        working = X[selected]
        prune_report = self.prune_correlated_features(
            working,
            y=y,
            max_correlation=max_correlation,
        )
        selected = prune_report["selected_features"]
        working = X[selected]

        if y is not None and len(selected) > n_features:
            importance = self._compute_feature_importance(working, y)
            selected = importance.head(n_features).index.tolist()
        else:
            selected = selected[:n_features]

        logger.info("Selected %d features from %d total", len(selected), X.shape[1])
        return selected

    def _find_low_variance_features(self, X: pd.DataFrame) -> List[str]:
        numeric = X.select_dtypes(include=[np.number])
        if numeric.empty:
            return []

        filled = numeric.fillna(numeric.mean(numeric_only=True)).fillna(0.0)
        std = filled.std().replace(0, np.nan)
        standardized = (filled - filled.mean()) / std
        variances = standardized.var().fillna(0.0)
        low_var_features = variances[variances <= self.variance_threshold].index.tolist()
        logger.info("Found %d low-variance features", len(low_var_features))
        return low_var_features

    def _find_correlated_features(self, X: pd.DataFrame) -> Dict[str, List[str]]:
        numeric = X.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
        if numeric.empty:
            return {}

        correlation_matrix = numeric.corr().abs()
        upper = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )

        correlated_pairs = {}
        for column in upper.columns:
            correlated = upper[column][upper[column] > self.correlation_threshold]
            if len(correlated) > 0:
                correlated_pairs[column] = correlated.index.tolist()

        logger.info(
            "Found %d feature groups with correlation > %.2f",
            len(correlated_pairs),
            self.correlation_threshold,
        )
        return correlated_pairs

    def _analyze_missing_values(self, X: pd.DataFrame) -> Dict[str, float]:
        missing_pct = (X.isnull().sum() / len(X)) * 100
        return missing_pct[missing_pct > 0].to_dict()

    def _detect_outliers(self, X: pd.DataFrame) -> Dict[str, int]:
        outliers_per_feature = {}
        for col in X.select_dtypes(include=[np.number]).columns:
            std = X[col].std()
            if std == 0 or pd.isna(std):
                continue
            z_scores = np.abs((X[col] - X[col].mean()) / std)
            outlier_count = (z_scores > 3).sum()
            if outlier_count > 0:
                outliers_per_feature[col] = int(outlier_count)
        logger.info("Detected outliers in %d features", len(outliers_per_feature))
        return outliers_per_feature

    def _compute_feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        X_clean = X.fillna(X.mean(numeric_only=True)).fillna(0.0)
        X_scaled = StandardScaler().fit_transform(X_clean)

        try:
            f_scores = SelectKBest(score_func=f_classif, k="all").fit(X_scaled, y)
            f_importance = pd.Series(f_scores.scores_, index=X.columns)
        except Exception:
            f_importance = pd.Series(0.0, index=X.columns)
            logger.warning("Could not compute F-scores")

        try:
            mi_scores = SelectKBest(score_func=mutual_info_classif, k="all").fit(X_clean, y)
            mi_importance = pd.Series(mi_scores.scores_, index=X.columns)
        except Exception:
            mi_importance = pd.Series(0.0, index=X.columns)
            logger.warning("Could not compute mutual information scores")

        importance_df = pd.DataFrame(
            {
                "f_score": f_importance,
                "mutual_information": mi_importance,
            }
        )
        importance_df["combined_score"] = (
            importance_df["f_score"].rank(ascending=False) +
            importance_df["mutual_information"].rank(ascending=False)
        ) / 2
        importance_df = importance_df.sort_values("combined_score")
        logger.info("Top 5 features: %s", importance_df.head().index.tolist())
        return importance_df

    def _compute_feature_stability(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int = 5,
    ) -> Dict[str, float]:
        fold_size = max(1, len(X) // n_folds)
        stability_scores = {}

        for col in X.columns:
            if pd.isna(X[col]).all():
                continue

            fold_scores = []
            for i in range(n_folds - 1):
                start = i * fold_size
                end = min((i + 1) * fold_size, len(X))
                fold_data = X.iloc[start:end]
                fold_target = y.iloc[start:end]

                if len(fold_data) > 1 and fold_data[col].std() > 0:
                    corr = fold_data[col].corr(fold_target)
                    if not pd.isna(corr):
                        fold_scores.append(abs(corr))

            if fold_scores:
                stability_scores[col] = float(np.std(fold_scores))

        return stability_scores

    def _importance_rank(self, X: pd.DataFrame, y: Optional[pd.Series]) -> Dict[str, int]:
        if y is None:
            return {column: idx for idx, column in enumerate(X.columns)}
        importance = self._compute_feature_importance(X, y)
        return {column: idx for idx, column in enumerate(importance.index.tolist())}

    def _pick_feature_to_keep(
        self,
        left: str,
        right: str,
        *,
        importance_rank: Dict[str, int],
        protected: set[str],
    ) -> str:
        if left in protected and right not in protected:
            return left
        if right in protected and left not in protected:
            return right

        left_rank = importance_rank.get(left, np.inf)
        right_rank = importance_rank.get(right, np.inf)
        if left_rank <= right_rank:
            return left
        return right
