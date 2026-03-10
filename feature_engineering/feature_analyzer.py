"""
Feature Analyzer - Feature Selection & Quality Assessment

Identify:
- Correlated features (remove redundancy)
- Important features (high predictive power)
- Unstable features (don't generalize)
- Near-zero variance features (useless)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import logging

logger = logging.getLogger(__name__)


class FeatureAnalyzer:
    """Analyze feature quality and select best features."""
    
    def __init__(self, correlation_threshold: float = 0.95, variance_threshold: float = 0.01):
        """
        Args:
            correlation_threshold: Remove features correlated > threshold
            variance_threshold: Remove features with variance < threshold
        """
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
    
    def analyze_features(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> Dict:
        """
        Comprehensive feature analysis.
        
        Args:
            X: Feature dataframe
            y: Optional target variable for supervised analysis
            
        Returns:
            Dictionary with analysis results
        """
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
    
    def _find_low_variance_features(self, X: pd.DataFrame) -> List[str]:
        """Find near-zero variance features (useless)."""
        
        variances = X.var()
        max_var = variances.max()
        threshold = max_var * self.variance_threshold
        
        low_var_features = variances[variances < threshold].index.tolist()
        
        logger.info(f"Found {len(low_var_features)} low-variance features")
        return low_var_features
    
    def _find_correlated_features(self, X: pd.DataFrame) -> Dict[str, List[str]]:
        """Find highly correlated feature pairs."""
        
        # Drop NaN columns
        X_clean = X.dropna(axis=1, how='all')
        
        # Compute correlation matrix
        correlation_matrix = X_clean.corr().abs()
        
        # Find upper triangle (avoid duplicates)
        upper = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        # Find correlated pairs
        correlated_pairs = {}
        for column in upper.columns:
            correlated = upper[column][upper[column] > self.correlation_threshold]
            if len(correlated) > 0:
                correlated_pairs[column] = correlated.index.tolist()
        
        logger.info(f"Found {len(correlated_pairs)} feature pairs with correlation > {self.correlation_threshold}")
        return correlated_pairs
    
    def _analyze_missing_values(self, X: pd.DataFrame) -> Dict[str, float]:
        """Analyze missing values per feature."""
        
        missing_pct = (X.isnull().sum() / len(X)) * 100
        missing_features = missing_pct[missing_pct > 0].to_dict()
        
        if missing_features:
            logger.info(f"Found missing values in {len(missing_features)} features")
        
        return missing_features
    
    def _detect_outliers(self, X: pd.DataFrame) -> Dict[str, int]:
        """Detect outliers using z-score."""
        
        outliers_per_feature = {}
        
        for col in X.select_dtypes(include=[np.number]).columns:
            if X[col].std() == 0:
                continue
            
            z_scores = np.abs((X[col] - X[col].mean()) / X[col].std())
            outlier_count = (z_scores > 3).sum()
            
            if outlier_count > 0:
                outliers_per_feature[col] = int(outlier_count)
        
        logger.info(f"Detected outliers in {len(outliers_per_feature)} features")
        return outliers_per_feature
    
    def _compute_feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        """Compute feature importance using mutual information and F-score."""
        
        X_clean = X.fillna(X.mean())
        X_scaled = StandardScaler().fit_transform(X_clean)
        
        # F-score (ANOVA)
        try:
            f_scores = SelectKBest(score_func=f_classif, k='all').fit(X_scaled, y)
            f_importance = pd.Series(f_scores.scores_, index=X.columns).sort_values(ascending=False)
        except:
            f_importance = pd.Series(0, index=X.columns)
            logger.warning("Could not compute F-scores")
        
        # Mutual information
        try:
            mi_scores = SelectKBest(score_func=mutual_info_classif, k='all').fit(X_clean, y)
            mi_importance = pd.Series(mi_scores.scores_, index=X.columns).sort_values(ascending=False)
        except:
            mi_importance = pd.Series(0, index=X.columns)
            logger.warning("Could not compute mutual information scores")
        
        # Combine
        importance_df = pd.DataFrame({
            "f_score": f_importance,
            "mutual_information": mi_importance,
        })
        
        # Aggregate score (average rank)
        importance_df["combined_score"] = (
            importance_df["f_score"].rank(ascending=False) +
            importance_df["mutual_information"].rank(ascending=False)
        ) / 2
        
        importance_df = importance_df.sort_values("combined_score")
        
        logger.info(f"Top 5 features: {importance_df.head().index.tolist()}")
        
        return importance_df
    
    def _compute_feature_stability(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int = 5,
    ) -> Dict[str, float]:
        """Measure feature stability across time periods."""
        
        # Split data into folds by time
        fold_size = len(X) // n_folds
        stability_scores = {}
        
        for col in X.columns:
            if pd.isna(X[col]).all():
                continue
            
            fold_scores = []
            for i in range(n_folds - 1):
                start = i * fold_size
                end = (i + 1) * fold_size
                next_end = (i + 2) * fold_size
                
                # Compute correlation between feature and target in fold i
                fold_data = X.iloc[start:end]
                fold_target = y.iloc[start:end]
                
                if len(fold_data) > 1 and fold_data[col].std() > 0:
                    corr = fold_data[col].corr(fold_target)
                    fold_scores.append(abs(corr))
            
            if fold_scores:
                # Stability = consistency of importance across folds
                stability_scores[col] = np.std(fold_scores)  # Lower = more stable
        
        return stability_scores
    
    def select_features(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        n_features: Optional[int] = None,
        max_correlation: float = 0.95,
    ) -> List[str]:
        """
        Select best features.
        
        Strategy:
        1. Remove low-variance features
        2. Remove correlated features (keep higher importance)
        3. If y provided, rank by importance
        4. Return top n_features
        
        Args:
            X: Feature dataframe
            y: Optional target
            n_features: Number of features to select  (default=50)
            max_correlation: Max correlation to allow
            
        Returns:
            List of selected feature names
        """
        if n_features is None:
            n_features = min(50, X.shape[1] // 2)
        
        # Step 1: Remove low variance
        selected = [col for col in X.columns if col not in self._find_low_variance_features(X)]
        X_selected = X[selected]
        
        # Step 2: Remove correlated
        correlated = self._find_correlated_features(X_selected)
        to_remove = set()
        
        for feature, corr_features in correlated.items():
            # Keep the one with higher importance if y available
            if y is not None:
                try:
                    importance = self._compute_feature_importance(X_selected[[feature] + corr_features], y)
                    # Remove all but the top one
                    to_remove.update(importance.tail(len(corr_features) - 1).index)
                except:
                    to_remove.update(corr_features[1:])  # Keep first, remove rest
            else:
                to_remove.update(corr_features[1:])
        
        selected = [col for col in selected if col not in to_remove]
        X_selected = X[selected]
        
        # Step 3: Rank by importance
        if y is not None and len(selected) > n_features:
            importance = self._compute_feature_importance(X_selected, y)
            selected = importance.head(n_features).index.tolist()
        else:
            selected = selected[:n_features]
        
        logger.info(f"Selected {len(selected)} features from {X.shape[1]} total")
        
        return selected
