"""
Backtesting Bias Detector

Identifies common backtesting pitfalls:
- Look-ahead bias (using future data)
- Survivorship bias
- Data snooping
- Calendar effects
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class BacktestingBiasDetector:
    """Detect common biases in backtesting setup."""
    
    def __init__(self):
        pass
    
    def detect_lookahead_bias(
        self,
        df: pd.DataFrame,
        signal_cols: List[str],
        price_col: str = "Close",
        date_col: str = "Date",
    ) -> Dict[str, any]:
        """
        Detect look-ahead bias:
        - If signal uses today's close, actual trade is tomorrow
        - If signal uses today's high/low, actual trade uses worse price
        
        Args:
            df: Backtest dataframe
            signal_cols: Columns that generate signals
            price_col: Price column used for returns
            date_col: Date column
            
        Returns:
            Dictionary with bias check results
        """
        issues = []
        
        # 1. Check if OHLC data is used correctly
        for col in signal_cols:
            if col not in df.columns:
                continue
            
            # Flag: signal generated from same day close price
            if "Close" in df[col].astype(str).values or price_col in df.columns:
                issues.append(
                    f"Signal '{col}' may use same-day close. "
                    f"Actual trade should use next-day open."
                )
        
        # 2. Check: are column names suggesting future data?
        future_names = ["next_", "tomorrow_", "forward_", "_future", "_next"]
        for col in df.columns:
            if any(future_name in col.lower() for future_name in future_names):
                issues.append(f"Column '{col}' looks like future data")
        
        # 3. Check: if data is sorted chronologically
        if date_col in df.columns:
            dates = pd.to_datetime(df[date_col])
            if not dates.is_monotonic_increasing:
                issues.append("Data not sorted chronologically - serious look-ahead risk")
        
        return {
            "bias_detected": len(issues) > 0,
            "issues": issues,
            "severity": "HIGH" if len(issues) > 1 else ("MEDIUM" if len(issues) == 1 else "NONE"),
        }
    
    def detect_survivorship_bias(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, any]:
        """
        Detect survivorship bias - backtest only includes surviving assets.
        
        Args:
            df: Backtest dataframe
            
        Returns:
            Dictionary with bias check results
        """
        issues = []
        
        # 1. Check for sudden data gaps (delisted assets)
        if "Close" in df.columns:
            data_gap = df["Close"].isna().sum()
            if data_gap > len(df) * 0.01:  # >1% missing
                issues.append(
                    f"Found {data_gap} NaN values ({data_gap/len(df)*100:.1f}%). "
                    f"This could indicate delisted/suspended stocks."
                )
        
        # 2. Check if symbols remain constant (no delisting)
        if "symbol" in df.columns:
            first_symbols = set(df.iloc[:len(df)//4]["symbol"].unique())
            last_symbols = set(df.iloc[-len(df)//4:]["symbol"].unique())
            
            removed_symbols = first_symbols - last_symbols
            if removed_symbols:
                issues.append(
                    f"Symbols disappeared over time: {removed_symbols}. "
                    f"This is survivorship bias."
                )
        
        return {
            "bias_detected": len(issues) > 0,
            "issues": issues,
            "severity": "HIGH" if len(issues) > 0 else "NONE",
        }
    
    def detect_data_snooping(
        self,
        df: pd.DataFrame,
        param_ranges: Dict[str, List[float]] = None,
    ) -> Dict[str, any]:
        """
        Detect overfitting/data snooping.
        
        Args:
            df: Backtest dataframe
            param_ranges: Dict of parameter ranges tested
            
        Returns:
            Dictionary with warnings
        """
        issues = []
        
        # 1. Large number of parameters
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 100:
            issues.append(
                f"Very large feature set ({len(numeric_cols)} features). "
                f"High risk of overfitting. Use feature selection."
            )
        
        # 2. Check if validation set used
        # This would typically be detected by cross-validation setup
        issues.append(
            "WARNING: Manual inspection needed - do you use walk-forward validation?"
        )
        
        # 3. Check parameter sensitivity
        if param_ranges:
            total_combinations = 1
            for param, values in param_ranges.items():
                total_combinations *= len(values)
            
            if total_combinations > 1000:
                issues.append(
                    f"High parameter combinations tested ({total_combinations}). "
                    f"Likely overfitting. Use nested cross-validation."
                )
        
        return {
            "bias_detected": len(issues) > 0,
            "issues": issues,
            "severity": "MEDIUM",
        }
    
    def validate_feature_leakage(
        self,
        df: pd.DataFrame,
        date_col: str = "Date",
    ) -> Dict[str, any]:
        """
        Detect feature leakage - using future information in features.
        
        Args:
            df: Dataframe with features
            date_col: Date column
            
        Returns:
            Dictionary with leakage checks
        """
        issues = []
        
        # Check for shifted features (should shift > 1)
        shifted_cols = [col for col in df.columns if "shift" in col.lower() or "lag" in col.lower()]
        
        for col in shifted_cols:
            # Extract shift amount if possible
            if "shift" in col.lower():
                # Warn if shift is 0 or 1
                issues.append(f"Column '{col}': verify shift period is >1 day")
        
        # Check for rolling statistics (should be lagged)
        rolling_cols = [col for col in df.columns if "rolling" in col.lower() or "_ma" in col.lower()]
        if rolling_cols:
            issues.append(
                f"Rolling features ({len(rolling_cols)}): ensure they use past data only, "
                f"not including current bar"
            )
        
        # Check for forward-looking fields
        forward_indicators = ["forecast", "predict", "expected", "next_return", "future"]
        for col in df.columns:
            if any(ind in col.lower() for ind in forward_indicators):
                issues.append(f"Column '{col}' looks like forward-looking data")
        
        return {
            "leakage_detected": len(issues) > 0,
            "issues": issues,
            "severity": "HIGH" if len(issues) > 5 else "MEDIUM" if len(issues) > 0 else "NONE",
        }
    
    def run_full_audit(
        self,
        df: pd.DataFrame,
        signal_cols: List[str],
        param_ranges: Dict[str, List[float]],
        date_col: str = "Date",
    ) -> Dict[str, Dict]:
        """
        Run complete backtesting audit.
        
        Args:
            df: Backtest dataframe
            signal_cols: Signal column names
            param_ranges: Parameter combinations tested
            date_col: Date column
            
        Returns:
            Complete audit report
        """
        logger.info("=" * 60)
        logger.info("BACKTESTING BIAS AUDIT")
        logger.info("=" * 60)
        
        audit = {
            "lookahead_bias": self.detect_lookahead_bias(df, signal_cols, date_col=date_col),
            "survivorship_bias": self.detect_survivorship_bias(df),
            "data_snooping": self.detect_data_snooping(df, param_ranges),
            "feature_leakage": self.validate_feature_leakage(df, date_col),
        }
        
        # Print summary
        total_issues = sum(len(v.get("issues", [])) for v in audit.values())
        
        logger.info(f"\n1. LOOK-AHEAD BIAS: {audit['lookahead_bias']['severity']}")
        for issue in audit['lookahead_bias'].get('issues', []):
            logger.warning(f"   ⚠️  {issue}")
        
        logger.info(f"\n2. SURVIVORSHIP BIAS: {audit['survivorship_bias']['severity']}")
        for issue in audit['survivorship_bias'].get('issues', []):
            logger.warning(f"   ⚠️  {issue}")
        
        logger.info(f"\n3. DATA SNOOPING: {audit['data_snooping']['severity']}")
        for issue in audit['data_snooping'].get('issues', []):
            logger.warning(f"   ⚠️  {issue}")
        
        logger.info(f"\n4. FEATURE LEAKAGE: {audit['feature_leakage']['severity']}")
        for issue in audit['feature_leakage'].get('issues', []):
            logger.warning(f"   ⚠️  {issue}")
        
        logger.info("\n" + "=" * 60)
        logger.info(f"TOTAL ISSUES FOUND: {total_issues}")
        logger.info("=" * 60)
        
        audit['total_issues'] = total_issues
        
        return audit
