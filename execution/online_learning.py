"""Online learning for continuous model updates in production."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class OnlineLearning:
    """Update models incrementally with new data.
    
    Key insights:
    - Market conditions change over time
    - Models trained 6 months ago may be stale
    - Daily retraining too expensive in production
    - Solution: Smart incremental updates
    
    Strategy:
    - Every N days (rolling window), retrain meta-learner
    - Keep base learners frozen (expensive to retrain)
    - Monitor for concept drift using performance metrics
    - Trigger full retraining if accuracy drops >5%
    
    Expected benefit: 1-2% accuracy improvement via freshness
    """

    def __init__(
        self,
        initial_model: Any,
        retraining_frequency: int = 21,  # Retrain every 3 weeks
        drift_threshold: float = 0.05,  # 5% accuracy drop triggers alert
        window_size: int = 252,  # 1 year of data for retraining
    ):
        """Initialize online learning system.
        
        Parameters
        ----------
        initial_model : model
            Initial trained model (ensemble or learner)
        retraining_frequency : int
            Days between retraining cycles
        drift_threshold : float
            Accuracy drop threshold for drift alert
        window_size : int
            Size of rolling training window
        """
        self.initial_model = initial_model
        self.current_model = initial_model
        self.retraining_frequency = retraining_frequency
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        
        self.last_retrain_date = datetime.now()
        self.performance_history = deque(maxlen=window_size)
        self.retraining_history = []
        self.drift_detected = False
        
        logger.info(
            f"OnlineLearning initialized: "
            f"retraining_freq={retraining_frequency}d, "
            f"drift_threshold={drift_threshold*100:.1f}%"
        )

    def update_performance_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        date: datetime = None,
    ) -> Dict[str, float]:
        """Record and analyze performance on new data.
        
        Parameters
        ----------
        predictions : array
            Model predictions
        actuals : array
            Actual outcomes
        date : datetime, optional
            Date of predictions
            
        Returns
        -------
        metrics : dict
            Performance metrics (accuracy, precision, f1)
        """
        date = date or datetime.now()
        
        # Binary classification metrics
        pred_binary = (predictions > 0.5).astype(int)
        
        accuracy = np.mean(pred_binary == actuals)
        precision = np.sum((pred_binary == 1) & (actuals == 1)) / (np.sum(pred_binary == 1) + 1e-10)
        recall = np.sum((pred_binary == 1) & (actuals == 1)) / (np.sum(actuals == 1) + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        metrics = {
            'date': date,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'sample_size': len(actuals),
        }
        
        self.performance_history.append(metrics)
        
        # Check for concept drift
        if len(self.performance_history) > 10:
            recent_acc = np.mean([m['accuracy'] for m in list(self.performance_history)[-10:]])
            baseline_acc = np.mean([m['accuracy'] for m in list(self.performance_history)[:10]])
            
            if baseline_acc - recent_acc > self.drift_threshold:
                self.drift_detected = True
                logger.warning(
                    f"Concept drift detected! "
                    f"Accuracy dropped {(baseline_acc - recent_acc)*100:.2f}% "
                    f"({baseline_acc:.2%} → {recent_acc:.2%})"
                )
        
        logger.debug(f"Performance recorded: accuracy={accuracy:.4f}, f1={f1:.4f}")
        
        return metrics

    def should_retrain(self) -> bool:
        """Check if retraining is needed.
        
        Returns
        -------
        should_retrain : bool
            True if scheduled retraining or drift detected
        """
        days_since_retrain = (datetime.now() - self.last_retrain_date).days
        
        if days_since_retrain >= self.retraining_frequency:
            logger.info(f"Scheduled retraining triggered ({days_since_retrain} days since last)")
            return True
        
        if self.drift_detected:
            logger.warning("Unscheduled retraining triggered due to concept drift")
            return True
        
        return False

    def retrain_meta_learner(
        self,
        X_recent: np.ndarray,
        y_recent: np.ndarray,
        base_learner_predictions: Optional[Dict[str, np.ndarray]] = None,
    ) -> Any:
        """Retrain meta-learner on recent data.
        
        Note: We keep base learners frozen to reduce computation.
        Only the meta-learner (2nd level) is retrained.
        
        Parameters
        ----------
        X_recent : array
            Recent feature data
        y_recent : array
            Recent targets
        base_learner_predictions : dict, optional
            Pre-computed base learner predictions
            
        Returns
        -------
        retrained_model : model
            Updated meta-learner
        """
        logger.info(f"Retraining meta-learner on {len(X_recent)} recent samples")
        
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        
        # If we have ensemble with base learner predictions
        if base_learner_predictions is not None:
            # Generate meta-features from base predictions
            meta_features = np.column_stack([preds for preds in base_learner_predictions.values()])
            
            # Retrain meta-learner (Ridge)
            scaler = StandardScaler()
            meta_features_scaled = scaler.fit_transform(meta_features)
            
            meta_learner = Ridge(alpha=1.0)
            meta_learner.fit(meta_features_scaled, y_recent)
            
            logger.info("Meta-learner retraining complete")
            
            return meta_learner
        else:
            # Fallback: retrain entire model if no base predictions available
            logger.warning("Full model retraining (slower)")
            self.current_model.fit(X_recent, y_recent)
            return self.current_model

    def check_model_staleness(self) -> Dict[str, Any]:
        """Check how stale the current model is.
        
        Returns
        -------
        staleness_report : dict
            Days since training, performance degradation, recommendation
        """
        days_since_retrain = (datetime.now() - self.last_retrain_date).days
        
        # Performance trend
        if len(self.performance_history) > 5:
            recent_perf = np.mean([m['accuracy'] for m in list(self.performance_history)[-5:]])
            baseline_perf = np.mean([m['accuracy'] for m in list(self.performance_history)[:5]])
        else:
            recent_perf = baseline_perf = 0.5
        
        perf_degradation = baseline_perf - recent_perf
        
        # Recommendation
        if days_since_retrain > self.retraining_frequency * 2:
            recommendation = "URGENT: Retrain immediately"
            severity = "CRITICAL"
        elif perf_degradation > self.drift_threshold * 2:
            recommendation = "URGENT: Concept drift detected, retrain immediately"
            severity = "CRITICAL"
        elif days_since_retrain > self.retraining_frequency:
            recommendation = "Retrain soon (scheduled)"
            severity = "HIGH"
        elif perf_degradation > self.drift_threshold:
            recommendation = "Schedule retraining within 1 week"
            severity = "MEDIUM"
        else:
            recommendation = "Model is fresh, no action needed"
            severity = "LOW"
        
        report = {
            'days_since_retrain': days_since_retrain,
            'retraining_frequency': self.retraining_frequency,
            'performance_degradation': perf_degradation,
            'recent_accuracy': recent_perf,
            'baseline_accuracy': baseline_perf,
            'drift_detected': self.drift_detected,
            'recommendation': recommendation,
            'severity': severity,
        }
        
        logger.info(f"Model staleness check: {recommendation} (severity: {severity})")
        
        return report

    def get_model_performance_report(self) -> pd.DataFrame:
        """Get detailed performance report over time.
        
        Returns
        -------
        report : DataFrame
            Daily performance metrics
        """
        if not self.performance_history:
            return pd.DataFrame()
        
        records = [
            {
                'date': m['date'],
                'accuracy': m['accuracy'],
                'precision': m['precision'],
                'recall': m['recall'],
                'f1': m['f1'],
                'sample_size': m['sample_size'],
            }
            for m in self.performance_history
        ]
        
        df = pd.DataFrame(records)
        
        # Add rolling averages
        df['accuracy_ma7'] = df['accuracy'].rolling(7, min_periods=1).mean()
        df['accuracy_ma21'] = df['accuracy'].rolling(21, min_periods=1).mean()
        
        return df

    def summary(self) -> str:
        """Get online learning system summary.
        
        Returns
        -------
        summary : str
        """
        lines = [
            "OnlineLearning Summary",
            "=" * 50,
            f"Retraining Frequency: {self.retraining_frequency} days",
            f"Drift Threshold: {self.drift_threshold*100:.1f}%",
            f"Last Retraining: {self.last_retrain_date.strftime('%Y-%m-%d')}",
            f"Retraining Cycles: {len(self.retraining_history)}",
            f"Concept Drift Detected: {self.drift_detected}",
        ]
        
        if self.performance_history:
            recent_metrics = self.performance_history[-1]
            lines.extend([
                "",
                "Recent Performance:",
                f"  Accuracy: {recent_metrics['accuracy']:.4f}",
                f"  Precision: {recent_metrics['precision']:.4f}",
                f"  Recall: {recent_metrics['recall']:.4f}",
                f"  F1: {recent_metrics['f1']:.4f}",
            ])
        
        return "\n".join(lines)
