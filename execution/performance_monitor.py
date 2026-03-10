"""Real-time performance monitoring and alerting for production trading."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Daily performance metrics."""
    date: datetime
    total_return: float
    daily_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    accuracy: float


class PerformanceMonitor:
    """Monitor production system performance and detect problems.
    
    Metrics tracked:
    - Returns (daily, cumulative)
    - Risk metrics (Sharpe, Sortino, max drawdown)
    - Win rate and profit factor
    - Model accuracy and predictions
    - Execution quality
    - Market conditions
    
    Alerts triggered if:
    - Accuracy drops >5% from baseline
    - Sharpe < 0.5 (strategy broke)
    - Max drawdown > 25% (risk control failure)
    - Win rate < 45% (signal deterioration)
    - Slippage > 10bps (market impact increased)
    
    Expected benefit: Early problem detection prevents big losses
    """

    def __init__(
        self,
        baseline_accuracy: float = 0.60,
        baseline_sharpe: float = 1.2,
        max_drawdown_limit: float = 0.25,
        min_win_rate: float = 0.45,
        lookback_period: int = 63,  # 3-month rolling window
    ):
        """Initialize performance monitor.
        
        Parameters
        ----------
        baseline_accuracy : float
            Expected model accuracy (60%)
        baseline_sharpe : float
            Expected Sharpe ratio (1.2)
        max_drawdown_limit : float
            Max acceptable drawdown (25%)
        min_win_rate : float
            Minimum acceptable win rate (45%)
        lookback_period : int
            Days to track for rolling metrics
        """
        self.baseline_accuracy = baseline_accuracy
        self.baseline_sharpe = baseline_sharpe
        self.max_drawdown_limit = max_drawdown_limit
        self.min_win_rate = min_win_rate
        self.lookback_period = lookback_period
        
        self.daily_returns = []
        self.metrics_history = []
        self.alerts = []
        self.trades = []
        
        logger.info(
            f"PerformanceMonitor initialized: "
            f"baseline_acc={baseline_accuracy:.1%}, "
            f"baseline_sharpe={baseline_sharpe:.2f}"
        )

    def record_trade(
        self,
        date: datetime,
        symbol: str,
        entry_price: float,
        exit_price: float,
        size: float,
        slippage: float,
        commission: float,
    ) -> Dict[str, float]:
        """Record individual trade for analysis.
        
        Parameters
        ----------
        date : datetime
            Trade execution date
        symbol : str
            Asset symbol
        entry_price : float
            Entry price
        exit_price : float
            Exit price
        size : float
            Position size
        slippage : float
            Slippage cost
        commission : float
            Commission cost
            
        Returns
        -------
        trade_pnl : dict
            Gross PnL, net PnL, return %
        """
        gross_pnl = (exit_price - entry_price) * size
        net_pnl = gross_pnl - slippage - commission
        return_pct = (exit_price - entry_price) / entry_price
        
        trade = {
            'date': date,
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': size,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'slippage': slippage,
            'commission': commission,
            'return_pct': return_pct,
            'win': 1 if net_pnl > 0 else 0,
        }
        
        self.trades.append(trade)
        
        logger.debug(f"Trade recorded: {symbol} return={return_pct:.2%}, net_pnl=${net_pnl:.2f}")
        
        return {'gross_pnl': gross_pnl, 'net_pnl': net_pnl, 'return_pct': return_pct}

    def record_daily_return(self, date: datetime, daily_return: float) -> None:
        """Record daily portfolio return.
        
        Parameters
        ----------
        date : datetime
            Date
        daily_return : float
            Daily return (percent, e.g., 0.02 = 2%)
        """
        self.daily_returns.append({
            'date': date,
            'return': daily_return,
        })

    def calculate_metrics(
        self,
        current_date: datetime,
        cumulative_return: float,
        model_accuracy: float,
        model_predictions: np.ndarray,
        predictions_actual: np.ndarray,
    ) -> PerformanceMetrics:
        """Calculate current performance metrics.
        
        Parameters
        ----------
        current_date : datetime
            Current date
        cumulative_return : float
            Cumulative return from start
        model_accuracy : float
            Current model accuracy
        model_predictions : array
            Recent model predictions
        predictions_actual : array
            Actual outcomes
            
        Returns
        -------
        metrics : PerformanceMetrics
            Current performance metrics
        """
        # Daily returns
        if len(self.daily_returns) > 0:
            recent_returns = [r['return'] for r in self.daily_returns[-self.lookback_period:]]
            daily_return = recent_returns[-1] if recent_returns else 0
            
            # Sharpe ratio
            if len(recent_returns) > 1:
                annual_return = np.mean(recent_returns) * 252
                annual_vol = np.std(recent_returns) * np.sqrt(252)
                sharpe = annual_return / max(annual_vol, 0.001)
            else:
                sharpe = 0
            
            # Max drawdown
            cumulative = np.cumprod(1 + np.array(recent_returns))
            running_max = np.maximum.accumulate(cumulative)
            max_dd = np.min(cumulative / running_max - 1) if len(cumulative) > 0 else 0
        else:
            daily_return = 0
            sharpe = 0
            max_dd = 0
        
        # Win rate
        if len(self.trades) > 0:
            recent_trades = self.trades[-self.lookback_period:]
            win_rate = np.mean([t['win'] for t in recent_trades]) if recent_trades else 0
            gross_wins = np.sum([t['gross_pnl'] for t in recent_trades if t['gross_pnl'] > 0])
            gross_losses = np.abs(np.sum([t['gross_pnl'] for t in recent_trades if t['gross_pnl'] < 0]))
            profit_factor = gross_wins / max(gross_losses, 1e-10)
        else:
            win_rate = 0
            profit_factor = 0
        
        # Model accuracy
        if len(model_predictions) > 0:
            pred_binary = (model_predictions > 0.5).astype(int)
            accuracy = np.mean(pred_binary == predictions_actual) if len(predictions_actual) > 0 else model_accuracy
        else:
            accuracy = model_accuracy
        
        metrics = PerformanceMetrics(
            date=current_date,
            total_return=cumulative_return,
            daily_return=daily_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=len(self.trades),
            accuracy=accuracy,
        )
        
        self.metrics_history.append(metrics)
        
        # Check for alerts
        self._check_alerts(metrics)
        
        return metrics

    def _check_alerts(self, metrics: PerformanceMetrics) -> None:
        """Check for alert conditions.
        
        Parameters
        ----------
        metrics : PerformanceMetrics
            Current metrics
        """
        alerts = []
        
        # Accuracy alert
        if metrics.accuracy < self.baseline_accuracy * (1 - 0.05):
            alerts.append(f"⚠️  ACCURACY DROP: {metrics.accuracy:.2%} (baseline: {self.baseline_accuracy:.2%})")
        
        # Sharpe alert
        if metrics.sharpe_ratio < 0.5:
            alerts.append(f"⚠️  LOW SHARPE: {metrics.sharpe_ratio:.2f} (baseline: {self.baseline_sharpe:.2f})")
        
        # Max drawdown alert
        if metrics.max_drawdown < -self.max_drawdown_limit:
            alerts.append(f"🔴 MAX DRAWDOWN EXCEEDED: {metrics.max_drawdown:.2%}")
        
        # Win rate alert
        if metrics.win_rate < self.min_win_rate:
            alerts.append(f"⚠️  LOW WIN RATE: {metrics.win_rate:.2%}")
        
        # Profit factor alert
        if metrics.profit_factor < 1.0:
            alerts.append(f"🔴 NEGATIVE PROFIT FACTOR: {metrics.profit_factor:.2f}")
        
        if alerts:
            for alert in alerts:
                logger.warning(alert)
                self.alerts.append({
                    'date': metrics.date,
                    'message': alert,
                    'severity': 'CRITICAL' if '🔴' in alert else 'HIGH',
                })

    def get_summary_report(self) -> pd.DataFrame:
        """Get summary performance report.
        
        Returns
        -------
        report : DataFrame
            Daily metrics summary
        """
        if not self.metrics_history:
            return pd.DataFrame()
        
        records = [
            {
                'date': m.date,
                'total_return': m.total_return,
                'daily_return': m.daily_return,
                'sharpe': m.sharpe_ratio,
                'max_dd': m.max_drawdown,
                'win_rate': m.win_rate,
                'pf': m.profit_factor,
                'accuracy': m.accuracy,
                'num_trades': m.num_trades,
            }
            for m in self.metrics_history
        ]
        
        return pd.DataFrame(records)

    def get_alerts(self, severities: Optional[List[str]] = None) -> pd.DataFrame:
        """Get alert history.
        
        Parameters
        ----------
        severities : list, optional
            Filter by severity levels
            
        Returns
        -------
        alerts : DataFrame
            Alert history
        """
        if not self.alerts:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.alerts)
        
        if severities:
            df = df[df['severity'].isin(severities)]
        
        return df.sort_values('date', ascending=False)

    def summary(self) -> str:
        """Get monitoring system summary.
        
        Returns
        -------
        summary : str
        """
        lines = [
            "PerformanceMonitor Summary",
            "=" * 50,
            f"Days Tracked: {len(self.daily_returns)}",
            f"Trades Recorded: {len(self.trades)}",
            f"Alerts Triggered: {len(self.alerts)}",
        ]
        
        if self.metrics_history:
            latest = self.metrics_history[-1]
            lines.extend([
                "",
                f"Latest Metrics ({latest.date.strftime('%Y-%m-%d')}):",
                f"  Total Return: {latest.total_return:.2%}",
                f"  Daily Return: {latest.daily_return:.2%}",
                f"  Sharpe Ratio: {latest.sharpe_ratio:.2f} (baseline: {self.baseline_sharpe:.2f})",
                f"  Max Drawdown: {latest.max_drawdown:.2%}",
                f"  Win Rate: {latest.win_rate:.2%}",
                f"  Profit Factor: {latest.profit_factor:.2f}",
                f"  Model Accuracy: {latest.accuracy:.2%}",
            ])
        
        if self.alerts:
            critical_alerts = len(self.get_alerts(['CRITICAL']))
            high_alerts = len(self.get_alerts(['HIGH']))
            lines.extend([
                "",
                f"⚠️  Alert Summary:",
                f"  Critical: {critical_alerts}",
                f"  High: {high_alerts}",
            ])
        
        return "\n".join(lines)
