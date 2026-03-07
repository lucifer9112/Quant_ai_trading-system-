"""
Drawdown Analysis - Detailed Drawdown Period Investigation

Analyzes:
- Drawdown periods (start, end, depth, duration)
- Recovery time
- Drawdown statistics
- Underwater plot data
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DrawdownPeriod:
    """Single drawdown period"""
    start_idx: int
    end_idx: int
    start_value: float
    bottom_value: float
    end_value: float
    depth: float  # As percentage
    duration: int  # In periods
    recovery_idx: Optional[int]  # Index when recovered
    recovery_duration: Optional[int]


class DrawdownAnalyzer:
    """
    Analyzes drawdown characteristics and periods.
    """
    
    def __init__(self):
        pass
    
    def analyze_equity_curve(
        self,
        equity_curve: List[float],
    ) -> Dict:
        """
        Comprehensive drawdown analysis.
        
        Args:
            equity_curve: List of portfolio values
            
        Returns:
            Dict with all drawdown statistics
        """
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        
        # Overall statistics
        max_drawdown = np.min(drawdown)
        avg_drawdown = np.mean(drawdown[drawdown < 0]) if np.any(drawdown < 0) else 0.0
        
        # Drawdown periods
        periods = self._identify_drawdown_periods(equity_array, drawdown)
        
        # Current drawdown
        current_dd = drawdown[-1]
        
        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'current_drawdown': current_dd,
            'num_drawdown_periods': len(periods),
            'periods': periods,
            'underwater_curve': drawdown,
        }
    
    def _identify_drawdown_periods(
        self,
        equity_curve: np.ndarray,
        drawdown: np.ndarray,
    ) -> List[DrawdownPeriod]:
        """
        Identify all drawdown periods and their characteristics.
        
        Args:
            equity_curve: Equity values
            drawdown: Drawdown percentages
            
        Returns:
            List of DrawdownPeriod objects
        """
        periods = []
        in_drawdown = False
        start_idx = 0
        peak_value = equity_curve[0]
        
        for i in range(len(drawdown)):
            if drawdown[i] < 0 and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                start_idx = i
                peak_value = equity_curve[i - 1] if i > 0 else equity_curve[0]
                bottom_value = equity_curve[i]
            
            elif drawdown[i] < 0 and in_drawdown:
                # Update bottom
                bottom_value = min(bottom_value, equity_curve[i])
            
            elif drawdown[i] >= 0 and in_drawdown:
                # End of drawdown - recovery
                in_drawdown = False
                recovery_idx = i
                
                period = DrawdownPeriod(
                    start_idx=start_idx,
                    end_idx=i - 1,
                    start_value=peak_value,
                    bottom_value=bottom_value,
                    end_value=equity_curve[i],
                    depth=(bottom_value - peak_value) / peak_value,
                    duration=i - 1 - start_idx,
                    recovery_idx=recovery_idx,
                    recovery_duration=recovery_idx - start_idx,
                )
                periods.append(period)
        
        # Handle ongoing drawdown
        if in_drawdown:
            bottom_value = np.min(equity_curve[start_idx:])
            period = DrawdownPeriod(
                start_idx=start_idx,
                end_idx=len(equity_curve) - 1,
                start_value=peak_value,
                bottom_value=bottom_value,
                end_value=equity_curve[-1],
                depth=(bottom_value - peak_value) / peak_value,
                duration=len(equity_curve) - 1 - start_idx,
                recovery_idx=None,
                recovery_duration=None,
            )
            periods.append(period)
        
        return periods
    
    def drawdown_statistics(
        self,
        periods: List[DrawdownPeriod],
    ) -> Dict:
        """
        Calculate statistics about drawdown periods.
        
        Args:
            periods: List of DrawdownPeriod objects
            
        Returns:
            Dict with statistics
        """
        if not periods:
            return {
                'num_periods': 0,
                'avg_depth': 0.0,
                'max_depth': 0.0,
                'avg_duration': 0.0,
                'max_duration': 0.0,
                'avg_recovery_time': 0.0,
                'worst_period_depth': 0.0,
                'worst_period_duration': 0,
            }
        
        depths = [p.depth for p in periods]
        durations = [p.duration for p in periods]
        recovery_times = [p.recovery_duration for p in periods if p.recovery_duration is not None]
        
        return {
            'num_periods': len(periods),
            'avg_depth': np.mean(depths),
            'max_depth': np.min(depths),  # Most negative
            'avg_duration': np.mean(durations),
            'max_duration': np.max(durations),
            'avg_recovery_time': np.mean(recovery_times) if recovery_times else None,
            'worst_period_depth': min(depths),
            'worst_period_duration': max(durations),
        }
    
    def underwater_plot(
        self,
        equity_curve: List[float],
    ) -> pd.DataFrame:
        """
        Generate data for underwater plot visualization.
        
        Args:
            equity_curve: List of portfolio values
            
        Returns:
            DataFrame with underwater plot data
        """
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max * 100  # As percentage
        
        df = pd.DataFrame({
            'Period': range(len(drawdown)),
            'Drawdown_Pct': drawdown,
        })
        
        return df
    
    def recovery_time_analysis(
        self,
        periods: List[DrawdownPeriod],
    ) -> Dict:
        """
        Analyze recovery characteristics.
        
        Args:
            periods: List of DrawdownPeriod objects
            
        Returns:
            Dict with recovery analysis
        """
        recovered_periods = [p for p in periods if p.recovery_idx is not None]
        
        if not recovered_periods:
            return {
                'num_recovered': 0,
                'avg_recovery_time': None,
                'max_recovery_time': None,
                'min_recovery_time': None,
            }
        
        recovery_times = [p.recovery_duration for p in recovered_periods]
        
        return {
            'num_recovered': len(recovered_periods),
            'avg_recovery_time': np.mean(recovery_times),
            'max_recovery_time': np.max(recovery_times),
            'min_recovery_time': np.min(recovery_times),
            'longest_recovery_period': max(recovered_periods, key=lambda p: p.recovery_duration),
        }
    
    def drawdown_impact(
        self,
        periods: List[DrawdownPeriod],
        total_return: float,
    ) -> Dict:
        """
        Analyze impact of drawdowns on total return.
        
        Args:
            periods: List of DrawdownPeriod objects
            total_return: Total return of strategy
            
        Returns:
            Dict with impact analysis
        """
        if not periods:
            return {
                'total_drawdown_impact': 0.0,
                'num_drawdowns': 0,
                'avg_impact_per_drawdown': 0.0,
            }
        
        impact_values = []
        
        for period in periods:
            # Impact = how much the drawdown reduced final return
            if period.start_value > 0:
                impact = (period.bottom_value - period.start_value) / period.start_value
                impact_values.append(impact)
        
        total_impact = sum(impact_values) if impact_values else 0.0
        
        return {
            'total_drawdown_impact': total_impact,
            'num_drawdowns': len(periods),
            'avg_impact_per_drawdown': np.mean(impact_values) if impact_values else 0.0,
            'worst_impact': min(impact_values) if impact_values else 0.0,
        }
    
    def recovery_delta(
        self,
        periods: List[DrawdownPeriod],
    ) -> Dict:
        """
        Analyze gap between drawdown bottom and recovery point.
        
        Args:
            periods: List of DrawdownPeriod objects
            
        Returns:
            Dict with recovery gap analysis
        """
        recovered_periods = [p for p in periods if p.recovery_idx is not None]
        
        if not recovered_periods:
            return {
                'avg_recovery_gap_pct': None,
                'max_recovery_gap_pct': None,
            }
        
        gaps = []
        
        for period in recovered_periods:
            gap = (period.end_value - period.bottom_value) / period.bottom_value
            gaps.append(gap)
        
        return {
            'avg_recovery_gap_pct': np.mean(gaps),
            'max_recovery_gap_pct': np.max(gaps),
            'gaps': gaps,
        }
