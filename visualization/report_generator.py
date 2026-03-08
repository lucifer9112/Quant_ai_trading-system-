"""
Report Generator - Combine visualizations into professional PDF reports

Generates:
- Full strategy reports with all metrics and charts
- Trade analysis reports
- Risk analysis reports
- Executive summaries
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import os


class ReportGenerator:
    """Generate professional PDF reports from backtest results."""
    
    def __init__(self, output_dir: str = "reports"):
        """
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_strategy_report(
        self,
        figures: List[plt.Figure],
        title: str = "Strategy Backtest Report",
        summary: Optional[Dict] = None,
        filename: Optional[str] = None,
    ) -> str:
        """
        Combine figures into a PDF report.
        
        Args:
            figures: List of matplotlib figures to include
            title: Report title
            summary: Optional summary metrics dictionary
            filename: Output filename (auto-generated if not provided)
            
        Returns:
            Path to generated PDF
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.pdf"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with PdfPages(filepath) as pdf:
            # Title page
            title_fig = self._create_title_page(title, summary)
            pdf.savefig(title_fig, bbox_inches='tight')
            plt.close(title_fig)
            
            # Add all figures
            for fig in figures:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # Table of contents
            pdf.infodict()['Title'] = title
            pdf.infodict()['Author'] = 'Quant AI Trading System'
            pdf.infodict()['Subject'] = 'Strategy Backtest Report'
            pdf.infodict()['Keywords'] = 'Trading, Backtest, Analysis'
            pdf.infodict()['CreationDate'] = datetime.now()
        
        return filepath
    
    def _create_title_page(
        self,
        title: str,
        summary: Optional[Dict] = None,
    ) -> plt.Figure:
        """Create professional title page."""
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.90, title, fontsize=28, fontweight='bold',
               ha='center', va='top', transform=ax.transAxes)
        
        # Subtitle
        ax.text(0.5, 0.83, "Quantitative Trading System Analysis",
               fontsize=14, ha='center', va='top', transform=ax.transAxes,
               style='italic', color='#666666')
        
        # Date
        ax.text(0.5, 0.75, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
               fontsize=11, ha='center', va='top', transform=ax.transAxes,
               color='#666666')
        
        # Summary section
        if summary:
            y_pos = 0.65
            ax.text(0.5, y_pos, "Key Metrics Summary", fontsize=16,
                   fontweight='bold', ha='center', transform=ax.transAxes)
            
            y_pos -= 0.08
            for key, value in summary.items():
                if isinstance(value, float):
                    if abs(value) < 100:
                        text = f"{key}: {value:.2f}"
                    else:
                        text = f"{key}: ${value:,.2f}"
                else:
                    text = f"{key}: {value}"
                
                ax.text(0.5, y_pos, text, fontsize=11, ha='center',
                       transform=ax.transAxes, family='monospace')
                y_pos -= 0.06
        
        # Footer
        ax.text(0.5, 0.05, "Confidential - For Authorized Use Only",
               fontsize=10, ha='center', va='bottom', transform=ax.transAxes,
               style='italic', color='#999999')
        
        return fig
    
    def generate_trade_analysis_report(
        self,
        figures: List[plt.Figure],
        trade_stats: Dict,
        filename: Optional[str] = None,
    ) -> str:
        """
        Generate detailed trade analysis report.
        
        Args:
            figures: List of trade analysis figures
            trade_stats: Dictionary of trade statistics
            filename: Output filename
            
        Returns:
            Path to generated PDF
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trade_analysis_{timestamp}.pdf"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with PdfPages(filepath) as pdf:
            # Title page with trade stats
            title_fig = self._create_trade_title_page(trade_stats)
            pdf.savefig(title_fig, bbox_inches='tight')
            plt.close(title_fig)
            
            # Trade figures
            for fig in figures:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
        
        return filepath
    
    def _create_trade_title_page(self, trade_stats: Dict) -> plt.Figure:
        """Create trade analysis title page."""
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.90, "Trade Analysis Report",
               fontsize=28, fontweight='bold', ha='center', va='top',
               transform=ax.transAxes)
        
        # Stats
        y_pos = 0.80
        
        stats_text = f"""
Total Trades: {trade_stats.get('total_trades', 0)}
Winning Trades: {trade_stats.get('winning_trades', 0)}
Losing Trades: {trade_stats.get('losing_trades', 0)}
Win Rate: {trade_stats.get('win_rate', 0):.1f}%

Average Win: ${trade_stats.get('avg_win', 0):,.2f}
Average Loss: ${trade_stats.get('avg_loss', 0):,.2f}
Profit Factor: {trade_stats.get('profit_factor', 0):.2f}
Expectancy: ${trade_stats.get('expectancy', 0):,.2f}

Max Consecutive Wins: {trade_stats.get('max_win_streak', 0)}
Max Consecutive Losses: {trade_stats.get('max_loss_streak', 0)}
Longest Trade (days): {trade_stats.get('longest_trade', 0)}
Shortest Trade (days): {trade_stats.get('shortest_trade', 0)}
"""
        
        ax.text(0.5, y_pos, stats_text, fontsize=11, ha='center', va='top',
               transform=ax.transAxes, family='monospace', 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        return fig
    
    def generate_risk_report(
        self,
        figures: List[plt.Figure],
        risk_metrics: Dict,
        filename: Optional[str] = None,
    ) -> str:
        """
        Generate risk analysis report.
        
        Args:
            figures: List of risk analysis figures
            risk_metrics: Dictionary of risk metrics
            filename: Output filename
            
        Returns:
            Path to generated PDF
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"risk_analysis_{timestamp}.pdf"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with PdfPages(filepath) as pdf:
            # Title page
            title_fig = self._create_risk_title_page(risk_metrics)
            pdf.savefig(title_fig, bbox_inches='tight')
            plt.close(title_fig)
            
            # Risk figures
            for fig in figures:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
        
        return filepath
    
    def _create_risk_title_page(self, risk_metrics: Dict) -> plt.Figure:
        """Create risk analysis title page."""
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.90, "Risk Analysis Report",
               fontsize=28, fontweight='bold', ha='center', va='top',
               transform=ax.transAxes)
        
        # Risk metrics
        y_pos = 0.80
        
        risk_text = f"""
Maximum Drawdown: {risk_metrics.get('max_drawdown', 0):.2f}%
Avg Drawdown: {risk_metrics.get('avg_drawdown', 0):.2f}%
Longest Drawdown (days): {risk_metrics.get('max_dd_duration', 0)}

Value at Risk (95%): {risk_metrics.get('var_95', 0):.2f}%
Conditional VaR (95%): {risk_metrics.get('cvar_95', 0):.2f}%

Annual Volatility: {risk_metrics.get('annual_volatility', 0):.2f}%
Skewness: {risk_metrics.get('skewness', 0):.2f}
Kurtosis: {risk_metrics.get('kurtosis', 0):.2f}

Recovery Factor: {risk_metrics.get('recovery_factor', 0):.2f}
Tail Ratio: {risk_metrics.get('tail_ratio', 0):.2f}
"""
        
        ax.text(0.5, y_pos, risk_text, fontsize=11, ha='center', va='top',
               transform=ax.transAxes, family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        return fig
    
    def generate_summary_table(
        self,
        data: Dict[str, List],
        filename: Optional[str] = None,
    ) -> str:
        """
        Generate summary table page as PDF.
        
        Args:
            data: Dictionary mapping column names to lists of values
            filename: Output filename
            
        Returns:
            Path to generated PDF
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"summary_table_{timestamp}.pdf"
        
        filepath = os.path.join(self.output_dir, filename)
        
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        # Create table
        table_data = []
        headers = list(data.keys())
        
        # Transpose data
        n_rows = len(next(iter(data.values())))
        for row_idx in range(n_rows):
            row = [data[col][row_idx] for col in headers]
            table_data.append(row)
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F0F0F0')
        
        plt.title("Summary Table", fontsize=16, fontweight='bold', pad=20)
        
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        return filepath
