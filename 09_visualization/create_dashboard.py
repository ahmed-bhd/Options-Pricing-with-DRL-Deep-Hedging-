# -*- coding: utf-8 -*-
"""
================================================================================
COMPLETE DASHBOARD
================================================================================

This module creates a complete dashboard with all visualizations:
    - Hedging performance plots
    - Risk metrics plots
    - Transaction cost plots
    - Summary table
    - Combined dashboard figure
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add the current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import plotter modules using direct imports
try:
    from plot_hedging_performance import HedgingPerformancePlotter
    from plot_risk_metrics import RiskMetricsPlotter
    from plot_transaction_costs import TransactionCostPlotter
    print("Imported plotters successfully")
except ImportError as e:
    print(f"Import error: {e}")
    # Define placeholder classes if imports fail
    class HedgingPerformancePlotter:
        def __init__(self, *args, **kwargs): 
            pass
        
        def plot_all(self, *args, **kwargs): 
            return {}
    
    class RiskMetricsPlotter:
        def __init__(self, *args, **kwargs): 
            pass
        
        def plot_all(self, *args, **kwargs): 
            return {}
    
    class TransactionCostPlotter:
        def __init__(self, *args, **kwargs): 
            pass
        
        def plot_all(self, *args, **kwargs): 
            return {}


class DashboardCreator:
    """
    Creates a complete dashboard with all visualizations.
    
    Combines:
        - Hedging performance plots
        - Risk metrics plots
        - Transaction cost plots
        - Summary table
    """
    
    def __init__(self, 
                 results: Dict[str, Dict],
                 comparison_df: Optional[pd.DataFrame] = None,
                 figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize dashboard creator.
        
        Args:
            results: Dictionary of backtest results
            comparison_df: DataFrame with performance comparison
            figsize: Figure size for individual plots
        """
        self.results = results
        self.comparison_df = comparison_df
        self.figsize = figsize
        
        # Initialize plotters
        self.performance_plotter = HedgingPerformancePlotter(results, figsize)
        self.risk_plotter = RiskMetricsPlotter(results, figsize)
        self.cost_plotter = TransactionCostPlotter(results, figsize)
    
    def create_summary_table(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a summary table of results.
        
        Args:
            save_path: Path to save the figure
        
        Returns:
            Matplotlib figure
        """
        if self.comparison_df is None:
            # Create summary from results
            summary_data = []
            for name, data in self.results.items():
                if 'pnls' in data and data['pnls'] is not None:
                    pnls = data['pnls']
                    if len(pnls) > 0:
                        summary_data.append({
                            'Strategy': name,
                            'Mean PnL': np.mean(pnls),
                            'Std PnL': np.std(pnls),
                            'Sharpe': np.mean(pnls) / np.std(pnls) * np.sqrt(252) if np.std(pnls) > 0 else 0,
                            'Win Rate %': (pnls > 0).sum() / len(pnls) * 100
                        })
            df = pd.DataFrame(summary_data).sort_values('Mean PnL', ascending=False)
        else:
            df = self.comparison_df
        
        fig, ax = plt.subplots(figsize=(12, max(4, len(df) * 0.5)))
        ax.axis('tight')
        ax.axis('off')
        
        # Format the data
        table_data = []
        for _, row in df.iterrows():
            table_data.append([
                row['Strategy'][:25] if len(row['Strategy']) > 25 else row['Strategy'],
                f"${row['Mean PnL']:.2f}" if 'Mean PnL' in row else f"${row.get('Mean PnL', 0):.2f}",
                f"${row['Std PnL']:.2f}" if 'Std PnL' in row else f"${row.get('Std PnL', 0):.2f}",
                f"{row['Sharpe']:.3f}" if 'Sharpe' in row else f"{row.get('Sharpe', 0):.3f}",
                f"{row['Win Rate %']:.1f}%" if 'Win Rate %' in row else f"{row.get('Win Rate %', 0):.1f}%"
            ])
        
        if table_data:
            table = ax.table(cellText=table_data,
                            colLabels=['Strategy', 'Mean PnL', 'Std PnL', 'Sharpe', 'Win Rate'],
                            cellLoc='center',
                            loc='center',
                            colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
        
        ax.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def create_combined_dashboard(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a combined dashboard with multiple subplots.
        
        Args:
            save_path: Path to save the figure
        
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(20, 16))
        
        # 1. PnL Distribution (top left)
        ax1 = plt.subplot(3, 3, 1)
        self._plot_pnl_distribution_on_ax(ax1)
        
        # 2. Sharpe Ratio Comparison (top middle)
        ax2 = plt.subplot(3, 3, 2)
        self._plot_sharpe_on_ax(ax2)
        
        # 3. VaR Comparison (top right)
        ax3 = plt.subplot(3, 3, 3)
        self._plot_var_on_ax(ax3)
        
        # 4. Cumulative PnL (middle left)
        ax4 = plt.subplot(3, 3, 4)
        self._plot_cumulative_on_ax(ax4)
        
        # 5. Risk-Return Scatter (middle middle)
        ax5 = plt.subplot(3, 3, 5)
        self._plot_risk_return_on_ax(ax5)
        
        # 6. Transaction Costs (middle right)
        ax6 = plt.subplot(3, 3, 6)
        self._plot_costs_on_ax(ax6)
        
        # 7. Drawdown (bottom left)
        ax7 = plt.subplot(3, 3, 7)
        self._plot_drawdown_on_ax(ax7)
        
        # 8. Cost Efficiency (bottom middle)
        ax8 = plt.subplot(3, 3, 8)
        self._plot_efficiency_on_ax(ax8)
        
        # 9. Summary Table (bottom right)
        ax9 = plt.subplot(3, 3, 9)
        self._plot_summary_table_on_ax(ax9)
        
        plt.suptitle('Deep Hedging Performance Dashboard', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        
        return fig
    
    def _plot_pnl_distribution_on_ax(self, ax):
        """
        Plot PnL distribution on given axis.
        """
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.results)))
        for i, (name, data) in enumerate(self.results.items()):
            if 'pnls' in data and data['pnls'] is not None:
                pnls = data['pnls']
                if len(pnls) > 0:
                    ax.hist(pnls, bins=30, alpha=0.5, label=name, color=colors[i], edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
        ax.set_xlabel('PnL ($)')
        ax.set_ylabel('Frequency')
        ax.set_title('PnL Distribution')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
    
    def _plot_sharpe_on_ax(self, ax):
        """
        Plot Sharpe ratio on given axis.
        """
        
        names = []
        sharpes = []
        for name, data in self.results.items():
            if 'pnls' in data and data['pnls'] is not None:
                pnls = data['pnls']
                if len(pnls) > 0:
                    sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252) if np.std(pnls) > 0 else 0
                    names.append(name)
                    sharpes.append(sharpe)
        
        if names:
            colors = plt.cm.RdYlGn(np.linspace(0, 1, len(names)))
            bars = ax.barh(names, sharpes, color=colors, edgecolor='black')
            ax.axvline(x=0, color='black', linestyle='-')
            ax.axvline(x=1, color='green', linestyle='--', alpha=0.5, label='Good')
            ax.axvline(x=2, color='blue', linestyle='--', alpha=0.5, label='Excellent')
            ax.set_xlabel('Sharpe Ratio')
            ax.set_title('Sharpe Ratio')
            ax.legend(fontsize=7)
            for bar, val in zip(bars, sharpes):
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{val:.2f}', va='center', fontsize=8)
    
    def _plot_var_on_ax(self, ax):
        """
        Plot VaR on given axis.
        """
        
        names = []
        vars_95 = []
        for name, data in self.results.items():
            if 'pnls' in data and data['pnls'] is not None:
                pnls = data['pnls']
                if len(pnls) > 0:
                    var = np.percentile(pnls, 5)
                    names.append(name)
                    vars_95.append(var)
        
        if names:
            colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in vars_95]
            bars = ax.barh(names, vars_95, color=colors, edgecolor='black')
            ax.axvline(x=0, color='black', linestyle='-')
            ax.set_xlabel('VaR (95%) ($)')
            ax.set_title('Value at Risk (95%)')
            for bar, val in zip(bars, vars_95):
                ax.text(bar.get_width() + (1 if val < 0 else -1), bar.get_y() + bar.get_height()/2,
                       f'${val:.0f}', va='center', ha='left' if val < 0 else 'right', fontsize=8)
    
    def _plot_cumulative_on_ax(self, ax):
        """
        Plot cumulative PnL on given axis.
        """
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.results)))
        for i, (name, data) in enumerate(self.results.items()):
            if 'daily_pnls' in data and data['daily_pnls'] is not None:
                daily_pnls = data['daily_pnls']
                if isinstance(daily_pnls, (list, np.ndarray)) and len(daily_pnls) > 0:
                    daily_pnls = np.array(daily_pnls)
                    if daily_pnls.ndim == 1:
                        cumulative = np.cumsum(daily_pnls)
                    else:
                        cumulative = np.cumsum(np.mean(daily_pnls, axis=0))
                    ax.plot(cumulative, label=name, color=colors[i], linewidth=1.5)
        ax.axhline(y=0, color='red', linestyle='--')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Cumulative PnL ($)')
        ax.set_title('Cumulative PnL')
        ax.legend(loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.3)
    
    def _plot_risk_return_on_ax(self, ax):
        """
        Plot risk-return scatter on given axis.
        """
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.results)))
        for i, (name, data) in enumerate(self.results.items()):
            if 'pnls' in data and data['pnls'] is not None:
                pnls = data['pnls']
                if len(pnls) > 0:
                    mean_ret = np.mean(pnls)
                    risk = np.std(pnls)
                    ax.scatter(risk, mean_ret, s=150, color=colors[i], edgecolor='black', linewidth=1.5)
                    ax.annotate(name, (risk, mean_ret), xytext=(8, 8), textcoords='offset points', fontsize=8)
        ax.axhline(y=0, color='gray', linestyle='--')
        ax.axvline(x=0, color='gray', linestyle='--')
        ax.set_xlabel('Risk (Std Dev)')
        ax.set_ylabel('Return (Mean PnL)')
        ax.set_title('Risk-Return Tradeoff')
        ax.grid(True, alpha=0.3)
    
    def _plot_costs_on_ax(self, ax):
        """
        Plot transaction costs on given axis.
        """
        
        names = []
        costs = []
        for name, data in self.results.items():
            if 'metrics' in data and 'total_transaction_costs' in data['metrics']:
                names.append(name)
                costs.append(data['metrics']['total_transaction_costs'])
        
        if names:
            colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
            bars = ax.barh(names, costs, color=colors, edgecolor='black')
            ax.set_xlabel('Transaction Costs ($)')
            ax.set_title('Total Transaction Costs')
            for bar, val in zip(bars, costs):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                       f'${val:.0f}', va='center', fontsize=8)
    
    def _plot_drawdown_on_ax(self, ax):
        """
        Plot drawdown on given axis.
        """
        
        names = []
        drawdowns = []
        for name, data in self.results.items():
            if 'portfolio_values' in data and data['portfolio_values'] is not None:
                values = data['portfolio_values']
                if len(values) > 0:
                    cummax = np.maximum.accumulate(values)
                    drawdown = (values - cummax) / cummax * 100
                    max_dd = np.min(drawdown)
                    names.append(name)
                    drawdowns.append(max_dd)
        
        if names:
            colors = ['#2ecc71' if d > -10 else '#e74c3c' for d in drawdowns]
            bars = ax.barh(names, drawdowns, color=colors, edgecolor='black')
            ax.axvline(x=0, color='black', linestyle='-')
            ax.set_xlabel('Max Drawdown (%)')
            ax.set_title('Maximum Drawdown')
            for bar, val in zip(bars, drawdowns):
                ax.text(bar.get_width() + (0.5 if val < 0 else -0.5), bar.get_y() + bar.get_height()/2,
                       f'{val:.1f}%', va='center', ha='left' if val < 0 else 'right', fontsize=8)
    
    def _plot_efficiency_on_ax(self, ax):
        """
        Plot cost efficiency on given axis.
        """
        
        names = []
        efficiencies = []
        for name, data in self.results.items():
            if 'pnls' in data and data['pnls'] is not None:
                pnl = np.mean(data['pnls'])
                if 'metrics' in data and 'total_transaction_costs' in data['metrics']:
                    cost = data['metrics']['total_transaction_costs']
                    efficiency = pnl / cost if cost > 0 else 0
                    names.append(name)
                    efficiencies.append(efficiency)
        
        if names:
            colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
            bars = ax.barh(names, efficiencies, color=colors, edgecolor='black')
            ax.set_xlabel('PnL per $1 Cost')
            ax.set_title('Cost Efficiency')
            for bar, val in zip(bars, efficiencies):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                       f'{val:.1f}x', va='center', fontsize=8)
    
    def _plot_summary_table_on_ax(self, ax):
        """
        Plot summary table on given axis.
        """
        
        ax.axis('tight')
        ax.axis('off')
        
        if self.comparison_df is None:
            summary_data = []
            for name, data in self.results.items():
                if 'pnls' in data and data['pnls'] is not None:
                    pnls = data['pnls']
                    if len(pnls) > 0:
                        summary_data.append([
                            name[:15],
                            f"${np.mean(pnls):.0f}",
                            f"{np.mean(pnls)/np.std(pnls)*np.sqrt(252):.2f}" if np.std(pnls) > 0 else "0",
                            f"{(pnls > 0).sum()/len(pnls)*100:.0f}%"
                        ])
        else:
            summary_data = self.comparison_df[['Strategy', 'Mean PnL', 'Sharpe', 'Win Rate %']].values.tolist()
            summary_data = [[str(s)[:15], f"${float(m):.0f}", f"{float(sh):.2f}", f"{float(w):.0f}%"] 
                           for s, m, sh, w in summary_data]
        
        if summary_data:
            table = ax.table(cellText=summary_data[:8],
                            colLabels=['Strategy', 'PnL', 'Sharpe', 'Win Rate'],
                            cellLoc='center',
                            loc='center',
                            colWidths=[0.35, 0.2, 0.2, 0.2])
            table.auto_set_font_size(False)
            table.set_fontsize(8)
        ax.set_title('Top Strategies', fontsize=10, fontweight='bold')
    
    def create_all(self, save_dir: str = "results/figures") -> Dict[str, str]:
        """
        Create all visualizations and dashboard.
        
        Args:
            save_dir: Directory to save figures
        
        Returns:
            Dictionary with file paths
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        file_paths = {}
        
        # Create individual plot sets
        print("\nCreating hedging performance plots...")
        try:
            perf_paths = self.performance_plotter.plot_all(save_dir)
            file_paths.update(perf_paths)
        except Exception as e:
            print(f"Performance plots error: {e}")
        
        print("\nCreating risk metrics plots...")
        try:
            risk_paths = self.risk_plotter.plot_all(save_dir)
            file_paths.update(risk_paths)
        except Exception as e:
            print(f"Risk metrics plots error: {e}")
        
        print("\nCreating transaction cost plots...")
        try:
            cost_paths = self.cost_plotter.plot_all(save_dir)
            file_paths.update(cost_paths)
        except Exception as e:
            print(f"Transaction cost plots error: {e}")
        
        # Create summary table
        file_paths['summary_table'] = os.path.join(save_dir, 'summary_table.png')
        self.create_summary_table(save_path=file_paths['summary_table'])
        
        # Create combined dashboard
        file_paths['combined_dashboard'] = os.path.join(save_dir, 'combined_dashboard.png')
        self.create_combined_dashboard(save_path=file_paths['combined_dashboard'])
        
        print(f"\nAll visualizations saved to {save_dir}")
        return file_paths


if __name__ == "__main__":
    print("="*70)
    print("DASHBOARD CREATOR TEST")
    print("="*70)
    
    # Create sample data
    np.random.seed(42)
    sample_results = {
        'Black-Scholes': {
            'pnls': np.random.normal(50, 100, 200),
            'daily_pnls': np.random.randn(252, 10) * 10,
            'portfolio_values': 10000 + np.cumsum(np.random.normal(50, 100, 200)),
            'metrics': {'total_transaction_costs': 25.0}
        },
        'Delta-Gamma': {
            'pnls': np.random.normal(60, 90, 200),
            'daily_pnls': np.random.randn(252, 10) * 9,
            'portfolio_values': 10000 + np.cumsum(np.random.normal(60, 90, 200)),
            'metrics': {'total_transaction_costs': 35.0}
        },
        'DRL_PPO': {
            'pnls': np.random.normal(80, 85, 200),
            'daily_pnls': np.random.randn(252, 10) * 8,
            'portfolio_values': 10000 + np.cumsum(np.random.normal(80, 85, 200)),
            'metrics': {'total_transaction_costs': 45.0}
        },
        'DRL_SAC': {
            'pnls': np.random.normal(70, 95, 200),
            'daily_pnls': np.random.randn(252, 10) * 9.5,
            'portfolio_values': 10000 + np.cumsum(np.random.normal(70, 95, 200)),
            'metrics': {'total_transaction_costs': 40.0}
        }
    }
    
    # Create comparison DataFrame
    sample_df = pd.DataFrame([
        {'Strategy': 'Black-Scholes', 'Mean PnL': 50, 'Sharpe': 0.85, 'Win Rate %': 52},
        {'Strategy': 'Delta-Gamma', 'Mean PnL': 60, 'Sharpe': 0.95, 'Win Rate %': 54},
        {'Strategy': 'DRL_PPO', 'Mean PnL': 80, 'Sharpe': 1.12, 'Win Rate %': 56},
        {'Strategy': 'DRL_SAC', 'Mean PnL': 70, 'Sharpe': 1.05, 'Win Rate %': 55}
    ])
    
    # Create dashboard
    dashboard = DashboardCreator(sample_results, sample_df)
    dashboard.create_all(save_dir="test_plots")
