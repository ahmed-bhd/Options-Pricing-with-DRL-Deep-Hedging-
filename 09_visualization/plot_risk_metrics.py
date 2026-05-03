# -*- coding: utf-8 -*-
"""
================================================================================
RISK METRICS PLOTS
================================================================================

This module creates visualizations for risk metrics:
    - VaR (Value at Risk) comparison
    - CVaR (Conditional Value at Risk) comparison
    - Drawdown plots
    - Risk-return scatter plots
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
try:
    plt.style.use('seaborn-v0-8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')


class RiskMetricsPlotter:
    """
    Plotter for risk metrics visualizations.
    
    Creates:
        - VaR (Value at Risk) comparison bar chart
        - CVaR (Conditional Value at Risk) comparison
        - Drawdown plots
        - Risk-return scatter plots
    """
    
    def __init__(self, results: Dict[str, Dict], figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize plotter.
        
        Args:
            results: Dictionary of backtest results from BacktestEngine
            figsize: Figure size (width, height)
        """
        self.results = results
        self.figsize = figsize
        self.colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    
    def plot_var_comparison(self, confidence: float = 0.95, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot VaR comparison bar chart.
        
        Args:
            confidence: Confidence level for VaR (default: 0.95)
            save_path: Path to save the figure
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        names = []
        var_values = []
        
        for name, data in self.results.items():
            if 'pnls' in data and data['pnls'] is not None:
                pnls = data['pnls']
                if len(pnls) > 0:
                    var = np.percentile(pnls, (1 - confidence) * 100)
                    names.append(name)
                    var_values.append(var)
        
        if not names:
            ax.text(0.5, 0.5, 'No VaR data available', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
            return fig
        
        # Sort by VaR (more negative = worse)
        sorted_idx = np.argsort(var_values)
        names = [names[i] for i in sorted_idx]
        var_values = [var_values[i] for i in sorted_idx]
        
        colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in var_values]
        bars = ax.barh(names, var_values, color=colors, edgecolor='black')
        
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel(f'VaR ({confidence*100:.0f}%) ($)')
        ax.set_title(f'Value at Risk (VaR) Comparison at {confidence*100:.0f}% Confidence')
        
        # Add value labels
        for bar, val in zip(bars, var_values):
            ax.text(val + (1 if val < 0 else -1), bar.get_y() + bar.get_height()/2,
                   f'${val:.0f}', va='center', ha='left' if val < 0 else 'right', fontsize=9)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_cvar_comparison(self, confidence: float = 0.95, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot CVaR comparison bar chart.
        
        Args:
            confidence: Confidence level for CVaR (default: 0.95)
            save_path: Path to save the figure
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        names = []
        cvar_values = []
        
        for name, data in self.results.items():
            if 'pnls' in data and data['pnls'] is not None:
                pnls = data['pnls']
                if len(pnls) > 0:
                    var = np.percentile(pnls, (1 - confidence) * 100)
                    losses_beyond_var = pnls[pnls <= var]
                    cvar = np.mean(losses_beyond_var) if len(losses_beyond_var) > 0 else var
                    names.append(name)
                    cvar_values.append(cvar)
        
        if not names:
            ax.text(0.5, 0.5, 'No CVaR data available', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
            return fig
        
        # Sort by CVaR (more negative = worse)
        sorted_idx = np.argsort(cvar_values)
        names = [names[i] for i in sorted_idx]
        cvar_values = [cvar_values[i] for i in sorted_idx]
        
        colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in cvar_values]
        bars = ax.barh(names, cvar_values, color=colors, edgecolor='black')
        
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel(f'CVaR ({confidence*100:.0f}%) ($)')
        ax.set_title(f'Conditional Value at Risk (CVaR) Comparison at {confidence*100:.0f}% Confidence')
        
        # Add value labels
        for bar, val in zip(bars, cvar_values):
            ax.text(val + (1 if val < 0 else -1), bar.get_y() + bar.get_height()/2,
                   f'${val:.0f}', va='center', ha='left' if val < 0 else 'right', fontsize=9)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_risk_return_scatter(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot risk-return scatter plot.
        
        Args:
            save_path: Path to save the figure
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        has_data = False
        
        for i, (name, data) in enumerate(self.results.items()):
            if 'pnls' in data and data['pnls'] is not None:
                pnls = data['pnls']
                if len(pnls) > 0:
                    mean_return = np.mean(pnls)
                    risk = np.std(pnls)
                    
                    ax.scatter(risk, mean_return, s=200, color=self.colors[i], 
                              edgecolor='black', linewidth=1.5, alpha=0.8)
                    ax.annotate(name, (risk, mean_return), xytext=(10, 10),
                               textcoords='offset points', fontsize=10, fontweight='bold')
                    has_data = True
        
        if not has_data:
            ax.text(0.5, 0.5, 'No risk-return data available', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
        
        # Add quadrant lines
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Risk (Std Dev of PnL)')
        ax.set_ylabel('Expected Return (Mean PnL)')
        ax.set_title('Risk-Return Tradeoff')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_sharpe_ratio(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Sharpe ratio comparison bar chart.
        
        Args:
            save_path: Path to save the figure
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        names = []
        sharpe_values = []
        
        for name, data in self.results.items():
            if 'pnls' in data and data['pnls'] is not None:
                pnls = data['pnls']
                if len(pnls) > 0:
                    sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252) if np.std(pnls) > 0 else 0
                    names.append(name)
                    sharpe_values.append(sharpe)
        
        if not names:
            ax.text(0.5, 0.5, 'No Sharpe ratio data available', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
            return fig
        
        # Sort by Sharpe
        sorted_idx = np.argsort(sharpe_values)
        names = [names[i] for i in sorted_idx]
        sharpe_values = [sharpe_values[i] for i in sorted_idx]
        
        # Color gradient from red to green
        norm = plt.Normalize(min(sharpe_values), max(sharpe_values))
        colors = plt.cm.RdYlGn(norm(sharpe_values))
        
        bars = ax.barh(names, sharpe_values, color=colors, edgecolor='black')
        
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(x=1, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='Good (1.0)')
        ax.axvline(x=2, color='blue', linestyle='--', alpha=0.5, linewidth=1.5, label='Excellent (2.0)')
        ax.set_xlabel('Sharpe Ratio (Annualized)')
        ax.set_title('Sharpe Ratio Comparison')
        ax.legend()
        
        # Add value labels
        for bar, val in zip(bars, sharpe_values):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{val:.2f}', va='center', fontsize=9)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_drawdown(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot maximum drawdown comparison.
        
        Args:
            save_path: Path to save the figure
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        names = []
        drawdowns = []
        
        for name, data in self.results.items():
            if 'portfolio_values' in data and data['portfolio_values'] is not None:
                values = data['portfolio_values']
                if len(values) > 0:
                    cummax = np.maximum.accumulate(values)
                    drawdown = (values - cummax) / cummax
                    max_dd = np.min(drawdown) * 100
                    names.append(name)
                    drawdowns.append(max_dd)
            elif 'pnls' in data and data['pnls'] is not None:
                # Estimate drawdown from PnL
                pnls = data['pnls']
                if len(pnls) > 0:
                    cumulative = np.cumsum(pnls)
                    cummax = np.maximum.accumulate(cumulative)
                    drawdown = (cumulative - cummax) / cummax
                    max_dd = np.min(drawdown) * 100 if len(drawdown) > 0 else 0
                    names.append(name)
                    drawdowns.append(max_dd)
        
        if not names:
            ax.text(0.5, 0.5, 'No drawdown data available', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
            return fig
        
        # Sort by drawdown (less negative = better)
        sorted_idx = np.argsort(drawdowns)[::-1]  # Reverse for better visualization
        names = [names[i] for i in sorted_idx]
        drawdowns = [drawdowns[i] for i in sorted_idx]
        
        colors = ['#2ecc71' if d > -10 else '#e74c3c' for d in drawdowns]
        bars = ax.barh(names, drawdowns, color=colors, edgecolor='black')
        
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Maximum Drawdown (%)')
        ax.set_title('Maximum Drawdown Comparison')
        
        # Add value labels
        for bar, val in zip(bars, drawdowns):
            ax.text(val + (0.5 if val < 0 else -0.5), bar.get_y() + bar.get_height()/2,
                   f'{val:.1f}%', va='center', ha='left' if val < 0 else 'right', fontsize=9)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_all(self, save_dir: str = "results/figures") -> Dict[str, str]:
        """
        Create all risk metrics plots.
        
        Args:
            save_dir: Directory to save figures
        
        Returns:
            Dictionary with file paths
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        file_paths = {}
        
        # VaR comparison
        file_paths['var_comparison'] = os.path.join(save_dir, 'var_comparison.png')
        self.plot_var_comparison(save_path=file_paths['var_comparison'])
        
        # CVaR comparison
        file_paths['cvar_comparison'] = os.path.join(save_dir, 'cvar_comparison.png')
        self.plot_cvar_comparison(save_path=file_paths['cvar_comparison'])
        
        # Risk-return scatter
        file_paths['risk_return_scatter'] = os.path.join(save_dir, 'risk_return_scatter.png')
        self.plot_risk_return_scatter(save_path=file_paths['risk_return_scatter'])
        
        # Sharpe ratio
        file_paths['sharpe_ratio'] = os.path.join(save_dir, 'sharpe_ratio.png')
        self.plot_sharpe_ratio(save_path=file_paths['sharpe_ratio'])
        
        # Drawdown
        file_paths['drawdown'] = os.path.join(save_dir, 'drawdown_comparison.png')
        self.plot_drawdown(save_path=file_paths['drawdown'])
        
        print(f"\nRisk metrics plots saved to {save_dir}")
        return file_paths


if __name__ == "__main__":
    print("="*70)
    print("RISK METRICS PLOTTER TEST")
    print("="*70)
    
    # Create sample data
    np.random.seed(42)
    sample_results = {
        'Black-Scholes': {
            'pnls': np.random.normal(50, 100, 200),
            'portfolio_values': 10000 + np.cumsum(np.random.normal(50, 100, 200))
        },
        'Delta-Gamma': {
            'pnls': np.random.normal(60, 90, 200),
            'portfolio_values': 10000 + np.cumsum(np.random.normal(60, 90, 200))
        },
        'DRL_PPO': {
            'pnls': np.random.normal(80, 85, 200),
            'portfolio_values': 10000 + np.cumsum(np.random.normal(80, 85, 200))
        },
        'DRL_SAC': {
            'pnls': np.random.normal(70, 95, 200),
            'portfolio_values': 10000 + np.cumsum(np.random.normal(70, 95, 200))
        }
    }
    
    plotter = RiskMetricsPlotter(sample_results)
    plotter.plot_all(save_dir="test_plots")
