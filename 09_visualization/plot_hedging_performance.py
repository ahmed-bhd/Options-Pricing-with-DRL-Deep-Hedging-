# -*- coding: utf-8 -*-
"""
================================================================================
HEDGING PERFORMANCE PLOTS
================================================================================

This module creates visualizations for hedging performance:
    - PnL distribution histograms
    - Box plots comparing strategies
    - Cumulative PnL over time
    - Probability density plots
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

sns.set_palette("husl")


class HedgingPerformancePlotter:
    """
    Plotter for hedging performance visualizations.
    
    Creates:
        - PnL distribution histograms
        - Box plots comparing strategies
        - Cumulative PnL over time
        - Probability density plots
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
    
    def plot_pnl_distribution(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot PnL distribution histograms for all strategies.
        
        Args:
            save_path: Path to save the figure
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for i, (name, data) in enumerate(self.results.items()):
            if 'pnls' in data and data['pnls'] is not None:
                pnls = data['pnls']
                if len(pnls) > 0:
                    ax.hist(pnls, bins=30, alpha=0.5, label=name, color=self.colors[i], edgecolor='black')
        
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Break-even')
        ax.set_xlabel('PnL ($)')
        ax.set_ylabel('Frequency')
        ax.set_title('PnL Distribution by Strategy')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_pnl_boxplot(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot box plot comparing PnL distributions.
        
        Args:
            save_path: Path to save the figure
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        data_to_plot = []
        labels = []
        
        for name, data in self.results.items():
            if 'pnls' in data and data['pnls'] is not None:
                pnls = data['pnls']
                if len(pnls) > 0:
                    data_to_plot.append(pnls)
                    labels.append(name)
        
        if not data_to_plot:
            print("No data to plot")
            return fig
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        # Color the boxes
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(self.colors[i % len(self.colors)])
            box.set_alpha(0.7)
        
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Break-even')
        ax.set_ylabel('PnL ($)')
        ax.set_title('PnL Distribution Box Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_cumulative_pnl(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot cumulative PnL over time (if time series data available).
        
        Args:
            save_path: Path to save the figure
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        has_data = False
        
        for i, (name, data) in enumerate(self.results.items()):
            if 'daily_pnls' in data and data['daily_pnls'] is not None:
                daily_pnls = data['daily_pnls']
                if isinstance(daily_pnls, (list, np.ndarray)) and len(daily_pnls) > 0:
                    # Convert to numpy array
                    daily_pnls = np.array(daily_pnls)
                    # Ensure we have a 2D array
                    if daily_pnls.ndim == 1:
                        cumulative = np.cumsum(daily_pnls)
                    else:
                        mean_daily = np.mean(daily_pnls, axis=0)
                        cumulative = np.cumsum(mean_daily)
                    
                    ax.plot(cumulative, label=name, color=self.colors[i], linewidth=2)
                    has_data = True
        
        if not has_data:
            ax.text(0.5, 0.5, 'No daily PnL data available', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
        
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Cumulative PnL ($)')
        ax.set_title('Average Cumulative PnL Over Time')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_violin(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot violin plot for PnL distributions.
        
        Args:
            save_path: Path to save the figure
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        data_to_plot = []
        labels = []
        
        for name, data in self.results.items():
            if 'pnls' in data and data['pnls'] is not None:
                pnls = data['pnls']
                if len(pnls) > 0:
                    data_to_plot.append(pnls)
                    labels.append(name)
        
        if not data_to_plot:
            print("No data to plot")
            return fig
        
        # Create violin plot
        parts = ax.violinplot(data_to_plot, positions=range(1, len(data_to_plot) + 1),
                              showmeans=True, showmedians=True)
        
        # Color the violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(self.colors[i % len(self.colors)])
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Break-even')
        ax.set_ylabel('PnL ($)')
        ax.set_title('PnL Distribution Violin Plot')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_all(self, save_dir: str = "results/figures") -> Dict[str, str]:
        """
        Create all hedging performance plots.
        
        Args:
            save_dir: Directory to save figures
        
        Returns:
            Dictionary with file paths
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        file_paths = {}
        
        # PnL distribution
        file_paths['pnl_distribution'] = os.path.join(save_dir, 'pnl_distribution.png')
        self.plot_pnl_distribution(save_path=file_paths['pnl_distribution'])
        
        # Box plot
        file_paths['pnl_boxplot'] = os.path.join(save_dir, 'pnl_boxplot.png')
        self.plot_pnl_boxplot(save_path=file_paths['pnl_boxplot'])
        
        # Cumulative PnL
        file_paths['cumulative_pnl'] = os.path.join(save_dir, 'cumulative_pnl.png')
        self.plot_cumulative_pnl(save_path=file_paths['cumulative_pnl'])
        
        # Violin plot
        file_paths['violin_plot'] = os.path.join(save_dir, 'violin_plot.png')
        self.plot_violin(save_path=file_paths['violin_plot'])
        
        print(f"\nHedging performance plots saved to {save_dir}")
        return file_paths


if __name__ == "__main__":
    print("="*70)
    print("HEDGING PERFORMANCE PLOTTER TEST")
    print("="*70)
    
    # Create sample data
    np.random.seed(42)
    sample_results = {
        'Black-Scholes': {'pnls': np.random.normal(50, 100, 200)},
        'Delta-Gamma': {'pnls': np.random.normal(60, 90, 200)},
        'DRL_PPO': {'pnls': np.random.normal(80, 85, 200)},
        'DRL_SAC': {'pnls': np.random.normal(70, 95, 200)}
    }
    
    # Add daily PnL for cumulative plot
    for name in sample_results:
        n_steps = 252
        daily = np.random.randn(n_steps, 10) * 10
        sample_results[name]['daily_pnls'] = daily
    
    plotter = HedgingPerformancePlotter(sample_results)
    plotter.plot_all(save_dir="test_plots")