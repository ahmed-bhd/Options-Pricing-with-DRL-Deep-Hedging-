# -*- coding: utf-8 -*-
"""
================================================================================
TRANSACTION COSTS PLOTS
================================================================================

This module creates visualizations for transaction cost analysis:
    - Total transaction costs by strategy (bar chart)
    - Cost per trade distribution (histogram)
    - Turnover comparison (bar chart)
    - Cost vs PnL scatter plot
    - Cost efficiency ratio
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


class TransactionCostPlotter:
    """
    Plotter for transaction cost visualizations.
    
    Creates:
        - Total transaction costs bar chart
        - Cost per trade distribution
        - Turnover comparison
        - Cost vs PnL scatter plot
        - Cost efficiency ratio
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
    
    def plot_total_costs(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot total transaction costs by strategy.
        
        Args:
            save_path: Path to save the figure
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        names = []
        costs = []
        
        for name, data in self.results.items():
            if 'metrics' in data and 'total_transaction_costs' in data['metrics']:
                names.append(name)
                costs.append(data['metrics']['total_transaction_costs'])
            elif 'pnls' in data:
                # Estimate costs if not available (using 10% of std as placeholder)
                names.append(name)
                costs.append(np.std(data['pnls']) * 0.1 if len(data['pnls']) > 0 else 0)
        
        if not names:
            ax.text(0.5, 0.5, 'No transaction cost data available', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
            return fig
        
        # Sort by cost
        sorted_idx = np.argsort(costs)
        names = [names[i] for i in sorted_idx]
        costs = [costs[i] for i in sorted_idx]
        
        bars = ax.barh(names, costs, color=self.colors[:len(names)], edgecolor='black')
        ax.set_xlabel('Total Transaction Costs ($)')
        ax.set_title('Total Transaction Costs by Strategy')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, val in zip(bars, costs):
            ax.text(val + (max(costs) * 0.01), bar.get_y() + bar.get_height()/2,
                   f'${val:.0f}', va='center', fontsize=9)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_turnover(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot turnover (trading volume) comparison.
        
        Args:
            save_path: Path to save the figure
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        names = []
        turnovers = []
        
        for name, data in self.results.items():
            if 'metrics' in data and 'turnover' in data['metrics']:
                names.append(name)
                turnovers.append(data['metrics']['turnover'])
            elif 'actions' in data:
                # Estimate turnover from actions
                actions = data['actions']
                if len(actions) > 0:
                    turnover = np.sum(np.abs(np.diff(actions))) if len(actions) > 1 else 0
                    names.append(name)
                    turnovers.append(turnover)
                else:
                    names.append(name)
                    turnovers.append(0)
        
        if not names:
            ax.text(0.5, 0.5, 'No turnover data available', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
            return fig
        
        # Sort by turnover
        sorted_idx = np.argsort(turnovers)
        names = [names[i] for i in sorted_idx]
        turnovers = [turnovers[i] for i in sorted_idx]
        
        bars = ax.barh(names, turnovers, color=self.colors[:len(names)], edgecolor='black')
        ax.set_xlabel('Turnover (Total Position Changes)')
        ax.set_title('Trading Turnover by Strategy')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, val in zip(bars, turnovers):
            ax.text(val + (max(turnovers) * 0.01), bar.get_y() + bar.get_height()/2,
                   f'{val:.0f}', va='center', fontsize=9)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_cost_distribution(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution of per-trade costs.
        
        Args:
            save_path: Path to save the figure
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        has_data = False
        
        for i, (name, data) in enumerate(self.results.items()):
            if 'transaction_costs' in data and data['transaction_costs'] is not None:
                costs = data['transaction_costs']
                if len(costs) > 0:
                    ax.hist(costs, bins=30, alpha=0.5, label=name, color=self.colors[i], edgecolor='black')
                    has_data = True
        
        if not has_data:
            ax.text(0.5, 0.5, 'No per-trade cost data available', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
            return fig
        
        ax.set_xlabel('Transaction Cost per Trade ($)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Per-Trade Transaction Costs')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_cost_vs_pnl(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot transaction costs vs PnL scatter plot.
        
        Args:
            save_path: Path to save the figure
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        has_data = False
        
        for i, (name, data) in enumerate(self.results.items()):
            if 'pnls' in data and data['pnls'] is not None:
                pnl = np.mean(data['pnls'])
                
                if 'metrics' in data and 'total_transaction_costs' in data['metrics']:
                    cost = data['metrics']['total_transaction_costs']
                else:
                    cost = np.std(data['pnls']) * 0.1 if len(data['pnls']) > 0 else 0
                
                ax.scatter(cost, pnl, s=200, color=self.colors[i], 
                          edgecolor='black', linewidth=1.5, alpha=0.8)
                ax.annotate(name, (cost, pnl), xytext=(10, 10),
                           textcoords='offset points', fontsize=10, fontweight='bold')
                has_data = True
        
        if not has_data:
            ax.text(0.5, 0.5, 'No cost vs PnL data available', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
            return fig
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Total Transaction Costs ($)')
        ax.set_ylabel('Mean PnL ($)')
        ax.set_title('Transaction Costs vs Performance')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_cost_efficiency(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot cost efficiency (PnL per $1 of transaction cost).
        
        Args:
            save_path: Path to save the figure
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        names = []
        efficiencies = []
        
        for name, data in self.results.items():
            if 'pnls' in data and data['pnls'] is not None:
                pnl = np.mean(data['pnls'])
                
                if 'metrics' in data and 'total_transaction_costs' in data['metrics']:
                    cost = data['metrics']['total_transaction_costs']
                    efficiency = pnl / cost if cost > 0 else (np.inf if pnl > 0 else 0)
                else:
                    efficiency = 0
                
                names.append(name)
                efficiencies.append(efficiency)
        
        if not names:
            ax.text(0.5, 0.5, 'No cost efficiency data available', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
            return fig
        
        # Sort by efficiency (descending)
        sorted_idx = np.argsort(efficiencies)[::-1]
        names = [names[i] for i in sorted_idx]
        efficiencies = [efficiencies[i] for i in sorted_idx]
        
        # Handle infinite values
        display_values = []
        for val in efficiencies:
            if val == np.inf:
                display_values.append(100)  # Cap at 100 for display
            else:
                display_values.append(val)
        
        bars = ax.barh(names, display_values, color=self.colors[:len(names)], edgecolor='black')
        ax.set_xlabel('PnL per $1 Transaction Cost')
        ax.set_title('Cost Efficiency (PnL / Transaction Cost)')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, val in zip(bars, efficiencies):
            if val == np.inf:
                label = '∞'
            else:
                label = f'{val:.1f}x'
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                   label, va='center', fontsize=9)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_all(self, save_dir: str = "results/figures") -> Dict[str, str]:
        """
        Create all transaction cost plots.
        
        Args:
            save_dir: Directory to save figures
        
        Returns:
            Dictionary with file paths
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        file_paths = {}
        
        # Total costs
        file_paths['total_costs'] = os.path.join(save_dir, 'total_transaction_costs.png')
        self.plot_total_costs(save_path=file_paths['total_costs'])
        
        # Turnover
        file_paths['turnover'] = os.path.join(save_dir, 'turnover_comparison.png')
        self.plot_turnover(save_path=file_paths['turnover'])
        
        # Cost distribution
        file_paths['cost_distribution'] = os.path.join(save_dir, 'cost_distribution.png')
        self.plot_cost_distribution(save_path=file_paths['cost_distribution'])
        
        # Cost vs PnL
        file_paths['cost_vs_pnl'] = os.path.join(save_dir, 'cost_vs_pnl.png')
        self.plot_cost_vs_pnl(save_path=file_paths['cost_vs_pnl'])
        
        # Cost efficiency
        file_paths['cost_efficiency'] = os.path.join(save_dir, 'cost_efficiency.png')
        self.plot_cost_efficiency(save_path=file_paths['cost_efficiency'])
        
        print(f"\nTransaction cost plots saved to {save_dir}")
        return file_paths


if __name__ == "__main__":
    print("="*70)
    print("TRANSACTION COST PLOTTER TEST")
    print("="*70)
    
    # Create sample data
    np.random.seed(42)
    sample_results = {
        'Black-Scholes': {
            'pnls': np.random.normal(50, 100, 200),
            'metrics': {'total_transaction_costs': 25.0, 'turnover': 150},
            'transaction_costs': np.random.exponential(2, 100)
        },
        'Delta-Gamma': {
            'pnls': np.random.normal(60, 90, 200),
            'metrics': {'total_transaction_costs': 35.0, 'turnover': 200},
            'transaction_costs': np.random.exponential(3, 100)
        },
        'DRL_PPO': {
            'pnls': np.random.normal(80, 85, 200),
            'metrics': {'total_transaction_costs': 45.0, 'turnover': 280},
            'transaction_costs': np.random.exponential(4, 100)
        },
        'DRL_SAC': {
            'pnls': np.random.normal(70, 95, 200),
            'metrics': {'total_transaction_costs': 40.0, 'turnover': 250},
            'transaction_costs': np.random.exponential(3.5, 100)
        }
    }
    
    plotter = TransactionCostPlotter(sample_results)
    plotter.plot_all(save_dir="test_plots")
