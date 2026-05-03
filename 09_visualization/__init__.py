# -*- coding: utf-8 -*-
"""
================================================================================
09_VISUALIZATION MODULE INITIALIZER
================================================================================

This module provides visualization tools for hedging results.

Exports:
    - HedgingPerformancePlotter: PnL distributions, histograms, box plots
    - RiskMetricsPlotter: VaR, CVaR, drawdown plots
    - TransactionCostPlotter: Cost analysis, turnover plots
    - DashboardCreator: Complete dashboard with all plots

Usage:
    from src.visualization import DashboardCreator
    
    dashboard = DashboardCreator(results, comparison_df)
    dashboard.create_all(save_dir="results/figures")
================================================================================
"""

from .plot_hedging_performance import HedgingPerformancePlotter
from .plot_risk_metrics import RiskMetricsPlotter
from .plot_transaction_costs import TransactionCostPlotter
from .create_dashboard import DashboardCreator

__all__ = [
    'HedgingPerformancePlotter',
    'RiskMetricsPlotter',
    'TransactionCostPlotter',
    'DashboardCreator'
]

__version__ = '1.0.0'
__author__ = 'Ahmed Baahmed'