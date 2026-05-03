# -*- coding: utf-8 -*-
"""
================================================================================
07_EVALUATION MODULE INITIALIZER
================================================================================

This module provides tools for evaluating and comparing hedging strategies.

Exports:
    - BacktestEngine: Run backtests for all strategies
    - MetricsCalculator: Calculate performance metrics
    - StatisticalTests: Significance testing (t-tests, p-values)

Usage:
    from src.evaluation import BacktestEngine, MetricsCalculator, StatisticalTests
    
    engine = BacktestEngine(test_paths, baselines, drl_agents)
    results = engine.run_all()
    metrics = MetricsCalculator.calculate(results)
    significance = StatisticalTests.compare(agent1, agent2)
================================================================================
"""

from .backtest import BacktestEngine
from .metrics import MetricsCalculator
from .statistical_tests import StatisticalTests

__all__ = [
    'BacktestEngine',
    'MetricsCalculator',
    'StatisticalTests'
]

__version__ = '1.0.0'
__author__ = 'Ahmed Baahmed'