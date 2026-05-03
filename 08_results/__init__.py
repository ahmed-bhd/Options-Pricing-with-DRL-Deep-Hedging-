# -*- coding: utf-8 -*-
"""
================================================================================
08_RESULTS MODULE INITIALIZER
================================================================================

This module provides tools for storing and analyzing backtest results.

Exports:
    - ResultsManager: Save/load results to CSV files
    - ResultsAnalyzer: Analyze stored results and generate statistics

Usage:
    from src.results import ResultsManager, ResultsAnalyzer
    
    manager = ResultsManager(output_dir="08_results")
    manager.save_results(backtest_results)
    
    analyzer = ResultsAnalyzer()
    summary = analyzer.get_summary()
================================================================================
"""

from .results_manager import ResultsManager
from .results_analyzer import ResultsAnalyzer

__all__ = [
    'ResultsManager',
    'ResultsAnalyzer'
]

__version__ = '1.0.0'
__author__ = 'Ahmed Baahmed'