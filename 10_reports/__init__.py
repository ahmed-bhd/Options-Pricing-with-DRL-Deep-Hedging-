# -*- coding: utf-8 -*-
"""
================================================================================
10_REPORTS MODULE INITIALIZER
================================================================================

This module provides tools for generating analysis reports from backtest results.

Exports:
    - ReportGenerator: Generate comprehensive analysis reports (MD, HTML, PDF)

Usage:
    from src.reports import ReportGenerator
    
    generator = ReportGenerator(results, comparison_df)
    generator.generate_markdown("analysis_report.md")
    generator.generate_html("analysis_report.html")
================================================================================
"""

from .report_generator import ReportGenerator

__all__ = [
    'ReportGenerator'
]

__version__ = '1.0.0'
__author__ = 'Ahmed Baahmed'