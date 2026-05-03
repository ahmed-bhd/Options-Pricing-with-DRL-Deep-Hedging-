# -*- coding: utf-8 -*-
"""
================================================================================
04_BASELINES MODULE INITIALIZER
================================================================================

This module provides classical hedging baselines for comparison with DRL agents.

Exports:
    - BlackScholesHedge: Standard delta hedging
    - DeltaGammaHedge: Second-order hedging
    - StaticHedge: Buy-and-hold option (no hedge)

Usage:
    from hedging_baselines import BlackScholesHedge, DeltaGammaHedge, StaticHedge
================================================================================
"""

from .black_scholes_hedge import BlackScholesHedge
from .delta_gamma_hedge import DeltaGammaHedge
from .static_hedge import StaticHedge

__all__ = [
    'BlackScholesHedge',
    'DeltaGammaHedge',
    'StaticHedge'
]

__version__ = '1.0.0'
__author__ = 'Ahmed Baahmed'