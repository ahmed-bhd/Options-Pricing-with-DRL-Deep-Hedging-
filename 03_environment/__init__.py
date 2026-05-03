# -*- coding: utf-8 -*-
"""
================================================================================
03_ENVIRONMENT MODULE INITIALIZER
================================================================================

This module provides the Gymnasium environment for option hedging using
Deep Reinforcement Learning (Deep Hedging).

Exports:
    - OptionHedgingEnv: Main Gymnasium environment for hedging European options
    - RiskMetrics: VaR, CVaR, and other risk calculations
    - calculate_hedging_pnl: Utility function for PnL calculation

Usage:
    from src.environment import OptionHedgingEnv, RiskMetrics
    
    env = OptionHedgingEnv(price_paths, strike=100, transaction_cost=0.001)
    obs, info = env.reset()
    action = agent.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
================================================================================
"""

from .option_hedging_env import OptionHedgingEnv
from .risk_metrics import RiskMetrics, calculate_hedging_pnl

__all__ = [
    'OptionHedgingEnv',
    'RiskMetrics',
    'calculate_hedging_pnl'
]

__version__ = '1.0.0'
__author__ = 'Ahmed Baahmed'
__doc__ = "Option hedging environment for deep hedging"