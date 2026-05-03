# -*- coding: utf-8 -*-
"""
@author: Ahmed Baahmed


================================================================================
02_DATA MODULE INITIALIZER
================================================================================

This module provides price simulators and data loaders for deep hedging.

Exports:
    - GeometricBrownianMotion: Standard GBM simulator (with real data calibration)
    - HestonModel: Stochastic volatility simulator (with real data calibration)
    - JumpDiffusionModel: Jump-diffusion simulator (with real data calibration)
    - ParameterCalibrator: Calibrates models from real market data
    - MarketDataLoader: Load real market data
    - create_simulator_from_real_data: Factory function for calibrated simulators

Usage:
    # Using real data calibration
    from src.data import create_simulator_from_real_data
    gbm = create_simulator_from_real_data('gbm', '2006-01-01', '2023-12-31')
    paths = gbm.simulate(steps=252, n_paths=10000)
    
    # Or manual parameters
    from src.data import GeometricBrownianMotion
    gbm = GeometricBrownianMotion(S0=100, mu=0.05, sigma=0.2)
================================================================================
"""

from .price_simulators import (
    GeometricBrownianMotion,
    HestonModel,
    JumpDiffusionModel,
    PriceSimulatorBase,
    ParameterCalibrator,
    create_simulator_from_real_data
)

from .data_loader import MarketDataLoader

from .options_simulator import OptionsDataSimulator

__all__ = [
    # Price Simulators
    'GeometricBrownianMotion',
    'HestonModel',
    'JumpDiffusionModel',
    'PriceSimulatorBase',
    'ParameterCalibrator',
    'create_simulator_from_real_data',
    # Data Loaders
    'MarketDataLoader',
    'OptionsDataSimulator'
]

__version__ = '1.0.0'
__author__ = 'Ahmed Baahmed'