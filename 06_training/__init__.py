# -*- coding: utf-8 -*-
"""
================================================================================
06_TRAINING MODULE INITIALIZER
================================================================================

This module provides training and validation pipelines for DRL agents.

Exports:
    - HedgingTrainer: Main training class with data splitting
    - HedgingValidator: Out-of-sample validation (walk-forward, rolling window)
    - HyperparameterTuner: Grid search, random search, and Optuna tuning

Usage:
    from src.training import HedgingTrainer, HedgingValidator, HyperparameterTuner
    
    trainer = HedgingTrainer(price_paths, train_ratio=0.6)
    agent, metrics = trainer.train_agent(PPOAgent, "PPO")
    results = trainer.evaluate_agent(agent, "PPO")
================================================================================
"""

from .trainer import HedgingTrainer
from .validator import HedgingValidator
from .hyperparameter_tuning import HyperparameterTuner

__all__ = [
    'HedgingTrainer',
    'HedgingValidator',
    'HyperparameterTuner'
]

__version__ = '1.0.0'
__author__ = 'Ahmed Baahmed'