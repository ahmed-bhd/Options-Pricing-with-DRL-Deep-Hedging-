# -*- coding: utf-8 -*-
"""
================================================================================
PRICE SIMULATORS FOR DEEP HEDGING
================================================================================

This module implements various stochastic processes for simulating
underlying asset paths.

Available Models:
    - Geometric Brownian Motion (GBM): Standard lognormal model
    - Heston Model: Stochastic volatility
    - Jump-Diffusion Model: Rare large movements
================================================================================
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import sys
import os


# =============================================================================
# BASE CLASS
# =============================================================================

class PriceSimulatorBase(ABC):
    """
    Abstract base class for all price simulators.
    """
    
    def __init__(self, S0: float = 100.0, dt: float = 1/252):
        self.S0 = S0
        self.dt = dt
    
    def _validate_positive(self, value: float, name: str):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    
    @abstractmethod
    def simulate(self, steps: int = 252, n_paths: int = 10000) -> np.ndarray:
        pass
    
    def get_returns(self, steps: int = 252, n_paths: int = 10000) -> np.ndarray:
        prices = self.simulate(steps, n_paths)
        return np.diff(np.log(prices), axis=0)


# =============================================================================
# CLASS 1: GEOMETRIC BROWNIAN MOTION
# =============================================================================

class GeometricBrownianMotion(PriceSimulatorBase):
    """
    Geometric Brownian Motion price simulator.
    """
    
    def __init__(self, S0: float = 100.0, mu: float = 0.05, sigma: float = 0.2, 
                 dt: float = 1/252):
        super().__init__(S0, dt)
        self.mu = mu
        self.sigma = sigma
        self._validate_positive(self.sigma, "Volatility sigma")
    
    def simulate(self, steps: int = 252, n_paths: int = 10000) -> np.ndarray:
        
        np.random.seed(42)
        
        prices = np.zeros((steps, n_paths))
        prices[0] = self.S0
        
        drift = (self.mu - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt)
        
        Z = np.random.standard_normal((steps - 1, n_paths))
        log_returns = drift + diffusion * Z
        log_prices = np.log(self.S0) + np.cumsum(log_returns, axis=0)
        prices[1:] = np.exp(log_prices)
        
        return prices


# =============================================================================
# CLASS 2: HESTON MODEL (Stochastic Volatility)
# =============================================================================

class HestonModel(PriceSimulatorBase):
    """
    Heston stochastic volatility model.
    """
    
    def __init__(self, S0: float = 100.0, mu: float = 0.05,
                 v0: float = 0.04, kappa: float = 3.0, theta: float = 0.04,
                 sigma_v: float = 0.3, rho: float = -0.7, dt: float = 1/252):
        
        super().__init__(S0, dt)
        self.mu = mu
        self.v0 = max(v0, 1e-6)
        self.kappa = kappa
        self.theta = max(theta, 1e-6)
        self.sigma_v = sigma_v
        self.rho = rho
        self._validate_positive(self.kappa, "Mean reversion kappa")
        self._validate_positive(self.sigma_v, "Vol of vol sigma_v")
        if not -1 <= self.rho <= 1:
            raise ValueError(f"Correlation rho must be between -1 and 1, got {self.rho}")
    
    def simulate(self, steps: int = 252, n_paths: int = 10000) -> np.ndarray:
        
        np.random.seed(42)
        
        prices = np.zeros((steps, n_paths))
        variances = np.zeros((steps, n_paths))
        
        prices[0] = self.S0
        variances[0] = self.v0
        
        Z1 = np.random.standard_normal((steps - 1, n_paths))
        Z2 = np.random.standard_normal((steps - 1, n_paths))
        
        Z_s = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
        Z_v = Z1
        
        for t in range(1, steps):
            var_prev = np.maximum(variances[t-1], 0)
            
            variance_drift = self.kappa * (self.theta - var_prev) * self.dt
            variance_diffusion = self.sigma_v * np.sqrt(var_prev * self.dt) * Z_v[t-1]
            variances[t] = var_prev + variance_drift + variance_diffusion
            variances[t] = np.maximum(variances[t], 1e-6)
            
            var_for_price = np.maximum(variances[t-1], 0)
            price_drift = (self.mu - 0.5 * var_for_price) * self.dt
            price_diffusion = np.sqrt(var_for_price * self.dt) * Z_s[t-1]
            prices[t] = prices[t-1] * np.exp(price_drift + price_diffusion)
        
        return prices


# =============================================================================
# CLASS 3: JUMP-DIFFUSION Merton MODEL
# =============================================================================

class JumpDiffusionModel(PriceSimulatorBase):
    """
    Merton Jump-Diffusion model.
    """
    
    def __init__(self, S0: float = 100.0, mu: float = 0.05, sigma: float = 0.2,
                 lambda_jump: float = 0.5, mu_j: float = -0.05, sigma_j: float = 0.1,
                 dt: float = 1/252):
        
        super().__init__(S0, dt)
        self.mu = mu
        self.sigma = sigma
        self.lambda_jump = lambda_jump
        self.mu_j = mu_j
        self.sigma_j = sigma_j
        self._validate_positive(self.sigma, "Volatility sigma")
        if self.lambda_jump < 0:
            raise ValueError(f"Jump intensity lambda must be non-negative, got {self.lambda_jump}")
        self._validate_positive(self.sigma_j, "Jump volatility sigma_j")
    
    def simulate(self, steps: int = 252, n_paths: int = 10000) -> np.ndarray:
        np.random.seed(42)
        
        prices = np.zeros((steps, n_paths))
        prices[0] = self.S0
        
        drift = (self.mu - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt)
        jump_prob = self.lambda_jump * self.dt
        
        for t in range(1, steps):
            Z = np.random.standard_normal(n_paths)
            gbm_component = drift + diffusion * Z
            
            jump_occurred = np.random.random(n_paths) < jump_prob
            jump_sizes = np.random.normal(self.mu_j, self.sigma_j, n_paths)
            
            total_return = gbm_component + jump_occurred * jump_sizes
            prices[t] = prices[t-1] * np.exp(total_return)
        
        return prices


# =============================================================================
# MAIN EXECUTION - TESTING
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("PRICE SIMULATORS TEST SUITE")
    print("="*70)
    
    # Test GBM
    print("\nTEST: Geometric Brownian Motion")
    print("-"*50)
    gbm = GeometricBrownianMotion()
    paths = gbm.simulate(steps=100, n_paths=50)
    print(f"Generated {paths.shape[1]} paths with {paths.shape[0]} steps")
    print(f"Final price mean: ${np.mean(paths[-1]):.2f}")
    