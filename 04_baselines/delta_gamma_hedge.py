# -*- coding: utf-8 -*-
"""
================================================================================
DELTA-GAMMA HEDGING
================================================================================

Second-order hedging that accounts for gamma (convexity).
This is an improvement over simple delta hedging as it better manages
the non-linear payoff of options.
================================================================================
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple, Dict


class DeltaGammaHedge:
    """
    Delta-Gamma hedging strategy.
    
    This strategy accounts for both delta and gamma risk.
    It provides better hedging for options with high convexity.
    """
    
    def __init__(self, 
                 strike: float = 100.0,
                 risk_free_rate: float = 0.02,
                 volatility: float = 0.20,
                 gamma_scaling: float = 0.5):
        """
        Initialize Delta-Gamma hedger.
        
        Args:
            strike: Option strike price
            risk_free_rate: Annual risk-free rate
            volatility: Implied volatility
            gamma_scaling: Scaling factor for gamma adjustment (0-1)
        """
        self.strike = strike
        self.r = risk_free_rate
        self.sigma = volatility
        self.gamma_scaling = gamma_scaling
        self.name = "Delta-Gamma Hedge"
    
    def get_delta(self, S: float, t: float) -> float:
        """
        Calculate delta with gamma adjustment.
        
        The gamma adjustment increases delta when gamma is high
        to better capture convexity.
        
        Args:
            S: Current stock price
            t: Time to expiry (in years)
        
        Returns:
            Adjusted delta value (between 0 and 1)
        """
        if t <= 0:
            return 1.0 if S > self.strike else 0.0
        
        # Calculate standard delta
        d1 = (np.log(S / self.strike) + (self.r + 0.5 * self.sigma**2) * t) / (self.sigma * np.sqrt(t))
        delta = norm.cdf(d1)
        
        # Calculate gamma
        gamma = norm.pdf(d1) / (S * self.sigma * np.sqrt(t))
        
        # Gamma adjustment (larger positions when gamma is high)
        gamma_adjustment = gamma * self.gamma_scaling * S
        
        # Adjusted delta (clipped to [0, 1])
        adjusted_delta = np.clip(delta + gamma_adjustment, 0, 1)
        
        return adjusted_delta
    
    def get_gamma(self, S: float, t: float) -> float:
        """
        Calculate option gamma.
        
        Args:
            S: Current stock price
            t: Time to expiry
        
        Returns:
            Gamma value
        """
        if t <= 0:
            return 0.0
        
        d1 = (np.log(S / self.strike) + (self.r + 0.5 * self.sigma**2) * t) / (self.sigma * np.sqrt(t))
        return norm.pdf(d1) / (S * self.sigma * np.sqrt(t))
    
    def hedge(self, 
              price_path: np.ndarray,
              dt: float = 1/252,
              transaction_cost: float = 0.001) -> Tuple[float, np.ndarray, Dict]:
        """
        Run hedging simulation on a single price path.
        
        Args:
            price_path: Array of stock prices over time
            dt: Time step in years
            transaction_cost: Transaction cost rate
        
        Returns:
            Tuple of (final PnL, array of daily PnL, metrics dictionary)
        """
        n_steps = len(price_path)
        cash = 0.0
        position = 0.0
        daily_pnl = np.zeros(n_steps)
        positions = np.zeros(n_steps)
        transaction_costs = np.zeros(n_steps)
        
        positions[0] = position
        
        for t in range(n_steps - 1):
            S = price_path[t]
            T_remaining = (n_steps - t - 1) * dt
            
            target_delta = self.get_delta(S, T_remaining)
            
            delta_change = target_delta - position
            cost = abs(delta_change) * S * transaction_cost
            
            cash -= delta_change * S
            cash -= cost
            position = target_delta
            
            positions[t + 1] = position
            transaction_costs[t] = cost
            
            S_next = price_path[t + 1]
            daily_pnl[t] = cash + position * S_next - (cash + position * S)
        
        # Terminal payoff
        S_final = price_path[-1]
        payoff = max(S_final - self.strike, 0)
        final_pnl = cash + position * S_final - payoff
        
        metrics = {
            'final_pnl': final_pnl,
            'total_transaction_costs': np.sum(transaction_costs),
            'max_position': np.max(np.abs(positions)),
            'avg_position': np.mean(np.abs(positions)),
            'turnover': np.sum(np.abs(np.diff(positions))),
            'pnl_std': np.std(daily_pnl)
        }
        
        return final_pnl, daily_pnl, metrics


if __name__ == "__main__":
    print("="*60)
    print("Delta-Gamma Hedge Test")
    print("="*60)
    
    # Create hedger
    hedger = DeltaGammaHedge(strike=100, risk_free_rate=0.02, volatility=0.20, gamma_scaling=0.5)
    
    print("\nDelta vs Gamma Values:")
    print("-"*50)
    
    test_prices = [90, 95, 100, 105, 110]
    for S in test_prices:
        delta = hedger.get_delta(S, t=0.5)
        gamma = hedger.get_gamma(S, t=0.5)
        print(f"S=${S:3d}: Δ={delta:.4f}, Γ={gamma:.4f}")
    
    # Test a simple price path
    print("\nSimple Hedge Test:")
    print("-"*40)
    
    # Create a simple price path
    np.random.seed(42)
    n_steps = 50
    dt = 1/252
    S0 = 100
    mu = 0.05
    sigma = 0.2
    
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    Z = np.random.standard_normal(n_steps)
    log_returns = drift + diffusion * Z
    log_prices = np.log(S0) + np.cumsum(log_returns)
    price_path = np.exp(log_prices)
    
    # Hedge
    pnl, daily_pnl, metrics = hedger.hedge(price_path, dt=dt, transaction_cost=0.001)
    
    print(f"Final PnL: ${pnl:.2f}")
    print(f"Total transaction costs: ${metrics['total_transaction_costs']:.2f}")
    print(f"Turnover: {metrics['turnover']:.2f}")
    print(f"Final price: ${price_path[-1]:.2f}")
 