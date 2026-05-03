# -*- coding: utf-8 -*-
"""
================================================================================
BLACK-SCHOLES DELTA HEDGING
================================================================================

Standard delta hedging using the Black-Scholes model.
This is the classical benchmark for option hedging.

Delta = N(d1)
where d1 = (ln(S/K) + (r + σ²/2)T) / (σ√T)
================================================================================
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple, Dict


class BlackScholesHedge:
    """
    Black-Scholes delta hedging strategy.
    
    This is the standard approach derived from the Black-Scholes model.
    The hedger maintains a delta-neutral position by continuously
    rebalancing the underlying asset.
    """
    
    def __init__(self, 
                 strike: float = 100.0,
                 risk_free_rate: float = 0.02,
                 volatility: float = 0.20):
        """
        Initialize Black-Scholes hedger.
        
        Args:
            strike: Option strike price
            risk_free_rate: Annual risk-free rate
            volatility: Implied volatility (assumed constant)
        """
        self.strike = strike
        self.r = risk_free_rate
        self.sigma = volatility
        self.name = "Black-Scholes Delta Hedge"
    
    def get_delta(self, S: float, t: float) -> float:
        """
        Calculate Black-Scholes delta.
        
        Formula: Δ = N(d1)
        
        Args:
            S: Current stock price
            t: Time to expiry (in years)
        
        Returns:
            Delta value (between 0 and 1)
        """
        if t <= 0:
            return 1.0 if S > self.strike else 0.0
        
        d1 = (np.log(S / self.strike) + (self.r + 0.5 * self.sigma**2) * t) / (self.sigma * np.sqrt(t))
        return norm.cdf(d1)
    
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
    
    def get_greeks(self, S: float, t: float) -> dict:
        """
        Calculate all Greeks for reference.
        
        Args:
            S: Current stock price
            t: Time to expiry
        
        Returns:
            Dictionary with delta, gamma, vega, theta
        """
        if t <= 0:
            return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0}
        
        d1 = (np.log(S / self.strike) + (self.r + 0.5 * self.sigma**2) * t) / (self.sigma * np.sqrt(t))
        d2 = d1 - self.sigma * np.sqrt(t)
        
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * self.sigma * np.sqrt(t))
        vega = S * norm.pdf(d1) * np.sqrt(t) / 100
        theta = -S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(t)) - self.r * self.strike * np.exp(-self.r * t) * norm.cdf(d2)
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta / 365  # Daily theta
        }


if __name__ == "__main__":
    print("="*60)
    print("Black-Scholes Delta Hedge Test")
    print("="*60)
    
    # Create hedger
    hedger = BlackScholesHedge(strike=100, risk_free_rate=0.02, volatility=0.20)
    
    # Test delta calculation
    print("\nDelta Values:")
    print("-"*40)
    test_prices = [90, 95, 100, 105, 110]
    for S in test_prices:
        delta = hedger.get_delta(S, t=0.5)
        print(f"S=${S}: Δ={delta:.4f}")
    
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
    print(f"Option payoff: ${max(price_path[-1] - 100, 0):.2f}")
