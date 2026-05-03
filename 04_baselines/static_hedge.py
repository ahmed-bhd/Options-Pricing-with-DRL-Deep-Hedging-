# -*- coding: utf-8 -*-
"""
================================================================================
STATIC HEDGE (Buy and Hold / No Hedge)
================================================================================

Static hedging strategies:
    1. No hedge (naked short option) - Sell option, do nothing
    2. Full hedge (buy and hold) - Buy 1 share at start, hold to expiry
    3. Buy option (opposite position) - Buy the option instead of selling

These serve as simple baselines to show the value of dynamic hedging.
================================================================================
"""

import numpy as np
from typing import Tuple, Dict


class StaticHedge:
    """
    Static hedging strategies (no dynamic rebalancing).
    
    Strategies:
        - 'none': Sell option, no hedge (naked short)
        - 'full': Buy 1 share at start and hold (fully hedged)
        - 'buy_option': Buy the option (opposite position)
    """
    
    def __init__(self, 
                 strike: float = 100.0,
                 strategy: str = 'none'):
        """
        Initialize static hedger.
        
        Args:
            strike: Option strike price
            strategy: 'none', 'full', or 'buy_option'
        """
        self.strike = strike
        self.strategy = strategy
        
        if strategy == 'none':
            self.name = "No Hedge (Naked Short Option)"
            self.hedge_ratio = 0.0
        elif strategy == 'full':
            self.name = "Full Hedge (Buy and Hold 1 Share)"
            self.hedge_ratio = 1.0
        elif strategy == 'buy_option':
            self.name = "Buy Option (Opposite Position)"
            self.hedge_ratio = 0.0  # No stock hedge, but we own the option
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def get_delta(self, S: float, t: float) -> float:
        """
        Return constant hedge ratio.
        
        Args:
            S: Current stock price (unused)
            t: Time to expiry (unused)
        
        Returns:
            Constant hedge ratio
        """
        return self.hedge_ratio
    
    def hedge(self, 
              price_path: np.ndarray,
              dt: float = 1/252,
              transaction_cost: float = 0.001) -> Tuple[float, np.ndarray, Dict]:
        """
        Run static hedging simulation.
        
        For static strategies, no rebalancing occurs after the initial trade.
        
        Args:
            price_path: Array of stock prices over time
            dt: Time step in years (unused for static)
            transaction_cost: Transaction cost rate
        
        Returns:
            Tuple of (final PnL, array of daily PnL, metrics dictionary)
        """
        n_steps = len(price_path)
        S0 = price_path[0]
        S_final = price_path[-1]
        
        if self.strategy == 'none':
            # Sell option, no hedge
            cash = 0.0  # Received premium (simplified)
            position = 0.0
            payoff = max(S_final - self.strike, 0)
            final_pnl = cash + position * S_final - payoff
            
        elif self.strategy == 'full':
            # Buy 1 share at start, hold to expiry
            cash = -S0  # Pay for 1 share
            position = 1.0
            payoff = max(S_final - self.strike, 0)
            final_pnl = cash + position * S_final - payoff
            
        elif self.strategy == 'buy_option':
            # Buy the option instead of selling (opposite position)
            # Simplified: we don't model option premium here
            payoff = -max(S_final - self.strike, 0)  # Opposite payoff
            final_pnl = payoff
            
        else:
            final_pnl = 0.0
        
        # Daily PnL (simplified - only final matters for static)
        daily_pnl = np.zeros(n_steps)
        daily_pnl[-1] = final_pnl
        
        metrics = {
            'final_pnl': final_pnl,
            'total_transaction_costs': 0.0,
            'max_position': abs(position) if 'position' in locals() else 0,
            'avg_position': abs(position) if 'position' in locals() else 0,
            'turnover': 0.0,
            'pnl_std': 0.0
        }
        
        return final_pnl, daily_pnl, metrics


class NoHedge(StaticHedge):
    """
    Sell option with no hedge (naked short).
    """
    
    def __init__(self, strike: float = 100.0):
        super().__init__(strike=strike, strategy='none')


class FullHedge(StaticHedge):
    """
    Buy and hold 1 share (fully hedged).
    """
    
    def __init__(self, strike: float = 100.0):
        super().__init__(strike=strike, strategy='full')


class BuyOption(StaticHedge):
    """
    Buy the option (opposite position).
    """
    
    def __init__(self, strike: float = 100.0):
        super().__init__(strike=strike, strategy='buy_option')


if __name__ == "__main__":
    print("="*60)
    print("Static Hedge Test")
    print("="*60)
    
    # Create a simple price path
    np.random.seed(42)
    n_steps = 50
    dt = 1/252
    S0 = 100
    
    drift = (0.05 - 0.5 * 0.2**2) * dt
    diffusion = 0.2 * np.sqrt(dt)
    Z = np.random.standard_normal(n_steps)
    log_returns = drift + diffusion * Z
    log_prices = np.log(S0) + np.cumsum(log_returns)
    price_path = np.exp(log_prices)
    
    print(f"\nTest Price Path:")
    print(f"Initial price: ${price_path[0]:.2f}")
    print(f"Final price: ${price_path[-1]:.2f}")
    print(f"Option payoff: ${max(price_path[-1] - 100, 0):.2f}")
    
    # Test different static strategies
    strategies = [
        ('none', "No Hedge (Naked Short)"),
        ('full', "Full Hedge (Buy and Hold)"),
        ('buy_option', "Buy Option (Opposite)")
    ]
    
    print("\nStatic Hedging Results:")
    print("-"*50)
    
    for strategy, name in strategies:
        hedger = StaticHedge(strike=100, strategy=strategy)
        pnl, _, metrics = hedger.hedge(price_path, transaction_cost=0.001)
        
        if strategy == 'buy_option':
            print(f"{name:<30} ${pnl:>10.2f}")
        else:
            print(f"{name:<30} ${pnl:>10.2f}")
