# -*- coding: utf-8 -*-
"""
================================================================================
BASE HEDGE CLASS
================================================================================

Abstract base class for all hedging strategies.
Defines the interface that all hedgers must implement.
================================================================================
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple, Optional


class BaseHedge(ABC):
    """
    Abstract base class for all hedging strategies.
    
    All hedging strategies must implement the hedge() method.
    """
    
    def __init__(self, name: str = "BaseHedge"):
        """
        Initialize base hedger.
        
        Args:
            name: Name of the hedging strategy
        """
        self.name = name
        self.transaction_cost = 0.0
    
    @abstractmethod
    def get_delta(self, S: float, t: float, **kwargs) -> float:
        """
        Calculate the target delta at a given state.
        
        Args:
            S: Current stock price
            t: Time to expiry (in years)
            **kwargs: Additional parameters
        
        Returns:
            Target delta position
        """
        pass
    
    def hedge(self, 
              price_path: np.ndarray,
              strike: float = 100.0,
              dt: float = 1/252,
              transaction_cost: float = 0.001,
              verbose: bool = False) -> Tuple[float, np.ndarray, Dict]:
        """
        Run hedging simulation on a single price path.
        
        Args:
            price_path: Array of stock prices over time
            strike: Option strike price
            dt: Time step in years
            transaction_cost: Transaction cost rate
            verbose: Print progress if True
        
        Returns:
            Tuple of (final PnL, array of daily PnL, metrics dictionary)
        """
        self.strike = strike
        n_steps = len(price_path)
        cash = 0.0
        position = 0.0
        daily_pnl = np.zeros(n_steps)
        positions = np.zeros(n_steps)
        transaction_costs = np.zeros(n_steps)
        
        # Store initial values
        positions[0] = position
        
        for t in range(n_steps - 1):
            S = price_path[t]
            T_remaining = (n_steps - t - 1) * dt
            
            # Get target delta
            target_delta = self.get_delta(S, T_remaining)
            
            # Calculate trade
            delta_change = target_delta - position
            cost = abs(delta_change) * S * transaction_cost
            
            # Update cash and position
            cash -= delta_change * S
            cash -= cost
            position = target_delta
            
            # Store
            positions[t + 1] = position
            transaction_costs[t] = cost
            
            # Daily PnL (mark-to-market)
            S_next = price_path[t + 1]
            daily_pnl[t] = cash + position * S_next - (cash + position * S)
        
        # Terminal payoff
        S_final = price_path[-1]
        payoff = max(S_final - strike, 0)
        final_pnl = cash + position * S_final - payoff
        
        # Metrics
        metrics = {
            'final_pnl': final_pnl,
            'total_transaction_costs': np.sum(transaction_costs),
            'max_position': np.max(np.abs(positions)),
            'avg_position': np.mean(np.abs(positions)),
            'turnover': np.sum(np.abs(np.diff(positions))),
            'pnl_std': np.std(daily_pnl)
        }
        
        return final_pnl, daily_pnl, metrics
    
    def hedge_multiple_paths(self,
                             price_paths: np.ndarray,
                             strike: float = 100.0,
                             dt: float = 1/252,
                             transaction_cost: float = 0.001,
                             verbose: bool = False) -> Dict:
        """
        Run hedging simulation on multiple price paths.
        
        Args:
            price_paths: Array of shape (steps, n_paths)
            strike: Option strike price
            dt: Time step in years
            transaction_cost: Transaction cost rate
            verbose: Print progress if True
        
        Returns:
            Dictionary with results
        """
        n_paths = price_paths.shape[1]
        all_pnls = []
        all_metrics = []
        
        for i in range(n_paths):
            if verbose and i % 100 == 0:
                print(f"   Hedging path {i+1}/{n_paths}")
            
            pnl, _, metrics = self.hedge(price_paths[:, i], strike, dt, transaction_cost)
            all_pnls.append(pnl)
            all_metrics.append(metrics)
        
        # Aggregate results
        pnls = np.array(all_pnls)
        
        return {
            'mean_pnl': np.mean(pnls),
            'std_pnl': np.std(pnls),
            'var_95': np.percentile(pnls, 5),
            'cvar_95': np.mean(pnls[pnls <= np.percentile(pnls, 5)]),
            'mean_transaction_costs': np.mean([m['total_transaction_costs'] for m in all_metrics]),
            'all_pnls': pnls
        }


if __name__ == "__main__":
    print("="*60)
    print("Base Hedge Class")
    print("="*60)
    print("This is an abstract base class - no direct testing needed.")