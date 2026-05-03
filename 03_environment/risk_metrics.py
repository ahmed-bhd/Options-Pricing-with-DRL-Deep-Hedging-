# -*- coding: utf-8 -*-
"""
================================================================================
RISK METRICS FOR HEDGING EVALUATION
================================================================================

This module provides risk metrics for evaluating hedging strategies:
    - Value at Risk (VaR)
    - Conditional Value at Risk (CVaR / Expected Shortfall)
    - Sharpe Ratio
    - Maximum Drawdown
    - Hedging Error Statistics
================================================================================

"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class RiskMetrics:
    """
    Calculate risk metrics for hedging strategies.
    
    Used to evaluate the performance of DRL hedging agents.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize risk metrics calculator.
        
        Args:
            confidence_level: Confidence level for VaR/CVaR (default: 0.95)
        """
        self.confidence_level = confidence_level
    
    def value_at_risk(self, pnl_array: np.ndarray) -> float:
        """
        Calculate Value at Risk (VaR).
        
        VaR is the maximum loss at a given confidence level.
        
        Args:
            pnl_array: Array of PnL values
        
        Returns:
            VaR value (negative number represents loss)
        """
        if len(pnl_array) == 0:
            return 0.0
        
        return np.percentile(pnl_array, (1 - self.confidence_level) * 100)
    
    def conditional_value_at_risk(self, pnl_array: np.ndarray) -> float:
        """
        Calculate Conditional Value at Risk (CVaR / Expected Shortfall).
        
        CVaR is the expected loss given that loss exceeds VaR.
        
        Args:
            pnl_array: Array of PnL values
        
        Returns:
            CVaR value (negative number represents loss)
        """
        if len(pnl_array) == 0:
            return 0.0
        
        var = self.value_at_risk(pnl_array)
        losses_beyond_var = pnl_array[pnl_array <= var]
        
        if len(losses_beyond_var) == 0:
            return var
        
        return np.mean(losses_beyond_var)
    
    def sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        Calculate annualized Sharpe ratio.
        
        Args:
            returns: Daily returns array
            risk_free_rate: Annual risk-free rate
        
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
    
    def maximum_drawdown(self, portfolio_values: np.ndarray) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            portfolio_values: Time series of portfolio values
        
        Returns:
            Maximum drawdown as percentage (negative number)
        """
        if len(portfolio_values) == 0:
            return 0.0
        
        cummax = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cummax) / cummax
        return np.min(drawdown)
    
    def hedging_error_stats(self, hedging_errors: np.ndarray) -> Dict[str, float]:
        """
        Calculate statistics of hedging errors.
        
        Args:
            hedging_errors: Array of hedging errors
        
        Returns:
            Dictionary with error statistics
        """
        if len(hedging_errors) == 0:
            return {
                'mean_error': 0.0,
                'std_error': 0.0,
                'rmse': 0.0,
                'max_error': 0.0,
                'min_error': 0.0
            }
        
        return {
            'mean_error': np.mean(hedging_errors),
            'std_error': np.std(hedging_errors),
            'rmse': np.sqrt(np.mean(hedging_errors**2)),
            'max_error': np.max(hedging_errors),
            'min_error': np.min(hedging_errors)
        }
    
    def calculate_all(self, 
                      pnl_array: np.ndarray,
                      returns: np.ndarray,
                      portfolio_values: np.ndarray,
                      hedging_errors: np.ndarray) -> Dict[str, float]:
        """
        Calculate all risk metrics.
        
        Args:
            pnl_array: Array of PnL values
            returns: Daily returns array
            portfolio_values: Time series of portfolio values
            hedging_errors: Array of hedging errors
        
        Returns:
            Dictionary with all metrics
        """
        return {
            'var_95': self.value_at_risk(pnl_array),
            'cvar_95': self.conditional_value_at_risk(pnl_array),
            'sharpe_ratio': self.sharpe_ratio(returns),
            'max_drawdown': self.maximum_drawdown(portfolio_values),
            'mean_hedging_error': self.hedging_error_stats(hedging_errors)['mean_error'],
            'rmse_hedging_error': self.hedging_error_stats(hedging_errors)['rmse'],
            'std_hedging_error': self.hedging_error_stats(hedging_errors)['std_error']
        }


def calculate_hedging_pnl(price_path: np.ndarray,
                          actions: np.ndarray,
                          strike: float = 100.0,
                          r: float = 0.02,
                          sigma: float = 0.20,
                          transaction_cost: float = 0.001) -> Tuple[float, np.ndarray]:
    """
    Calculate hedging PnL for a given price path and action sequence.
    
    Args:
        price_path: Array of stock prices over time
        actions: Array of target delta actions
        strike: Option strike price
        r: Risk-free rate
        sigma: Implied volatility
        transaction_cost: Transaction cost rate
    
    Returns:
        Tuple of (final PnL, array of daily PnL)
    """
    n_steps = len(price_path)
    dt = 1 / 252
    
    cash = 0.0
    position = 0.0
    daily_pnl = np.zeros(n_steps)
    
    for t in range(n_steps - 1):
        S = price_path[t]
        target_delta = actions[t]
        
        # Trade
        delta_change = target_delta - position
        cost = abs(delta_change) * S * transaction_cost
        cash -= delta_change * S
        cash -= cost
        position = target_delta
        
        # Daily PnL (mark-to-market)
        S_next = price_path[t + 1]
        daily_pnl[t] = cash + position * S_next - (cash + position * S)
    
    # Terminal payoff
    S_final = price_path[-1]
    payoff = max(S_final - strike, 0)
    final_pnl = cash + position * S_final - payoff
    
    return final_pnl, daily_pnl


if __name__ == "__main__":
    print("="*70)
    print("RISK METRICS TEST")
    print("="*70)
    
    # Generate sample data
    np.random.seed(42)
    sample_pnl = np.random.normal(0, 100, 1000)
    sample_returns = np.random.normal(0.0005, 0.01, 1000)
    sample_values = 10000 * (1 + sample_returns).cumprod()
    sample_errors = np.random.normal(0, 50, 1000)
    
    # Calculate metrics
    rm = RiskMetrics(confidence_level=0.95)
    metrics = rm.calculate_all(sample_pnl, sample_returns, sample_values, sample_errors)
    
    print("\nRisk Metrics Results:")
    print("-"*50)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    print("\n" + "="*70)
    print("Risk Metrics ready!")
    print("="*70)