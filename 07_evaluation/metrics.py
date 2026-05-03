# -*- coding: utf-8 -*-
"""
================================================================================
PERFORMANCE METRICS
================================================================================

This module provides comprehensive performance metrics for evaluating
hedging strategies:

    - Sharpe Ratio (risk-adjusted return)
    - Sortino Ratio (downside risk-adjusted)
    - Calmar Ratio (return / max drawdown)
    - Value at Risk (VaR)
    - Conditional Value at Risk (CVaR)
    - Maximum Drawdown
    - Win Rate
    - Profit Factor
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union


class MetricsCalculator:
    """
    Calculate performance metrics for hedging strategies.
    
    All metrics are calculated from PnL arrays.
    """
    
    @staticmethod
    def sharpe_ratio(pnls: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        Calculate annualized Sharpe ratio.
        
        Sharpe = (mean_return - risk_free) / std_return
        
        Args:
            pnls: Array of PnL values
            risk_free_rate: Annual risk-free rate
        
        Returns:
            Sharpe ratio
        """
        if len(pnls) == 0 or np.std(pnls) == 0:
            return 0.0
        
        daily_rf = risk_free_rate / 252
        excess_returns = pnls - daily_rf
        
        return np.mean(excess_returns) / np.std(pnls) * np.sqrt(252)
    
    @staticmethod
    def sortino_ratio(pnls: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino ratio (only penalizes downside deviation).
        
        Sortino = (mean_return - risk_free) / downside_deviation
        
        Args:
            pnls: Array of PnL values
            risk_free_rate: Annual risk-free rate
        
        Returns:
            Sortino ratio
        """
        if len(pnls) == 0:
            return 0.0
        
        daily_rf = risk_free_rate / 252
        excess_returns = pnls - daily_rf
        
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else np.std(pnls)
        
        if downside_std == 0:
            return 0.0
        
        return np.mean(excess_returns) / downside_std * np.sqrt(252)
    
    @staticmethod
    def calmar_ratio(pnls: np.ndarray, portfolio_values: Optional[np.ndarray] = None) -> float:
        """
        Calculate Calmar ratio (return / max drawdown).
        
        Args:
            pnls: Array of PnL values
            portfolio_values: Daily portfolio values (if None, computed from pnls)
        
        Returns:
            Calmar ratio
        """
        if portfolio_values is None:
            portfolio_values = np.cumsum(pnls)
        
        total_return = portfolio_values[-1] - portfolio_values[0] if len(portfolio_values) > 0 else 0
        max_dd = MetricsCalculator.max_drawdown(portfolio_values)
        
        if max_dd == 0:
            return 0.0
        
        return total_return / abs(max_dd)
    
    @staticmethod
    def value_at_risk(pnls: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) at given confidence level.
        
        Args:
            pnls: Array of PnL values
            confidence: Confidence level (default: 0.95)
        
        Returns:
            VaR (negative number represents loss)
        """
        if len(pnls) == 0:
            return 0.0
        
        return np.percentile(pnls, (1 - confidence) * 100)
    
    @staticmethod
    def conditional_value_at_risk(pnls: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR / Expected Shortfall).
        
        Args:
            pnls: Array of PnL values
            confidence: Confidence level (default: 0.95)
        
        Returns:
            CVaR (negative number represents loss)
        """
        if len(pnls) == 0:
            return 0.0
        
        var = MetricsCalculator.value_at_risk(pnls, confidence)
        losses_beyond_var = pnls[pnls <= var]
        
        if len(losses_beyond_var) == 0:
            return var
        
        return np.mean(losses_beyond_var)
    
    @staticmethod
    def max_drawdown(portfolio_values: np.ndarray) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            portfolio_values: Time series of portfolio values
        
        Returns:
            Maximum drawdown (negative number)
        """
        if len(portfolio_values) == 0:
            return 0.0
        
        cummax = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cummax) / cummax
        
        return np.min(drawdown)
    
    @staticmethod
    def win_rate(pnls: np.ndarray) -> float:
        """
        Calculate win rate (percentage of positive PnL).
        
        Args:
            pnls: Array of PnL values
        
        Returns:
            Win rate (0-100)
        """
        if len(pnls) == 0:
            return 0.0
        
        return (pnls > 0).sum() / len(pnls) * 100
    
    @staticmethod
    def profit_factor(pnls: np.ndarray) -> float:
        """
        Calculate profit factor (gross profit / gross loss).
        
        Profit Factor = Sum(positive PnL) / |Sum(negative PnL)|
        
        - > 1: Profitable strategy
        - = 1: Break-even
        - < 1: Losing strategy
        - inf: No losses (all positive trades)
        - 0: No profits (all negative trades)
        
        Args:
            pnls: Array of PnL values
        
        Returns:
            Profit factor
        """
        if len(pnls) == 0:
            return 0.0
        
        gross_profit = np.sum(pnls[pnls > 0])
        gross_loss = abs(np.sum(pnls[pnls < 0]))
        
        if gross_loss > 0:
            return gross_profit / gross_loss
        elif gross_profit > 0:
            return np.inf  # No losses, infinite profit factor
        else:
            return 0.0  # No profits, no losses
    
    @staticmethod
    def calculate_all(pnls: np.ndarray, portfolio_values: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate all metrics.
        
        Args:
            pnls: Array of PnL values
            portfolio_values: Daily portfolio values (optional)
        
        Returns:
            Dictionary with all metrics
        """
        if portfolio_values is None:
            portfolio_values = np.cumsum(pnls)
        
        return {
            'sharpe_ratio': MetricsCalculator.sharpe_ratio(pnls),
            'sortino_ratio': MetricsCalculator.sortino_ratio(pnls),
            'calmar_ratio': MetricsCalculator.calmar_ratio(pnls, portfolio_values),
            'var_95': MetricsCalculator.value_at_risk(pnls, 0.95),
            'cvar_95': MetricsCalculator.conditional_value_at_risk(pnls, 0.95),
            'var_99': MetricsCalculator.value_at_risk(pnls, 0.99),
            'cvar_99': MetricsCalculator.conditional_value_at_risk(pnls, 0.99),
            'max_drawdown': MetricsCalculator.max_drawdown(portfolio_values),
            'win_rate': MetricsCalculator.win_rate(pnls),
            'profit_factor': MetricsCalculator.profit_factor(pnls),
            'total_return': portfolio_values[-1] - portfolio_values[0] if len(portfolio_values) > 0 else 0,
            'mean_pnl': np.mean(pnls) if len(pnls) > 0 else 0,
            'std_pnl': np.std(pnls) if len(pnls) > 0 else 0
        }
    
    @staticmethod
    def compare_strategies(results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple strategies' metrics.
        
        Args:
            results: Dictionary of {strategy_name: {'pnls': array, 'portfolio_values': array}}
        
        Returns:
            DataFrame with comparison
        """
        comparison = []
        
        for name, data in results.items():
            pnls = data.get('pnls', data.get('values', []))
            portfolio_values = data.get('portfolio_values', np.cumsum(pnls))
            
            if len(pnls) > 0:
                metrics = MetricsCalculator.calculate_all(pnls, portfolio_values)
                metrics['Strategy'] = name
                comparison.append(metrics)
        
        df = pd.DataFrame(comparison)
        return df.sort_values('sharpe_ratio', ascending=False)


if __name__ == "__main__":
    print("="*70)
    print("METRICS CALCULATOR TEST")
    print("="*70)
    
    # Generate sample PnL data
    np.random.seed(42)
    sample_pnls = np.random.normal(100, 200, 252)
    
    # Calculate all metrics
    metrics = MetricsCalculator.calculate_all(sample_pnls)
    
    print("\nSample Metrics:")
    for name, value in metrics.items():
        if name == 'profit_factor':
            if value == np.inf:
                print(f"   {name}: ∞")
            else:
                print(f"   {name}: {value:.4f}")
        else:
            print(f"   {name}: {value:.4f}")
    
    # Test profit factor with different scenarios
    print("\nProfit Factor Test Scenarios:")
    
    # Scenario 1: Mixed profits and losses
    mixed = np.array([100, -50, 200, -30, 50])
    pf = MetricsCalculator.profit_factor(mixed)
    print(f"Mixed profits/losses: {pf:.2f}")
    
    # Scenario 2: All profits
    all_profits = np.array([100, 50, 200, 30, 50])
    pf = MetricsCalculator.profit_factor(all_profits)
    print(f"All profits: {pf}")
    
    # Scenario 3: All losses
    all_losses = np.array([-100, -50, -200, -30, -50])
    pf = MetricsCalculator.profit_factor(all_losses)
    print(f"All losses: {pf:.2f}")
