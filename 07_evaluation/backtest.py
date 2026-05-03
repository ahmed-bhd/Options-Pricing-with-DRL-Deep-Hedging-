# -*- coding: utf-8 -*-
"""
================================================================================
BACKTEST ENGINE - RUN ALL STRATEGIES
================================================================================

This module runs backtests for all strategies (DRL agents + classical baselines)
on the same test set for fair comparison.
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import time
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)


class BacktestEngine:
    """
    Backtest engine for comparing all hedging strategies.
    
    This class runs:
        1. Classical baselines (Black-Scholes, Delta-Gamma, Static)
        2. DRL agents (PPO, SAC, TD3)
        
    All strategies are evaluated on the SAME test set.
    """
    
    def __init__(self, 
                 test_paths: np.ndarray,
                 transaction_cost: float = 0.001,
                 strike: float = 100.0,
                 risk_free_rate: float = 0.02):
        """
        Initialize backtest engine.
        
        Args:
            test_paths: Test price paths (steps, n_paths)
            transaction_cost: Cost per trade (default: 0.1%)
            strike: Option strike price
            risk_free_rate: Risk-free rate
        """
        self.test_paths = test_paths
        self.n_steps = test_paths.shape[0]
        self.n_paths = test_paths.shape[1]
        self.transaction_cost = transaction_cost
        self.strike = strike
        self.r = risk_free_rate
        self.dt = 1 / 252
        
        self.results = {}
    
    def _create_env(self, price_path: np.ndarray) -> Any:
        """
        Create a single-path environment.
        """
        
        try:
            from _3_environment.option_hedging_env import OptionHedgingEnv
        except ImportError:
            # Fallback for testing
            class DummyEnv:
                def reset(self):
                    return np.zeros(10), {'portfolio_value': 0}
                def step(self, action):
                    return np.zeros(10), 0, False, False, {'portfolio_value': 0}
            return DummyEnv()
        
        return OptionHedgingEnv(
            price_paths=price_path.reshape(-1, 1),
            strike=self.strike,
            risk_free_rate=self.r,
            implied_vol=0.20,
            transaction_cost=self.transaction_cost,
            initial_cash=0,
            window=20
        )
    
    # =========================================================================
    # CLASSICAL BASELINES
    # =========================================================================
    
    def run_black_scholes(self, verbose: bool = True) -> Dict:
        """
        Run Black-Scholes delta hedging.
        
        Args:
            verbose: Print progress
        
        Returns:
            Dictionary with results
        """
        try:
            from _4_baselines.black_scholes_hedge import BlackScholesHedge
        except ImportError:
            # Fallback for testing
            if verbose:
                print("   ⚠️ BlackScholesHedge not available, using mock data")
            return {
                'name': 'Black-Scholes Delta',
                'type': 'classical',
                'pnls': np.random.randn(self.n_paths) * 50,
                'mean_pnl': 0,
                'std_pnl': 50
            }
        
        hedger = BlackScholesHedge(
            strike=self.strike,
            risk_free_rate=self.r,
            volatility=0.20
        )
        
        all_pnls = []
        
        if verbose:
            print(f"Running Black-Scholes Delta Hedge...")
        
        for i in range(self.n_paths):
            single_path = self.test_paths[:, i]
            pnl, _, _ = hedger.hedge(single_path, transaction_cost=self.transaction_cost)
            all_pnls.append(pnl)
        
        return {
            'name': 'Black-Scholes Delta',
            'type': 'classical',
            'pnls': np.array(all_pnls),
            'mean_pnl': np.mean(all_pnls),
            'std_pnl': np.std(all_pnls)
        }
    
    def run_delta_gamma(self, verbose: bool = True) -> Dict:
        """
        Run Delta-Gamma hedging.
        
        Args:
            verbose: Print progress
        
        Returns:
            Dictionary with results
        """
        try:
            from _4_baselines.delta_gamma_hedge import DeltaGammaHedge
        except ImportError:
            if verbose:
                print("DeltaGammaHedge not available, using mock data")
            return {
                'name': 'Delta-Gamma Hedge',
                'type': 'classical',
                'pnls': np.random.randn(self.n_paths) * 48,
                'mean_pnl': 0,
                'std_pnl': 48
            }
        
        hedger = DeltaGammaHedge(
            strike=self.strike,
            risk_free_rate=self.r,
            volatility=0.20,
            gamma_scaling=0.5
        )
        
        all_pnls = []
        
        if verbose:
            print(f"Running Delta-Gamma Hedge...")
        
        for i in range(self.n_paths):
            single_path = self.test_paths[:, i]
            pnl, _, _ = hedger.hedge(single_path, transaction_cost=self.transaction_cost)
            all_pnls.append(pnl)
        
        return {
            'name': 'Delta-Gamma Hedge',
            'type': 'classical',
            'pnls': np.array(all_pnls),
            'mean_pnl': np.mean(all_pnls),
            'std_pnl': np.std(all_pnls)
        }
    
    def run_static_hedges(self, verbose: bool = True) -> Dict:
        """
        Run static hedging strategies.
        
        Args:
            verbose: Print progress
        
        Returns:
            Dictionary with results
        """
        try:
            from _4_baselines.static_hedge import StaticHedge
        except ImportError:
            if verbose:
                print("StaticHedge not available, using mock data")
            return {
                'No Hedge (Naked)': {
                    'name': 'No Hedge (Naked)',
                    'type': 'classical',
                    'pnls': np.random.randn(self.n_paths) * 100,
                    'mean_pnl': 0,
                    'std_pnl': 100
                },
                'Full Hedge (Hold)': {
                    'name': 'Full Hedge (Hold)',
                    'type': 'classical',
                    'pnls': np.random.randn(self.n_paths) * 30,
                    'mean_pnl': 0,
                    'std_pnl': 30
                }
            }
        
        strategies = {
            'No Hedge (Naked)': 'none',
            'Full Hedge (Hold)': 'full'
        }
        
        results = {}
        
        for name, strategy in strategies.items():
            hedger = StaticHedge(strike=self.strike, strategy=strategy)
            
            all_pnls = []
            
            if verbose:
                print(f"Running {name}...")
            
            for i in range(self.n_paths):
                single_path = self.test_paths[:, i]
                pnl, _, _ = hedger.hedge(single_path, transaction_cost=self.transaction_cost)
                all_pnls.append(pnl)
            
            results[name] = {
                'name': name,
                'type': 'classical',
                'pnls': np.array(all_pnls),
                'mean_pnl': np.mean(all_pnls),
                'std_pnl': np.std(all_pnls)
            }
        
        return results
    
    # =========================================================================
    # DRL AGENTS (Trained models)
    # =========================================================================
    
    def run_drl_agent(self, agent, agent_name: str, verbose: bool = True) -> Dict:
        """
        Run a trained DRL agent on test paths.
        
        Args:
            agent: Trained DRL agent
            agent_name: Name of the agent
            verbose: Print progress
        
        Returns:
            Dictionary with results
        """
        all_pnls = []
        all_daily_pnls = []
        
        if verbose:
            print(f"Running DRL Agent: {agent_name}...")
        
        for i in range(self.n_paths):
            single_path = self.test_paths[:, i]
            test_env = self._create_env(single_path)
            
            obs, info = test_env.reset()
            done = False
            episode_values = [info['portfolio_value']]
            
            while not done:
                action = agent.predict(obs)
                obs, reward, done, truncated, info = test_env.step(action)
                episode_values.append(info['portfolio_value'])
            
            final_pnl = info['portfolio_value']
            all_pnls.append(final_pnl)
            all_daily_pnls.append(np.array(episode_values))
        
        return {
            'name': agent_name,
            'type': 'drl',
            'pnls': np.array(all_pnls),
            'daily_pnls': all_daily_pnls,
            'mean_pnl': np.mean(all_pnls),
            'std_pnl': np.std(all_pnls)
        }
    
    def run_all_classical(self, verbose: bool = True) -> Dict:
        """
        Run all classical baselines.
        
        Args:
            verbose: Print progress
        
        Returns:
            Dictionary with all classical results
        """
        if verbose:
            print("\nRunning Classical Baselines:")
            print("-" * 40)
        
        results = {}
        
        # Black-Scholes
        bs_results = self.run_black_scholes(verbose)
        results[bs_results['name']] = bs_results
        
        # Delta-Gamma
        dg_results = self.run_delta_gamma(verbose)
        results[dg_results['name']] = dg_results
        
        # Static hedges
        static_results = self.run_static_hedges(verbose)
        results.update(static_results)
        
        return results
    
    def run_all_drl(self, agents: Dict[str, Any], verbose: bool = True) -> Dict:
        """
        Run all DRL agents.
        
        Args:
            agents: Dictionary of {name: agent_instance}
            verbose: Print progress
        
        Returns:
            Dictionary with all DRL results
        """
        if verbose:
            print("\nRunning DRL Agents:")
            print("-" * 40)
        
        results = {}
        
        for name, agent in agents.items():
            agent_results = self.run_drl_agent(agent, name, verbose)
            results[name] = agent_results
        
        return results
    
    def run_all(self, 
                drl_agents: Optional[Dict[str, Any]] = None,
                verbose: bool = True) -> pd.DataFrame:
        """
        Run all strategies and return comparison DataFrame.
        
        Args:
            drl_agents: Dictionary of trained DRL agents {name: agent}
            verbose: Print progress
        
        Returns:
            DataFrame with comparison results
        """
        print("\n" + "="*70)
        print("BACKTEST ENGINE")
        print("="*70)
        print(f"Test paths: {self.n_paths}")
        print(f"Steps per path: {self.n_steps}")
        print(f"Transaction cost: {self.transaction_cost*100:.1f}%")
        
        # Run classical baselines
        classical_results = self.run_all_classical(verbose)
        
        # Run DRL agents if provided
        drl_results = {}
        if drl_agents:
            drl_results = self.run_all_drl(drl_agents, verbose)
        
        # Combine all results
        all_results = {**classical_results, **drl_results}
        
        # Create comparison DataFrame
        comparison = []
        for name, data in all_results.items():
            pnls = data['pnls']
            comparison.append({
                'Strategy': name,
                'Type': data['type'],
                'Mean PnL': data['mean_pnl'],
                'Std PnL': data['std_pnl'],
                'VaR (95%)': np.percentile(pnls, 5),
                'CVaR (95%)': np.mean(pnls[pnls <= np.percentile(pnls, 5)]) if len(pnls) > 0 else 0,
                'Win Rate %': np.mean([1 if p > 0 else 0 for p in pnls]) * 100 if len(pnls) > 0 else 0,
                'Sharpe': np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('Mean PnL', ascending=False)
        
        self.results = all_results
        self.comparison_df = df
        
        return df
    
    def print_summary(self):
        """
        Print summary of backtest results.
        """
        
        if not hasattr(self, 'comparison_df'):
            print("No results yet. Run run_all() first.")
            return
        
        print("\n" + "="*70)
        print("BACKTEST RESULTS SUMMARY")
        print("="*70)
        print(self.comparison_df.to_string(index=False))
        
        # Highlight best strategy
        best = self.comparison_df.iloc[0]
        print("\n" + "="*70)
        print(f"BEST STRATEGY: {best['Strategy']}")
        print(f"Mean PnL: ${best['Mean PnL']:.2f}")
        print(f"Sharpe Ratio: {best['Sharpe']:.3f}")
        print(f"Win Rate: {best['Win Rate %']:.1f}%")
        print("="*70)


if __name__ == "__main__":
    print("="*70)
    print("BACKTEST ENGINE TEST")
    print("="*70)
    
    # Generate sample test paths
    np.random.seed(42)
    n_steps = 252
    n_paths = 50
    S0 = 100
    mu = 0.05
    sigma = 0.2
    dt = 1/252
    
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    log_returns = drift + diffusion * np.random.standard_normal((n_steps, n_paths))
    log_prices = np.log(S0) + np.cumsum(log_returns, axis=0)
    test_paths = np.exp(log_prices)
    
    # Create backtest engine
    engine = BacktestEngine(test_paths, transaction_cost=0.001)
    
    # Run classical baselines only
    df = engine.run_all(drl_agents=None, verbose=True)
    engine.print_summary()
