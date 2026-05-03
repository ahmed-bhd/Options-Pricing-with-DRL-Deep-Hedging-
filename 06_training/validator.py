# -*- coding: utf-8 -*-
"""
================================================================================
HEDGING VALIDATOR - OUT-OF-SAMPLE VALIDATION
================================================================================

This module provides out-of-sample validation for hedging strategies.
It ensures that models generalize to unseen data using:
    - Walk-forward validation (time series cross-validation)
    - Rolling window validation
    - Expanding window validation
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)


class HedgingValidator:
    """
    Out-of-sample validator for hedging strategies.
    
    This class provides:
        - Walk-forward validation (time series cross-validation)
        - Rolling window validation
        - Expanding window validation
        - Performance persistence testing
    """
    
    def __init__(self, price_paths: np.ndarray, n_splits: int = 5):
        """
        Initialize validator.
        
        Args:
            price_paths: Array of shape (steps, n_paths)
            n_splits: Number of cross-validation splits
        """
        self.price_paths = price_paths
        self.n_steps = price_paths.shape[0]
        self.n_paths = price_paths.shape[1]
        self.n_splits = min(n_splits, self.n_paths // 2)
    
    def walk_forward_validation(self,
                                agent_class,
                                agent_name: str,
                                total_timesteps: int = 50000,
                                train_window: int = 50,
                                test_window: int = 10,
                                **kwargs) -> Dict:
        """
        Walk-forward validation (time series cross-validation).
        
        This method simulates a realistic trading environment by:
            1. Training on a fixed window of past data
            2. Testing on the next period
            3. Rolling the window forward
        
        Args:
            agent_class: DRL agent class
            agent_name: Name of the agent
            total_timesteps: Training timesteps per window
            train_window: Number of paths in training window
            test_window: Number of paths in test window
            **kwargs: Additional arguments for agent/environment
        
        Returns:
            Dictionary with validation results
        """
        from .trainer import HedgingTrainer
        
        results = []
        n_windows = (self.n_paths - train_window) // test_window
        
        print(f"\nWalk-Forward Validation: {n_windows} windows")
        print(f"Train size: {train_window} paths")
        print(f"Test size: {test_window} paths")
        
        all_pnls = []
        
        for window in range(n_windows):
            start_idx = window * test_window
            train_end = start_idx + train_window
            test_end = min(train_end + test_window, self.n_paths)
            
            train_paths = self.price_paths[:, start_idx:train_end]
            test_paths = self.price_paths[:, train_end:test_end]
            
            print(f"\nWindow {window+1}/{n_windows}:")
            print(f"Train: {train_paths.shape[1]} paths")
            print(f"Test: {test_paths.shape[1]} paths")
            
            # Create temporary trainer for this window
            temp_trainer = HedgingTrainer(
                train_paths, 
                train_ratio=1.0, 
                val_ratio=0.0, 
                test_ratio=0.0,
                random_split=False
            )
            
            try:
                # Train agent on training window
                agent, _ = temp_trainer.train_agent(
                    agent_class, f"{agent_name}_window{window}", 
                    total_timesteps=total_timesteps, 
                    verbose=False
                )
                
                # Evaluate on test window
                window_pnls = []
                for i in range(test_paths.shape[1]):
                    single_path = test_paths[:, i:i+1]
                    test_env = temp_trainer._create_env(single_path)
                    
                    obs, info = test_env.reset()
                    done = False
                    
                    while not done:
                        action = agent.predict(obs)
                        obs, reward, done, truncated, info = test_env.step(action)
                    
                    window_pnls.append(info['portfolio_value'])
                
                results.append({
                    'window': window,
                    'mean_pnl': np.mean(window_pnls),
                    'std_pnl': np.std(window_pnls),
                    'pnls': window_pnls
                })
                all_pnls.extend(window_pnls)
                
                print(f"Mean PnL: ${np.mean(window_pnls):.2f}")
                
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        # Aggregate results
        summary = {
            'agent_name': agent_name,
            'n_windows': len(results),
            'overall_mean_pnl': np.mean(all_pnls) if all_pnls else 0,
            'overall_std_pnl': np.std(all_pnls) if all_pnls else 0,
            'window_results': results,
            'all_pnls': all_pnls
        }
        
        print(f"\nOverall Results:")
        print(f"Mean PnL: ${summary['overall_mean_pnl']:.2f}")
        print(f"Std PnL: ${summary['overall_std_pnl']:.2f}")
        
        return summary
    
    def rolling_window_validation(self,
                                   agent_class,
                                   agent_name: str,
                                   total_timesteps: int = 50000,
                                   window_size: int = 30,
                                   **kwargs) -> Dict:
        """
        Rolling window validation (expanding window).
        
        Each new window includes all previous data (expanding window).
        
        Args:
            agent_class: DRL agent class
            agent_name: Name of the agent
            total_timesteps: Training timesteps per window
            window_size: Initial training window size
            **kwargs: Additional arguments
        
        Returns:
            Dictionary with validation results
        """
        from .trainer import HedgingTrainer
        
        results = []
        all_pnls = []
        
        n_windows = self.n_paths - window_size
        
        print(f"\nRolling Window Validation: {n_windows} windows")
        print(f"Initial window size: {window_size} paths")
        
        for window in range(n_windows):
            current_size = window_size + window
            train_paths = self.price_paths[:, :current_size]
            test_paths = self.price_paths[:, current_size:current_size+1]
            
            if test_paths.shape[1] == 0:
                continue
            
            print(f"\nWindow {window+1}/{n_windows}:")
            print(f"Train: {train_paths.shape[1]} paths")
            print(f"Test: {test_paths.shape[1]} paths")
            
            temp_trainer = HedgingTrainer(
                train_paths, 
                train_ratio=1.0, 
                val_ratio=0.0, 
                test_ratio=0.0,
                random_split=False
            )
            
            try:
                agent, _ = temp_trainer.train_agent(
                    agent_class, f"{agent_name}_rolling{window}", 
                    total_timesteps=total_timesteps, 
                    verbose=False
                )
                
                test_env = temp_trainer._create_env(test_paths)
                obs, info = test_env.reset()
                done = False
                
                while not done:
                    action = agent.predict(obs)
                    obs, reward, done, truncated, info = test_env.step(action)
                
                pnl = info['portfolio_value']
                all_pnls.append(pnl)
                
                results.append({
                    'window': window,
                    'train_size': current_size,
                    'pnl': pnl
                })
                
                print(f"PnL: ${pnl:.2f}")
                
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        summary = {
            'agent_name': agent_name,
            'n_windows': len(results),
            'mean_pnl': np.mean(all_pnls) if all_pnls else 0,
            'std_pnl': np.std(all_pnls) if all_pnls else 0,
            'results': results,
            'all_pnls': all_pnls
        }
        
        print(f"\nOverall Results:")
        print(f"Mean PnL: ${summary['mean_pnl']:.2f}")
        print(f"Std PnL: ${summary['std_pnl']:.2f}")
        
        return summary
    
    def compare_models(self,
                       models: Dict[str, any],
                       n_test_paths: int = 100) -> pd.DataFrame:
        """
        Compare multiple models on the same test set.
        
        Args:
            models: Dictionary of {name: model_instance}
            n_test_paths: Number of test paths
        
        Returns:
            DataFrame with comparison results
        """
        from .trainer import HedgingTrainer
        
        # Use last portion as test set
        test_paths = self.price_paths[:, -n_test_paths:]
        
        all_results = []
        
        for name, agent in models.items():
            print(f"\nTesting {name}...")
            
            pnls = []
            
            for i in range(test_paths.shape[1]):
                temp_trainer = HedgingTrainer(
                    test_paths[:, i:i+1], 
                    train_ratio=1.0, 
                    val_ratio=0.0, 
                    test_ratio=0.0,
                    random_split=False
                )
                
                test_env = temp_trainer._create_env(test_paths[:, i:i+1])
                obs, info = test_env.reset()
                done = False
                
                while not done:
                    action = agent.predict(obs)
                    obs, reward, done, truncated, info = test_env.step(action)
                
                pnls.append(info['portfolio_value'])
            
            all_results.append({
                'Model': name,
                'Mean PnL': np.mean(pnls),
                'Std PnL': np.std(pnls),
                'VaR (95%)': np.percentile(pnls, 5),
                'Win Rate %': np.mean([1 if p > 0 else 0 for p in pnls]) * 100
            })
        
        return pd.DataFrame(all_results).sort_values('Mean PnL', ascending=False)


if __name__ == "__main__":
    print("="*70)
    print("HEDGING VALIDATOR TEST")
    print("="*70)
    
    # Generate sample data
    np.random.seed(42)
    n_steps = 252
    n_paths = 100
    S0 = 100
    mu = 0.05
    sigma = 0.2
    dt = 1/252
    
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    log_returns = drift + diffusion * np.random.standard_normal((n_steps, n_paths))
    log_prices = np.log(S0) + np.cumsum(log_returns, axis=0)
    price_paths = np.exp(log_prices)
    
    # Create validator
    validator = HedgingValidator(price_paths, n_splits=5)

    print("="*70)