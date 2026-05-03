# -*- coding: utf-8 -*-
"""
================================================================================
HEDGING TRAINER - TRAINING LOOP WITH DATA SPLITTING
================================================================================

This module handles the training loop for DRL agents with proper
data splitting to avoid data leakage.

Key principles:
    1. Training data: NEVER seen by agent during evaluation
    2. Validation data: Used for hyperparameter tuning
    3. Test data: ONLY used for final evaluation
    4. No lookahead bias: Agent cannot see future prices
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import time
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)


class HedgingTrainer:
    """
    Trainer for DRL agents with proper train/validation/test split.
    
    Splitting strategy:
        - Training: 60% (used to train the agent)
        - Validation: 20% (used for hyperparameter tuning)
        - Test: 20% (used for final evaluation, NEVER seen during training)
    """
    
    def __init__(self, 
                 price_paths: np.ndarray,
                 train_ratio: float = 0.6,
                 val_ratio: float = 0.2,
                 test_ratio: float = 0.2,
                 random_split: bool = True,
                 seed: int = 42):
        """
        Initialize trainer with price paths.
        
        Args:
            price_paths: Array of shape (steps, n_paths)
            train_ratio: Ratio for training (default: 0.6)
            val_ratio: Ratio for validation (default: 0.2)
            test_ratio: Ratio for testing (default: 0.2)
            random_split: If True, random split. If False, sequential split.
            seed: Random seed for reproducibility
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        self.full_paths = price_paths
        self.n_steps = price_paths.shape[0]
        self.n_paths = price_paths.shape[1]
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_split = random_split
        self.seed = seed
        
        np.random.seed(seed)
        
        # Split the data
        self._split_data()
        
        print(f"\n{'='*60}")
        print("DATA SPLIT SUMMARY")
        print(f"{'='*60}")
        print(f"Total paths: {self.n_paths}")
        print(f"Training paths: {self.train_paths.shape[1]} ({train_ratio*100:.0f}%)")
        print(f"Validation paths: {self.val_paths.shape[1]} ({val_ratio*100:.0f}%)")
        print(f"Test paths: {self.test_paths.shape[1]} ({test_ratio*100:.0f}%)")
        print(f"Steps per path: {self.n_steps}")
        print(f"Split method: {'Random' if random_split else 'Sequential'}")
    
    def _split_data(self):
        """
        Split price paths into train, validation, and test sets.
        """
        
        if self.random_split:
            # Random split (shuffle paths)
            indices = np.random.permutation(self.n_paths)
            
            train_end = int(self.n_paths * self.train_ratio)
            val_end = train_end + int(self.n_paths * self.val_ratio)
            
            train_indices = indices[:train_end]
            val_indices = indices[train_end:val_end]
            test_indices = indices[val_end:]
            
            self.train_paths = self.full_paths[:, train_indices]
            self.val_paths = self.full_paths[:, val_indices]
            self.test_paths = self.full_paths[:, test_indices]
        else:
            # Sequential split (first N for train, next M for val, last for test)
            train_end = int(self.n_paths * self.train_ratio)
            val_end = train_end + int(self.n_paths * self.val_ratio)
            
            self.train_paths = self.full_paths[:, :train_end]
            self.val_paths = self.full_paths[:, train_end:val_end]
            self.test_paths = self.full_paths[:, val_end:]
    
    def _create_env(self, price_paths: np.ndarray, **env_kwargs) -> Any:
        """
        Create an environment with the given price paths.
        """
        from _3_environment.option_hedging_env import OptionHedgingEnv
        
        default_kwargs = {
            'strike': 100,
            'risk_free_rate': 0.02,
            'implied_vol': 0.20,
            'transaction_cost': 0.001,
            'initial_cash': 0,
            'window': 20
        }
        
        default_kwargs.update(env_kwargs)
        
        return OptionHedgingEnv(price_paths=price_paths, **default_kwargs)
    
    def train_agent(self, 
                    agent_class,
                    agent_name: str,
                    total_timesteps: int = 100000,
                    agent_kwargs: Optional[Dict] = None,
                    env_kwargs: Optional[Dict] = None,
                    verbose: bool = True) -> Tuple[Any, Dict]:
        """
        Train a DRL agent on TRAINING data only.
        
        Args:
            agent_class: DRL agent class (PPOAgent, SACAgent, etc.)
            agent_name: Name of the agent
            total_timesteps: Number of timesteps to train
            agent_kwargs: Additional arguments for agent
            env_kwargs: Additional arguments for environment
            verbose: Print progress
        
        Returns:
            Trained agent and training metrics
        """
        if agent_kwargs is None:
            agent_kwargs = {}
        if env_kwargs is None:
            env_kwargs = {}
        
        if verbose:
            print(f"\nTraining {agent_name}")
            print(f"Training paths: {self.train_paths.shape[1]}")
            print(f"Timesteps: {total_timesteps:,}")
        
        # Create training environment with TRAINING paths ONLY
        train_env = self._create_env(self.train_paths, **env_kwargs)
        
        # Create agent
        agent = agent_class(train_env, **agent_kwargs)
        
        # Train
        start_time = time.time()
        agent.train(total_timesteps=total_timesteps)
        training_time = time.time() - start_time
        
        if verbose:
            print(f"Training completed in {training_time/60:.2f} minutes")
        
        metrics = {
            'training_time': training_time,
            'total_timesteps': total_timesteps,
            'train_paths': self.train_paths.shape[1]
        }
        
        return agent, metrics
    
    def validate_agent(self,
                       agent,
                       agent_name: str,
                       n_episodes: Optional[int] = None,
                       verbose: bool = True) -> Dict:
        """
        Validate agent on VALIDATION data (for hyperparameter tuning).
        
        Args:
            agent: Trained DRL agent
            agent_name: Name of the agent
            n_episodes: Number of validation episodes (default: all)
            verbose: Print progress
        
        Returns:
            Dictionary with validation metrics
        """
        if verbose:
            print(f"\nValidating {agent_name} on validation set...")
        
        n_episodes = n_episodes or self.val_paths.shape[1]
        n_episodes = min(n_episodes, self.val_paths.shape[1])
        
        all_pnls = []
        
        for i in range(n_episodes):
            single_path = self.val_paths[:, i:i+1]
            val_env = self._create_env(single_path)
            
            obs, info = val_env.reset()
            done = False
            
            while not done:
                action = agent.predict(obs)
                obs, reward, done, truncated, info = val_env.step(action)
            
            all_pnls.append(info['portfolio_value'])
        
        results = {
            'agent_name': agent_name,
            'mean_pnl': np.mean(all_pnls),
            'std_pnl': np.std(all_pnls),
            'var_95': np.percentile(all_pnls, 5),
            'median_pnl': np.median(all_pnls),
            'n_episodes': len(all_pnls)
        }
        
        if verbose:
            print(f"Mean PnL: ${results['mean_pnl']:.2f} ± ${results['std_pnl']:.2f}")
        
        return results
    
    def evaluate_agent(self,
                       agent,
                       agent_name: str,
                       n_episodes: Optional[int] = None,
                       verbose: bool = True) -> Dict:
        """
        Evaluate trained agent on TEST data (unseen during training).
        
        Args:
            agent: Trained DRL agent
            agent_name: Name of the agent
            n_episodes: Number of test episodes (default: all)
            verbose: Print progress
        
        Returns:
            Dictionary with evaluation metrics
        """
        if verbose:
            print(f"\nEvaluating {agent_name} on TEST set (unseen data)...")
        
        n_episodes = n_episodes or self.test_paths.shape[1]
        n_episodes = min(n_episodes, self.test_paths.shape[1])
        
        all_pnls = []
        all_daily_pnls = []
        
        for i in range(n_episodes):
            single_path = self.test_paths[:, i:i+1]
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
        
        results = {
            'agent_name': agent_name,
            'mean_pnl': np.mean(all_pnls),
            'std_pnl': np.std(all_pnls),
            'var_95': np.percentile(all_pnls, 5),
            'cvar_95': np.mean([p for p in all_pnls if p <= np.percentile(all_pnls, 5)]),
            'win_rate': np.mean([1 if p > 0 else 0 for p in all_pnls]),
            'max_pnl': np.max(all_pnls),
            'min_pnl': np.min(all_pnls),
            'median_pnl': np.median(all_pnls),
            'all_pnls': all_pnls,
            'all_daily_pnls': all_daily_pnls,
            'n_episodes': len(all_pnls)
        }
        
        if verbose:
            print(f"Mean PnL: ${results['mean_pnl']:.2f}")
            print(f"Std PnL: ${results['std_pnl']:.2f}")
            print(f"VaR (95%): ${results['var_95']:.2f}")
            print(f"Win Rate: {results['win_rate']*100:.1f}%")
        
        return results
    
    def run_full_pipeline(self,
                          agent_class,
                          agent_name: str,
                          total_timesteps: int = 100000,
                          agent_kwargs: Optional[Dict] = None,
                          env_kwargs: Optional[Dict] = None,
                          verbose: bool = True) -> Dict:
        """
        Run complete training pipeline: train → validate → evaluate.
        
        Args:
            agent_class: DRL agent class
            agent_name: Name of the agent
            total_timesteps: Training timesteps
            agent_kwargs: Agent arguments
            env_kwargs: Environment arguments
            verbose: Print progress
        
        Returns:
            Dictionary with all results
        """
        # Train
        agent, train_metrics = self.train_agent(
            agent_class, agent_name, total_timesteps, 
            agent_kwargs, env_kwargs, verbose
        )
        
        # Validate
        val_results = self.validate_agent(agent, agent_name, verbose=verbose)
        
        # Evaluate on test set
        test_results = self.evaluate_agent(agent, agent_name, verbose=verbose)
        
        return {
            'agent': agent,
            'train_metrics': train_metrics,
            'validation_results': val_results,
            'test_results': test_results
        }


if __name__ == "__main__":
    print("="*70)
    print("HEDGING TRAINER TEST")
    print("="*70)
    
    # Generate sample price paths
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
    
    print(f"\nGenerated {n_paths} price paths with {n_steps} steps each")
    
    # Create trainer
    trainer = HedgingTrainer(price_paths, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

    print("="*70)