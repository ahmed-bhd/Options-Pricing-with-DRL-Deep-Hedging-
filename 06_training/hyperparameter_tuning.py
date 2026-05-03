# -*- coding: utf-8 -*-
"""
================================================================================
HYPERPARAMETER TUNING FOR DRL AGENTS
================================================================================

This module provides hyperparameter optimization using:
    - Grid Search: Exhaustive search over parameter grid
    - Random Search: Random sampling of parameters
    - Optuna: Bayesian Optimization (recommended for DRL)

Optuna is recommended due to its efficiency with few trials.
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
from itertools import product
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)


class HyperparameterTuner:
    """
    Hyperparameter tuner for DRL agents.
    
    Supports:
        - Grid search: Exhaustive search over parameter grid
        - Random search: Random sampling of parameters
        - Optuna: Bayesian optimization (requires optuna package)
    """
    
    def __init__(self, trainer, agent_class, param_space: Dict):
        """
        Initialize hyperparameter tuner.
        
        Args:
            trainer: HedgingTrainer instance
            agent_class: DRL agent class to tune
            param_space: Dictionary of parameter names and ranges
        """
        self.trainer = trainer
        self.agent_class = agent_class
        self.param_space = param_space
        self.best_params = None
        self.best_score = -np.inf
        self.all_results = []
    
    def grid_search(self, 
                    total_timesteps: int = 50000,
                    n_val_episodes: int = 50,
                    verbose: bool = True) -> Dict:
        """
        Perform grid search over parameter space.
        
        Args:
            total_timesteps: Training timesteps per configuration
            n_val_episodes: Number of validation episodes
            verbose: Print progress
        
        Returns:
            Best parameters found
        """
        # Generate all combinations
        param_names = list(self.param_space.keys())
        param_values = list(self.param_space.values())
        combinations = list(product(*param_values))
        
        print(f"\n🔍 Grid Search: {len(combinations)} combinations")
        
        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            
            if verbose:
                print(f"\n[{i+1}/{len(combinations)}] Testing: {params}")
            
            try:
                # Train agent with these parameters
                agent, _ = self.trainer.train_agent(
                    self.agent_class, 
                    f"grid_{i}",
                    total_timesteps=total_timesteps,
                    agent_kwargs=params,
                    verbose=False
                )
                
                # Validate
                val_results = self.trainer.validate_agent(
                    agent, 
                    f"grid_{i}",
                    n_episodes=n_val_episodes,
                    verbose=False
                )
                
                score = val_results['mean_pnl']
                
                self.all_results.append({
                    'params': params,
                    'score': score,
                    'mean_pnl': val_results['mean_pnl'],
                    'std_pnl': val_results['std_pnl'],
                    'var_95': val_results['var_95']
                })
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params
                    if verbose:
                        print(f"New best! Score: ${score:.2f}")
                        
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        print(f"\nBest parameters: {self.best_params}")
        print(f"Best score: ${self.best_score:.2f}")
        
        return self.best_params
    
    def random_search(self,
                      n_trials: int = 20,
                      total_timesteps: int = 50000,
                      n_val_episodes: int = 50,
                      verbose: bool = True) -> Dict:
        """
        Perform random search over parameter space.
        
        Args:
            n_trials: Number of random trials
            total_timesteps: Training timesteps per trial
            n_val_episodes: Number of validation episodes
            verbose: Print progress
        
        Returns:
            Best parameters found
        """
        print(f"\nRandom Search: {n_trials} trials")
        
        for i in range(n_trials):
            # Sample random parameters
            params = {}
            for name, values in self.param_space.items():
                if isinstance(values, list):
                    params[name] = np.random.choice(values)
                elif isinstance(values, tuple) and len(values) == 2:
                    # Continuous parameter
                    if isinstance(values[0], int):
                        params[name] = np.random.randint(values[0], values[1])
                    else:
                        params[name] = np.random.uniform(values[0], values[1])
                elif isinstance(values, dict) and values.get('type') == 'log':
                    params[name] = np.random.uniform(values['low'], values['high'])
                else:
                    params[name] = values
            
            if verbose:
                print(f"\nTrial {i+1}/{n_trials}: {params}")
            
            try:
                agent, _ = self.trainer.train_agent(
                    self.agent_class, 
                    f"random_{i}",
                    total_timesteps=total_timesteps,
                    agent_kwargs=params,
                    verbose=False
                )
                
                val_results = self.trainer.validate_agent(
                    agent, 
                    f"random_{i}",
                    n_episodes=n_val_episodes,
                    verbose=False
                )
                
                score = val_results['mean_pnl']
                
                self.all_results.append({
                    'params': params,
                    'score': score,
                    'mean_pnl': val_results['mean_pnl'],
                    'std_pnl': val_results['std_pnl']
                })
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params
                    if verbose:
                        print(f"New best! Score: ${score:.2f}")
                        
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        print(f"\nBest parameters: {self.best_params}")
        print(f"Best score: ${self.best_score:.2f}")
        
        return self.best_params
    
    def optuna_tuning(self,
                      n_trials: int = 30,
                      total_timesteps: int = 50000,
                      n_val_episodes: int = 50,
                      timeout: Optional[int] = None) -> Dict:
        """
        Perform Bayesian optimization using Optuna.
        
        Requires: pip install optuna
        
        Args:
            n_trials: Number of trials
            total_timesteps: Training timesteps per trial
            n_val_episodes: Number of validation episodes
            timeout: Timeout in seconds
        
        Returns:
            Best parameters found
        """
        try:
            import optuna
            
            def objective(trial):
                # Suggest hyperparameters based on space definition
                params = {}
                for name, space in self.param_space.items():
                    if isinstance(space, list):
                        params[name] = trial.suggest_categorical(name, space)
                    elif isinstance(space, tuple) and len(space) == 2:
                        if isinstance(space[0], int):
                            params[name] = trial.suggest_int(name, space[0], space[1])
                        else:
                            params[name] = trial.suggest_float(name, space[0], space[1])
                    elif isinstance(space, dict) and space.get('type') == 'log':
                        params[name] = trial.suggest_float(
                            name, space['low'], space['high'], log=True
                        )
                    else:
                        params[name] = space
                
                # Train and validate
                agent, _ = self.trainer.train_agent(
                    self.agent_class,
                    f"optuna_{trial.number}",
                    total_timesteps=total_timesteps,
                    agent_kwargs=params,
                    verbose=False
                )
                
                val_results = self.trainer.validate_agent(
                    agent,
                    f"optuna_{trial.number}",
                    n_episodes=n_val_episodes,
                    verbose=False
                )
                
                score = val_results['mean_pnl']
                
                # Store result
                self.all_results.append({
                    'params': params,
                    'score': score,
                    'trial': trial.number
                })
                
                return score
            
            # Create study and optimize
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials, timeout=timeout)
            
            self.best_params = study.best_params
            self.best_score = study.best_value
            
            print(f"\nBest parameters: {self.best_params}")
            print(f"Best score: ${self.best_score:.2f}")
            
            return self.best_params
            
        except ImportError:
            print("Optuna not installed. Run: pip install optuna")
            print("Falling back to random search...")
            return self.random_search(n_trials=n_trials)
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get all tuning results as DataFrame.
        """
        
        return pd.DataFrame(self.all_results)


# Parameter spaces for different algorithms
PPO_PARAM_SPACE = {
    'learning_rate': (1e-5, 1e-3),  # log scale
    'n_steps': [1024, 2048, 4096],
    'batch_size': [32, 64, 128, 256],
    'n_epochs': [5, 10, 20],
    'gamma': (0.95, 0.999),
    'ent_coef': (0.001, 0.05),
    'clip_range': (0.1, 0.3)
}

SAC_PARAM_SPACE = {
    'learning_rate': (1e-5, 1e-3),
    'buffer_size': [50000, 100000, 200000],
    'batch_size': [128, 256, 512],
    'tau': (0.001, 0.01),
    'gamma': (0.95, 0.999),
    'ent_coef': (0.01, 0.2)
}

TD3_PARAM_SPACE = {
    'learning_rate': (1e-5, 1e-3),
    'buffer_size': [50000, 100000, 200000],
    'batch_size': [128, 256],
    'tau': (0.001, 0.01),
    'gamma': (0.95, 0.999),
    'policy_delay': [2, 3, 4]
}


if __name__ == "__main__":
    print("="*70)
    print("HYPERPARAMETER TUNING TEST")
    print("="*70)
    
    # Create dummy trainer
    class DummyTrainer:
        def train_agent(self, agent_class, name, **kwargs):
            class DummyAgent:
                def predict(self, obs):
                    return np.random.uniform(-1, 1, 1)
            return DummyAgent(), {}
        
        def validate_agent(self, agent, name, **kwargs):
            return {'mean_pnl': np.random.randn() * 10, 'std_pnl': 5, 'var_95': -5}
    
    trainer = DummyTrainer()
    
    # Test tuner with small search
    tuner = HyperparameterTuner(trainer, None, {'learning_rate': [0.001, 0.0005]})
    print("="*70)