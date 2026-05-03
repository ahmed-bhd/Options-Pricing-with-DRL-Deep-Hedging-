# -*- coding: utf-8 -*-
"""
================================================================================
PPO AGENT (Proximal Policy Optimization)
================================================================================

PPO is a policy gradient method that uses clipped objectives for stable training.
It is widely used as the industry standard for continuous control tasks.

Key features:
    - Clipped surrogate objective
    - Multiple epochs of updates per batch
    - Advantage normalization
    - Excellent for option hedging due to stability
================================================================================
"""

import numpy as np
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


class PPOAgent:
    """
    PPO (Proximal Policy Optimization) agent for option hedging.
    
    This implementation uses Stable-Baselines3's PPO.
    """
    
    def __init__(self, 
                 env,
                 learning_rate: float = 3e-4,
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: float = 0.2,
                 ent_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 verbose: int = 0):
        """
        Initialize PPO agent.
        
        Args:
            env: Gymnasium environment
            learning_rate: Learning rate for optimizer
            n_steps: Number of steps per rollout
            batch_size: Batch size for updates
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: Clipping parameter
            ent_coef: Entropy coefficient (exploration)
            max_grad_norm: Gradient clipping norm
            verbose: Verbosity level
        """
        self.env = env
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.verbose = verbose
        self.name = "PPO"
        self._model = None
        self._is_trained = False
        
        self._init_model()
    
    def _init_model(self):
        """
        Initialize the Stable-Baselines3 PPO model.
        """
        
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.monitor import Monitor
            
            # Wrap environment with Monitor for logging
            self._env = Monitor(self.env)
            
            # Policy kwargs for better performance
            policy_kwargs = dict(
                net_arch=[dict(pi=[128, 128], vf=[128, 128])],
                activation_fn=torch.nn.Tanh
            )
            
            self._model = PPO(
                "MlpPolicy",
                self._env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                clip_range=self.clip_range,
                ent_coef=self.ent_coef,
                max_grad_norm=self.max_grad_norm,
                policy_kwargs=policy_kwargs,
                verbose=self.verbose,
                seed=42
            )
            
        except ImportError:
            print("Stable-Baselines3 not installed. Run: pip install stable-baselines3")
            print("Using random policy fallback.")
            self._model = None
    
    def train(self, total_timesteps: int = 100000, **kwargs) -> 'PPOAgent':
        """
        Train the PPO agent.
        
        Args:
            total_timesteps: Number of timesteps to train
        
        Returns:
            Self (trained agent)
        """
        if self._model is None:
            print("No model available. Using random policy.")
            self._is_trained = True
            return self
        
        print(f"Training PPO for {total_timesteps} timesteps...")
        self._model.learn(total_timesteps=total_timesteps)
        self._is_trained = True
        print("PPO training completed!")
        
        return self
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Predict action from observation.
        
        Args:
            observation: Current observation from environment
            deterministic: If True, return deterministic action
        
        Returns:
            Action (target delta position)
        """
        if self._model is None:
            # Random policy fallback
            return np.random.uniform(-1, 1, 1)
        
        action, _ = self._model.predict(observation, deterministic=deterministic)
        return action
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        """
        
        if self._model is not None:
            self._model.save(path)
            print(f"✅ Model saved to {path}")
    
    def load(self, path: str) -> 'PPOAgent':
        """
        Load model from disk.
        """
        
        if self._model is not None:
            from stable_baselines3 import PPO
            self._model = PPO.load(path)
            self._is_trained = True
            print(f"✅ Model loaded from {path}")
        return self
    
    def get_params(self) -> dict:
        """
        Get agent parameters.
        """
        
        return {
            'name': self.name,
            'learning_rate': self.learning_rate,
            'n_steps': self.n_steps,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'gamma': self.gamma,
            'ent_coef': self.ent_coef,
            'trained': self._is_trained
        }


# Import torch for activation function
try:
    import torch
except ImportError:
    class torch:
        class nn:
            Tanh = None


if __name__ == "__main__":
    print("="*60)
    print("PPO Agent Test")
    print("="*60)
    
    # Create a dummy environment for testing
    import gymnasium as gym
    from gymnasium import spaces
    
    class DummyEnv(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,))
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
            
        def reset(self, seed=None, options=None):
            return np.zeros(10), {}
        
        def step(self, action):
            return np.zeros(10), 0, False, False, {}
    
    dummy_env = DummyEnv()
    
    # Test PPO agent
    print("\nCreating PPO Agent...")
    agent = PPOAgent(dummy_env, verbose=0)
    print(f"Agent name: {agent.name}")
    print(f"Parameters: {agent.get_params()}")
    
    print("\nTesting prediction...")
    obs = np.random.randn(10)
    action = agent.predict(obs)
    print(f"Observation shape: {obs.shape}")
    print(f"Action: {action}")
