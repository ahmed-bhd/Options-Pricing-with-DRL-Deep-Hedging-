# -*- coding: utf-8 -*-
"""
================================================================================
SAC AGENT (Soft Actor-Critic)
================================================================================

SAC is an off-policy algorithm that maximizes both expected return and entropy.
It is known for excellent sample efficiency and exploration.

Key features:
    - Entropy regularization for exploration
    - Off-policy learning (replay buffer)
    - Twin Q-functions to reduce overestimation
    - Excellent for continuous action spaces
================================================================================
"""

import numpy as np
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


class SACAgent:
    """
    SAC (Soft Actor-Critic) agent for option hedging.
    
    This implementation uses Stable-Baselines3's SAC.
    """
    
    def __init__(self,
                 env,
                 learning_rate: float = 3e-4,
                 buffer_size: int = 100000,
                 batch_size: int = 256,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 train_freq: int = 1,
                 gradient_steps: int = 1,
                 ent_coef: str = 'auto',
                 verbose: int = 0):
        """
        Initialize SAC agent.
        
        Args:
            env: Gymnasium environment
            learning_rate: Learning rate for optimizer
            buffer_size: Size of replay buffer
            batch_size: Batch size for updates
            tau: Soft update coefficient
            gamma: Discount factor
            train_freq: Training frequency
            gradient_steps: Gradient steps per update
            ent_coef: Entropy coefficient ('auto' or float)
            verbose: Verbosity level
        """
        self.env = env
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.ent_coef = ent_coef
        self.verbose = verbose
        self.name = "SAC"
        self._model = None
        self._is_trained = False
        
        self._init_model()
    
    def _init_model(self):
        """
        Initialize the Stable-Baselines3 SAC model.
        """
        
        try:
            from stable_baselines3 import SAC
            from stable_baselines3.common.monitor import Monitor
            
            self._env = Monitor(self.env)
            
            # Policy kwargs for better performance
            policy_kwargs = dict(
                net_arch=[256, 256],
                activation_fn=torch.nn.ReLU
            )
            
            self._model = SAC(
                "MlpPolicy",
                self._env,
                learning_rate=self.learning_rate,
                buffer_size=self.buffer_size,
                batch_size=self.batch_size,
                tau=self.tau,
                gamma=self.gamma,
                train_freq=self.train_freq,
                gradient_steps=self.gradient_steps,
                ent_coef=self.ent_coef,
                policy_kwargs=policy_kwargs,
                verbose=self.verbose,
                seed=42
            )
            
        except ImportError:
            print("Stable-Baselines3 not installed. Run: pip install stable-baselines3")
            print("Using random policy fallback.")
            self._model = None
    
    def train(self, total_timesteps: int = 100000, **kwargs) -> 'SACAgent':
        """
        Train the SAC agent.
        
        Args:
            total_timesteps: Number of timesteps to train
        
        Returns:
            Self (trained agent)
        """
        if self._model is None:
            print("No model available. Using random policy.")
            self._is_trained = True
            return self
        
        print(f"Training SAC for {total_timesteps} timesteps...")
        self._model.learn(total_timesteps=total_timesteps)
        self._is_trained = True
        print("SAC training completed!")
        
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
            print(f"Model saved to {path}")
    
    def load(self, path: str) -> 'SACAgent':
        """
        Load model from disk.
        """
        
        if self._model is not None:
            from stable_baselines3 import SAC
            self._model = SAC.load(path)
            self._is_trained = True
            print(f"Model loaded from {path}")
        return self
    
    def get_params(self) -> dict:
        """
        Get agent parameters.
        """
        
        return {
            'name': self.name,
            'learning_rate': self.learning_rate,
            'buffer_size': self.buffer_size,
            'batch_size': self.batch_size,
            'tau': self.tau,
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
            ReLU = None


if __name__ == "__main__":
    print("="*60)
    print("SAC Agent Test")
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
    
    # Test SAC agent
    print("\nCreating SAC Agent...")
    agent = SACAgent(dummy_env, verbose=0)
    print(f"Agent name: {agent.name}")
    print(f"Parameters: {agent.get_params()}")
    
    print("\nTesting prediction...")
    obs = np.random.randn(10)
    action = agent.predict(obs)
    print(f"Observation shape: {obs.shape}")
    print(f"Action: {action}")
