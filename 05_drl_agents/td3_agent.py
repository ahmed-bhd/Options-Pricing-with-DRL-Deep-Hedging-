# -*- coding: utf-8 -*-
"""
================================================================================
TD3 AGENT (Twin Delayed Deep Deterministic Policy Gradient)
================================================================================

TD3 is an improvement over DDPG that addresses overestimation bias.
It uses twin Q-functions, delayed policy updates, and target policy smoothing.

Key features:
    - Twin Q-functions to reduce overestimation
    - Delayed policy updates
    - Target policy smoothing
    - Excellent for continuous control
================================================================================
"""

import numpy as np
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


class TD3Agent:
    """
    TD3 (Twin Delayed DDPG) agent for option hedging.
    
    This implementation uses Stable-Baselines3's TD3.
    """
    
    def __init__(self,
                 env,
                 learning_rate: float = 1e-3,
                 buffer_size: int = 100000,
                 batch_size: int = 256,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 train_freq: int = 1,
                 gradient_steps: int = 1,
                 policy_delay: int = 2,
                 target_policy_noise: float = 0.2,
                 target_noise_clip: float = 0.5,
                 verbose: int = 0):
        """
        Initialize TD3 agent.
        
        Args:
            env: Gymnasium environment
            learning_rate: Learning rate for optimizer
            buffer_size: Size of replay buffer
            batch_size: Batch size for updates
            tau: Soft update coefficient
            gamma: Discount factor
            train_freq: Training frequency
            gradient_steps: Gradient steps per update
            policy_delay: Delayed policy updates
            target_policy_noise: Target policy smoothing noise
            target_noise_clip: Noise clipping
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
        self.policy_delay = policy_delay
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.verbose = verbose
        self.name = "TD3"
        self._model = None
        self._is_trained = False
        
        self._init_model()
    
    def _init_model(self):
        """
        Initialize the Stable-Baselines3 TD3 model.
        """
        try:
            from stable_baselines3 import TD3
            from stable_baselines3.common.monitor import Monitor
            from stable_baselines3.common.noise import NormalActionNoise
            
            self._env = Monitor(self.env)
            
            # Action noise for exploration
            n_actions = self.env.action_space.shape[0]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
            
            # Policy kwargs
            policy_kwargs = dict(net_arch=[256, 256])
            
            self._model = TD3(
                "MlpPolicy",
                self._env,
                learning_rate=self.learning_rate,
                buffer_size=self.buffer_size,
                batch_size=self.batch_size,
                tau=self.tau,
                gamma=self.gamma,
                train_freq=self.train_freq,
                gradient_steps=self.gradient_steps,
                policy_delay=self.policy_delay,
                target_policy_noise=self.target_policy_noise,
                target_noise_clip=self.target_noise_clip,
                action_noise=action_noise,
                policy_kwargs=policy_kwargs,
                verbose=self.verbose,
                seed=42
            )
            
        except ImportError:
            print("Stable-Baselines3 not installed. Run: pip install stable-baselines3")
            print("Using random policy fallback.")
            self._model = None
        except Exception as e:
            print(f"Error initializing TD3: {e}")
            print("Using random policy fallback.")
            self._model = None
    
    def train(self, total_timesteps: int = 100000, **kwargs) -> 'TD3Agent':
        """
        Train the TD3 agent.
        
        Args:
            total_timesteps: Number of timesteps to train
        
        Returns:
            Self (trained agent)
        """
        if self._model is None:
            print("No model available. Using random policy.")
            self._is_trained = True
            return self
        
        print(f"Training TD3 for {total_timesteps} timesteps...")
        self._model.learn(total_timesteps=total_timesteps)
        self._is_trained = True
        print("TD3 training completed!")
        
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
    
    def load(self, path: str) -> 'TD3Agent':
        """
        Load model from disk.
        """
        
        if self._model is not None:
            from stable_baselines3 import TD3
            self._model = TD3.load(path)
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
            'buffer_size': self.buffer_size,
            'batch_size': self.batch_size,
            'tau': self.tau,
            'gamma': self.gamma,
            'policy_delay': self.policy_delay,
            'trained': self._is_trained
        }


if __name__ == "__main__":
    print("="*60)
    print("TD3 Agent Test")
    print("="*60)
    
    # Create a dummy environment for testing
    import gymnasium as gym
    from gymnasium import spaces
    
    class DummyEnv(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,))
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
            self._max_episode_steps = 100
            
        def reset(self, seed=None, options=None):
            return np.zeros(10).astype(np.float32), {}
        
        def step(self, action):
            return np.zeros(10).astype(np.float32), 0.0, False, False, {}
    
    dummy_env = DummyEnv()
    
    # Test TD3 agent
    print("\nCreating TD3 Agent...")
    agent = TD3Agent(dummy_env, verbose=0)
    print(f"Agent name: {agent.name}")
    print(f"Parameters: {agent.get_params()}")
    
    print("\nTesting prediction...")
    obs = np.random.randn(10).astype(np.float32)
    action = agent.predict(obs)
    print(f"Observation shape: {obs.shape}")
    print(f"Action: {action}")
