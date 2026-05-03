# -*- coding: utf-8 -*-
"""
================================================================================
BASE AGENT CLASS
================================================================================

Abstract base class for all Deep Reinforcement Learning agents.
Defines the interface that all DRL agents must implement.
================================================================================
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any


class BaseAgent(ABC):
    """
    Abstract base class for all DRL agents.
    
    All DRL agents must implement:
        - train(): Train the agent on an environment
        - predict(): Get action from observation
        - save(): Save model to disk
        - load(): Load model from disk
    """
    
    def __init__(self, name: str = "BaseAgent"):
        """
        Initialize base agent.
        
        Args:
            name: Name of the agent
        """
        self.name = name
        self._is_trained = False
    
    @abstractmethod
    def train(self, env, total_timesteps: int = 100000, **kwargs) -> 'BaseAgent':
        """
        Train the agent on the given environment.
        
        Args:
            env: Gymnasium environment
            total_timesteps: Number of timesteps to train
            **kwargs: Additional training parameters
        
        Returns:
            Self (trained agent)
        """
        pass
    
    @abstractmethod
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Predict action from observation.
        
        Args:
            observation: Current observation from environment
            deterministic: If True, return deterministic action
        
        Returns:
            Action to take
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: File path to save model
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> 'BaseAgent':
        """
        Load model from disk.
        
        Args:
            path: File path to load model from
        
        Returns:
            Self (loaded agent)
        """
        pass
    
    def is_trained(self) -> bool:
        """
        Check if agent has been trained.
        """
        
        return self._is_trained
    
    def get_name(self) -> str:
        """
        Get agent name.
        """
        
        return self.name
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get agent parameters (for logging).
        """
        
        return {
            'name': self.name,
            'trained': self._is_trained,
            'type': self.__class__.__name__
        }


if __name__ == "__main__":
    print("="*60)
    print("Base Agent Class")
    print("="*60)
    print("This is an abstract base class.")
