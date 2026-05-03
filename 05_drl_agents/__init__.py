# -*- coding: utf-8 -*-
"""
================================================================================
05_DRL_AGENTS MODULE INITIALIZER
================================================================================

This module provides Deep Reinforcement Learning agents for option hedging.

Exports:
    - BaseAgent: Abstract base class for all DRL agents
    - PPOAgent: Proximal Policy Optimization agent
    - SACAgent: Soft Actor-Critic agent
    - TD3Agent: Twin Delayed DDPG agent

Usage:
    from src.drl_agents import PPOAgent, SACAgent
    
    agent = PPOAgent(env)
    agent.train(total_timesteps=100000)
    action = agent.predict(observation)
================================================================================
"""

from .base_agent import BaseAgent
from .ppo_agent import PPOAgent
from .sac_agent import SACAgent
from .td3_agent import TD3Agent

__all__ = [
    'BaseAgent',
    'PPOAgent',
    'SACAgent',
    'TD3Agent'
]

__version__ = '1.0.0'
__author__ = 'Ahmed Baahmed'