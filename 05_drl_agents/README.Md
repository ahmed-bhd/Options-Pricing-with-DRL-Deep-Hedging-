# 🤖 05_drl_agents - Deep Reinforcement Learning Agents

## Overview

This module provides Deep Reinforcement Learning agents for option hedging.

## Available Agents

| Agent | Type | Key Features | Best For |
|-------|------|--------------|----------|
| **PPO** | On-policy | Stable, industry standard | Most applications |
| **SAC** | Off-policy | Sample efficient, exploration | Limited data |
| **TD3** | Off-policy | Reduced overestimation | Low noise environments |

## Algorithm Comparison

| Feature | PPO | SAC | TD3 |
|---------|-----|-----|-----|
| **Learning Type** | On-policy | Off-policy | Off-policy |
| **Sample Efficiency** | Low | High | Medium |
| **Stability** | High | Medium | Medium |
| **Exploration** | Entropy | Entropy | Noise |
| **Hyperparameters** | Few | Many | Many |

## Usage Example

```python
from src.drl_agents import PPOAgent
from src.environment import OptionHedgingEnv

# Create environment
env = OptionHedgingEnv(price_paths, strike=100)

# Create agent
agent = PPOAgent(env, learning_rate=3e-4, n_steps=2048)

# Train
agent.train(total_timesteps=200000)

# Hedge
obs, info = env.reset()
done = False
while not done:
    action = agent.predict(obs)
    obs, reward, done, truncated, info = env.step(action)