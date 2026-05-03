# 📈 03_environment - Option Hedging Environment

## Overview

This module provides a custom Gymnasium environment for hedging European call options using Deep Reinforcement Learning (Deep Hedging).

## Key Components

| File | Description |
|------|-------------|
| `option_hedging_env.py` | Main Gymnasium environment |
| `risk_metrics.py` | VaR, CVaR, Sharpe ratio calculators |

## Environment Details

### State Space (Observation)
- Normalized stock price (S/K)
- Time to expiry
- Current delta position
- Moneyness (K/S)
- Implied volatility
- Rolling historical returns

### Action Space
- Target delta position (continuous, -1 to 1)

### Reward
- Daily PnL change minus transaction costs
- Final reward = final portfolio value - option payoff

## Usage Example

```python
from src.environment import OptionHedgingEnv
from src.data import GeometricBrownianMotion

# Generate price paths
gbm = GeometricBrownianMotion()
paths = gbm.simulate(steps=252, n_paths=10000)

# Create environment
env = OptionHedgingEnv(
    price_paths=paths,
    strike=100,
    transaction_cost=0.001
)

# Train DRL agent
obs, info = env.reset()
done = False
while not done:
    action = agent.predict(obs)
    obs, reward, done, truncated, info = env.step(action)