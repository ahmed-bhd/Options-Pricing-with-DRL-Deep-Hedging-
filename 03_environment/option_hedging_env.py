# -*- coding: utf-8 -*-
"""
================================================================================
OPTION HEDGING ENVIRONMENT
================================================================================

Gymnasium environment for hedging a European call option using Deep Reinforcement Learning.
The agent learns to manage a delta-hedging position to minimize hedging error
while accounting for transaction costs.

State Space (Observation):
    - Normalized stock price (S / K)
    - Time to expiry (T_remaining)
    - Current delta position
    - Moneyness (K / S)
    - Implied volatility (or rolling historical vol)

Action Space:
    - Target delta position (continuous, range [-1, 1])

Reward:
    - Negative of hedging error (PnL - transaction costs)
    - Optional: Penalty for large drawdowns

Terminal:
    - Option expires, final payoff is settled
================================================================================
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
from scipy.stats import norm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# BLACK-SCHOLES HELPERS
# =============================================================================

def black_scholes_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate Black-Scholes call option price.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate
        sigma: Implied volatility
    
    Returns:
        Call option price
    """
    
    if T <= 0:
        return max(S - K, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_call_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate Black-Scholes call option delta.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate
        sigma: Implied volatility
    
    Returns:
        Call option delta (hedge ratio)
    """
    from scipy.stats import norm
    
    if T <= 0:
        return 1.0 if S > K else 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


# =============================================================================
# MAIN ENVIRONMENT CLASS
# =============================================================================

class OptionHedgingEnv(gym.Env):
    """
    Gymnasium environment for hedging a European call option.
    
    The agent starts with a short call option position (sold the option)
    and must hedge dynamically by trading the underlying asset.
    
    Episode ends at option expiry (T = 0).
    """
    
    metadata = {'render_modes': ['human', 'none']}
    
    def __init__(self,
                 price_paths: np.ndarray,
                 strike: float = 100.0,
                 risk_free_rate: float = 0.02,
                 implied_vol: float = 0.20,
                 transaction_cost: float = 0.001,
                 initial_cash: float = 0.0,
                 window: int = 20,
                 reward_scaling: float = 1.0,
                 render_mode: Optional[str] = None):
        """
        Initialize the option hedging environment.
        
        Args:
            price_paths: Simulated price paths (steps x n_paths)
            strike: Option strike price
            risk_free_rate: Annual risk-free rate
            implied_vol: Implied volatility for option pricing
            transaction_cost: Cost per trade as percentage of notional
            initial_cash: Initial cash balance (default: 0)
            window: Rolling window for observation
            reward_scaling: Scaling factor for rewards
            render_mode: Render mode ('human' or 'none')
        """
        super().__init__()
        
        # Store parameters
        self.price_paths = price_paths
        self.n_steps = price_paths.shape[0]
        self.n_paths = price_paths.shape[1]
        self.strike = strike
        self.r = risk_free_rate
        self.sigma = implied_vol
        self.transaction_cost = transaction_cost
        self.initial_cash = initial_cash
        self.window = window
        self.reward_scaling = reward_scaling
        self.render_mode = render_mode
        
        # Action space: target delta position (continuous)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Observation space
        # Features: price, time, position, moneyness, volatility, rolling returns
        obs_size = 5 + window  # 5 base features + window rolling returns
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
        # Reset environment state
        self.reset()
    
    def _get_obs(self) -> np.ndarray:
        """
        Construct observation vector.
        
        Returns:
            Observation array
        """
        # Current state
        S = self.current_price
        T_remaining = self.T_remaining
        delta_pos = self.delta_position
        moneyness = self.strike / S if S > 0 else 1.0
        
        # Rolling returns (last 'window' days)
        start_idx = max(0, self.current_step - self.window)
        rolling_returns = self.returns[start_idx:self.current_step]
        
        if len(rolling_returns) < self.window:
            pad_width = self.window - len(rolling_returns)
            rolling_returns = np.pad(rolling_returns, (pad_width, 0), 'constant')
        
        # Normalize rolling returns
        rolling_returns = np.clip(rolling_returns, -0.1, 0.1) / 0.1
        
        # Features
        features = np.array([
            S / self.strike,           # Normalized price
            T_remaining,                # Time to expiry
            delta_pos,                  # Current position
            moneyness,                  # Moneyness (K/S)
            self.sigma                  # Implied volatility
        ], dtype=np.float32)
        
        # Combine features with rolling returns
        obs = np.concatenate([features, rolling_returns])
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information about the current state.
        
        Returns:
            Dictionary with info
        """
        # Calculate theoretical option price and delta
        if self.T_remaining > 0:
            option_price = black_scholes_call_price(
                self.current_price, self.strike, self.T_remaining, self.r, self.sigma
            )
            option_delta = black_scholes_call_delta(
                self.current_price, self.strike, self.T_remaining, self.r, self.sigma
            )
        else:
            option_price = max(self.current_price - self.strike, 0)
            option_delta = 1.0 if self.current_price > self.strike else 0.0
        
        # Portfolio value = cash + position * stock price
        portfolio_value = self.cash + self.delta_position * self.current_price
        
        # Hedging error relative to theoretical option price
        hedging_error = portfolio_value - option_price
        
        return {
            'step': self.current_step,
            'price': self.current_price,
            'time_remaining': self.T_remaining,
            'delta_position': self.delta_position,
            'cash': self.cash,
            'portfolio_value': portfolio_value,
            'option_price': option_price,
            'option_delta': option_delta,
            'hedging_error': hedging_error,
            'total_transaction_cost': self.total_transaction_cost
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
        
        Returns:
            Initial observation and info
        """
        super().reset(seed=seed)
        
        # Select a random path for this episode
        if hasattr(self, 'current_path'):
            self.current_path = np.random.randint(0, self.n_paths)
        else:
            self.current_path = np.random.randint(0, self.n_paths)
        
        # Reset state variables
        self.current_step = 0
        self.current_price = self.price_paths[0, self.current_path]
        self.delta_position = 0.0
        self.cash = self.initial_cash
        self.total_transaction_cost = 0.0
        
        # Calculate returns for observation
        self.returns = np.diff(np.log(self.price_paths[:, self.current_path]))
        self.returns = np.insert(self.returns, 0, 0)
        
        # Store previous portfolio value for reward calculation
        self.prev_portfolio_value = self.cash + self.delta_position * self.current_price
        
        # Time to expiry (in years)
        self.T_remaining = (self.n_steps - self.current_step) / 252
        
        return self._get_obs(), self._get_info()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one hedging step.
        
        Args:
            action: Target delta position (scalar in [-1, 1])
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Clip action to valid range
        target_delta = np.clip(action[0], -1.0, 1.0)
        
        # Calculate trade
        delta_change = target_delta - self.delta_position
        trade_volume = delta_change  # Each unit of delta corresponds to 1 share
        
        # Transaction cost
        cost = abs(trade_volume) * self.current_price * self.transaction_cost
        self.total_transaction_cost += cost
        
        # Update cash (negative for buying, positive for selling)
        self.cash -= trade_volume * self.current_price
        self.cash -= cost
        
        # Update position
        self.delta_position = target_delta
        
        # Move to next time step
        self.current_step += 1
        
        # Update time to expiry
        self.T_remaining = (self.n_steps - self.current_step) / 252
        
        # Check if episode is done
        terminated = self.current_step >= self.n_steps - 1
        
        # Update price for next step (or final)
        if not terminated:
            self.current_price = self.price_paths[self.current_step, self.current_path]
        else:
            # At expiry, final price is the last price
            self.current_price = self.price_paths[-1, self.current_path]
        
        # Calculate current portfolio value
        current_portfolio_value = self.cash + self.delta_position * self.current_price
        
        # Calculate PnL change
        pnl_change = current_portfolio_value - self.prev_portfolio_value
        self.prev_portfolio_value = current_portfolio_value
        
        # Calculate reward
        if not terminated:
            # Daily reward is the PnL change
            reward = pnl_change * self.reward_scaling
        else:
            # At expiry, settle the option
            payoff = max(self.current_price - self.strike, 0)
            final_pnl = current_portfolio_value - payoff
            reward = final_pnl * self.reward_scaling
        
        # Truncated (not used)
        truncated = False
        
        # Generate observation and info
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """
        Render the current state.
        """
        
        if self.render_mode == 'human':
            info = self._get_info()
            print(f"\n{'='*60}")
            print(f"Step: {info['step']}/{self.n_steps}")
            print(f"{'='*60}")
            print(f"Stock Price: ${info['price']:.2f}")
            print(f"Time Remaining: {info['time_remaining']*252:.0f} days")
            print(f"Delta Position: {info['delta_position']:.4f}")
            print(f"Cash: ${info['cash']:.2f}")
            print(f"Portfolio Value: ${info['portfolio_value']:.2f}")
            print(f"Option Price: ${info['option_price']:.2f}")
            print(f"Hedging Error: ${info['hedging_error']:.2f}")
            print(f"Total Transaction Costs: ${info['total_transaction_cost']:.2f}")


# =============================================================================
# MAIN EXECUTION - TESTING
# =============================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("OPTION HEDGING ENVIRONMENT TEST")
    print("="*70)
    
    # Generate a simple price path for testing
    np.random.seed(42)
    n_steps = 252
    n_paths = 10
    
    # Simple GBM simulation
    S0 = 100
    mu = 0.05
    sigma = 0.2
    dt = 1/252
    
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    Z = np.random.standard_normal((n_steps, n_paths))
    log_returns = drift + diffusion * Z
    log_prices = np.log(S0) + np.cumsum(log_returns, axis=0)
    prices = np.exp(log_prices)
    
    print(f"\nTest Data:")
    print(f"Steps: {n_steps}")
    print(f"Paths: {n_paths}")
    print(f"Initial price: ${prices[0, 0]:.2f}")
    print(f" Final price range: ${prices[-1].min():.2f} - ${prices[-1].max():.2f}")
    
    # Create environment
    print("\nCreating environment...")
    env = OptionHedgingEnv(
        price_paths=prices,
        strike=100,
        risk_free_rate=0.02,
        implied_vol=0.20,
        transaction_cost=0.001,
        initial_cash=0
    )
    
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}")
    
    # Test a random policy
    print("\nTesting random policy...")
    obs, info = env.reset()
    done = False
    total_reward = 0
    portfolio_values = []
    
    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        portfolio_values.append(info['portfolio_value'])
    
    print(f"\nRandom Policy Results:")
    print(f"Final portfolio value: ${info['portfolio_value']:.2f}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Hedging error: ${info['hedging_error']:.2f}")
    print(f"Total transaction costs: ${info['total_transaction_cost']:.2f}")
    
    # Plot portfolio value over time
    print("\nGenerating test plot...")
    plt.figure(figsize=(12, 5))
    plt.plot(portfolio_values, label='Portfolio Value', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Time Step')
    plt.ylabel('Portfolio Value ($)')
    plt.title('Hedging Portfolio Value Over Time (Random Policy)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('test_hedging.png', dpi=150, bbox_inches='tight')
    plt.show()
    