import os

# Change to the deep_hedging directory
os.chdir('C:/Users/xps/Desktop/RL/DRL_Courses/Projects/Options Pricing with DRL (Deep Hedging)/deep_hedging')

print("Current directory:", os.getcwd())

# Create config.yaml
config_content = '''# =============================================================================
# DEEP HEDGING WITH DRL - CONFIGURATION FILE
# =============================================================================

project:
  name: "Deep Hedging with DRL"
  version: "1.0.0"

data:
  ticker: "SPY"
  start_date: "2018-01-01"
  end_date: "2023-12-31"
  
  simulation:
    n_steps: 100
    n_paths: 100
    model: "gbm"
    
  gbm:
    S0: 100.0
    mu: 0.05
    sigma: 0.20

environment:
  strike: 100.0
  risk_free_rate: 0.02
  implied_vol: 0.20
  transaction_cost: 0.001
  initial_cash: 0.0
  window: 20
  reward_scaling: 1.0

training:
  train_ratio: 0.6
  val_ratio: 0.2
  test_ratio: 0.2
  total_timesteps: 1000
  n_eval_episodes: 10
  random_split: true
  seed: 42

drl:
  PPO:
    enabled: true
    learning_rate: 0.0003
    n_steps: 2048
    batch_size: 64
    n_epochs: 10
    gamma: 0.99
    
  SAC:
    enabled: false
    
  TD3:
    enabled: false

baselines:
  black_scholes:
    enabled: true
    volatility: 0.20
  delta_gamma:
    enabled: true
    volatility: 0.20
    gamma_scaling: 0.5
  constant_hedge:
    enabled: true
    hedge_ratios: [0.0, 0.5, 1.0]

output:
  results_dir: "results"
  figures_dir: "results/figures"
  reports_dir: "10_reports"
  save_models: true
  save_plots: true
  save_csv: true
  verbose: true
'''

with open('config.yaml', 'w') as f:
    f.write(config_content)

print("✅ config.yaml created successfully in:", os.getcwd())

# Verify files
print("\nFiles in deep_hedging directory:")
for f in os.listdir('.'):
    if f.endswith('.py') or f.endswith('.yaml'):
        print(f"  - {f}")