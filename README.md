# Options Pricing and Hedging with Deep Reinforcement Learning (Deep Hedging)
This project implements a production-grade Deep Hedging framework using Deep Reinforcement Learning (DRL) to optimally price and hedge European call options. The framework addresses the limitations of the traditional Black-Scholes-Merton (BSM) model by accounting for transaction costs, market impact, and discrete trading times.

---

## Key Features

* **Multiple Price Simulators:** Generate asset paths using Geometric Brownian Motion (GBM), Heston Stochastic Volatility, and Jump-Diffusion models.
* **Classical Baselines:** Built-in traditional baselines for performance comparison (Black-Scholes Delta, Delta-Gamma, and Static hedges).
* **DRL Agents:** Modular PPO, SAC, and TD3 implementations built on top of the Stable-Baselines3 API.
* **Real Market Data:** Automated downloading and CSV caching of real options data using `yfinance`.
* **Risk Analytics & Validation:** Computes advanced quantitative metrics such as Sharpe Ratio, Sortino Ratio, VaR, CVaR, Calmar Ratio, and Profit Factor.
* **Statistical Significance Testing:** Includes Student's t-tests and p-values to prove outperformance over classical hedging techniques.
* **Visualization Engine:** Generates 11+ publication-ready plots and comprehensive dashboards.

---

## Project Structure

The project follows a modular, step-by-step pipeline from theory to report generation:

deep_hedging/
├── 01_theory/                  # Mathematical underpinnings and BSM formulas
├── 02_data/                    # Simulators: GBM, Heston, Jump-Diffusion
├── 03_environment/             # Custom Gymnasium option hedging environment
├── 04_baselines/               # Classical methods: Delta, Delta-Gamma, Static
├── 05_drl_agents/              # DRL agent definitions: PPO, SAC, TD3
├── 06_training/                # Training, validation, and hyperparameter tuning
├── 07_evaluation/              # Backtesting, metrics, and statistical tests
├── 08_results/                 # Saved performance data, transactions log
├── 09_visualization/           # Dashboard generation and performance plots
├── 10_reports/                 # Analysis reports and figures
├── config.yaml                 # Configuration parameters
├── main.py                     # Orchestrator
└── requirements.txt            # Dependencies

