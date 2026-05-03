# 🚀 Deep Hedging with Deep Reinforcement Learning

## 📋 Project Overview

This project implements **Deep Hedging** - using Deep Reinforcement Learning (DRL) to optimally hedge European call options in the presence of transaction costs and discrete trading.

The system compares:
- **Classical Baselines**: Black-Scholes Delta, Delta-Gamma, Static hedges
- **DRL Agents**: PPO, SAC, TD3

## 🎯 Key Results

| Metric | Best Classical | Best DRL |
|--------|----------------|----------|
| Sharpe Ratio | X.XX | X.XX |
| Mean PnL | $X.XX | $X.XX |
| Win Rate | XX% | XX% |

## 📁 Project Structure
deep_hedging/
├── 01_theory/ # Black-Scholes, option pricing theory
├── 02_data/ # Price simulators (GBM, Heston, Jump)
├── 03_environment/ # Gymnasium hedging environment
├── 04_baselines/ # Classical hedging strategies
├── 05_drl_agents/ # DRL agents (PPO, SAC, TD3)
├── 06_training/ # Training pipeline with data split
├── 07_evaluation/ # Backtest and comparison
├── 08_results/ # Results storage
├── 09_visualization/ # Professional plots
├── 10_reports/ # Analysis reports
├── config.yaml # Configuration file
├── main.py # Main orchestrator
├── requirements.txt # Dependencies
└── README.md # This file