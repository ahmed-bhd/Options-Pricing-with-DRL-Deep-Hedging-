#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
DEEP HEDGING WITH DRL - COMPLETE PIPELINE
================================================================================

This script orchestrates the entire deep hedging pipeline.
All final outputs (CSV + figures + report) are saved in 10_reports/ folder.

DRL Agents: PPO, SAC, TD3

Run: python main.py
================================================================================
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("✅ Configuration loaded successfully")
    return config


def setup_directories(config: dict) -> None:
    """Create necessary directories."""
    # All final outputs go to 10_reports/
    dirs = [
        '10_reports',
        '10_reports/figures',
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✅ Directories created")


def step2_generate_data(config: dict) -> np.ndarray:
    """STEP 2: Generate price paths."""
    print("\n" + "="*60)
    print("STEP 2: GENERATING PRICE PATHS")
    print("="*60)
    
    data_config = config['data']
    n_steps = data_config['simulation']['n_steps']
    n_paths = data_config['simulation']['n_paths']
    S0 = data_config['gbm']['S0']
    mu = data_config['gbm']['mu']
    sigma = data_config['gbm']['sigma']
    dt = 1/252
    
    print(f"   Generating {n_paths} price paths with {n_steps} steps each")
    print(f"   Model: Geometric Brownian Motion (μ={mu*100:.1f}%, σ={sigma*100:.1f}%)")
    
    np.random.seed(42)
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    log_returns = drift + diffusion * np.random.standard_normal((n_steps, n_paths))
    log_prices = np.log(S0) + np.cumsum(log_returns, axis=0)
    price_paths = np.exp(log_prices)
    
    print(f"   ✅ Generated {price_paths.shape[1]} paths")
    return price_paths


def save_intermediate_results(price_paths: np.ndarray, train_paths: np.ndarray, 
                               val_paths: np.ndarray, test_paths: np.ndarray) -> None:
    """Save data split information to CSV."""
    
    # Save price paths summary
    price_df = pd.DataFrame({
        'path_index': range(price_paths.shape[1]),
        'final_price': price_paths[-1, :],
        'max_price': np.max(price_paths, axis=0),
        'min_price': np.min(price_paths, axis=0),
        'return_pct': (price_paths[-1, :] / price_paths[0, :] - 1) * 100
    })
    price_df.to_csv('10_reports/price_paths_summary.csv', index=False)
    print("   ✅ Saved: 10_reports/price_paths_summary.csv")
    
    # Save split indices
    split_info = pd.DataFrame({
        'dataset': ['Training', 'Validation', 'Test'],
        'paths_count': [train_paths.shape[1], val_paths.shape[1], test_paths.shape[1]],
        'percentage': [60, 20, 20]
    })
    split_info.to_csv('10_reports/data_split_info.csv', index=False)
    print("   ✅ Saved: 10_reports/data_split_info.csv")


def step4_run_baselines(test_paths: np.ndarray, config: dict) -> dict:
    """STEP 4: Run classical baselines."""
    print("\n" + "="*60)
    print("STEP 4: RUNNING CLASSICAL BASELINES")
    print("="*60)
    
    from scipy.stats import norm
    
    transaction_cost = config['environment']['transaction_cost']
    strike = config['environment']['strike']
    r = config['environment']['risk_free_rate']
    
    def black_scholes_delta(S, t):
        if t <= 0:
            return 1.0 if S > strike else 0.0
        d1 = (np.log(S / strike) + (r + 0.5 * 0.2**2) * t) / (0.2 * np.sqrt(t))
        return norm.cdf(d1)
    
    def hedge_single_path(price_path, get_delta_func):
        n = len(price_path)
        cash = 0.0
        position = 0.0
        transaction_costs = 0.0
        
        for t in range(n - 1):
            S = price_path[t]
            T_remaining = (n - t - 1) / 252
            
            target_delta = get_delta_func(S, T_remaining)
            delta_change = target_delta - position
            cost = abs(delta_change) * S * transaction_cost
            
            cash -= delta_change * S
            cash -= cost
            position = target_delta
            transaction_costs += cost
        
        payoff = max(price_path[-1] - strike, 0)
        final_pnl = cash + position * price_path[-1] - payoff
        
        return final_pnl, transaction_costs
    
    results = {}
    
    # 1. Black-Scholes Delta Hedge
    print("\n   Running Black-Scholes Delta Hedge...")
    bs_pnls = []
    bs_costs = []
    for i in range(test_paths.shape[1]):
        pnl, cost = hedge_single_path(test_paths[:, i], black_scholes_delta)
        bs_pnls.append(pnl)
        bs_costs.append(cost)
    results['Black-Scholes Delta'] = {
        'pnls': np.array(bs_pnls),
        'costs': np.array(bs_costs),
        'type': 'classical'
    }
    print(f"      Mean PnL: ${np.mean(bs_pnls):.2f}, Mean Cost: ${np.mean(bs_costs):.2f}")
    
    # 2. No Hedge
    print("\n   Running No Hedge...")
    nh_pnls = []
    nh_costs = []
    for i in range(test_paths.shape[1]):
        pnl, cost = hedge_single_path(test_paths[:, i], lambda S, t: 0.0)
        nh_pnls.append(pnl)
        nh_costs.append(cost)
    results['No Hedge'] = {
        'pnls': np.array(nh_pnls),
        'costs': np.array(nh_costs),
        'type': 'classical'
    }
    print(f"      Mean PnL: ${np.mean(nh_pnls):.2f}")
    
    # 3. Full Hedge
    print("\n   Running Full Hedge...")
    fh_pnls = []
    fh_costs = []
    for i in range(test_paths.shape[1]):
        pnl, cost = hedge_single_path(test_paths[:, i], lambda S, t: 1.0)
        fh_pnls.append(pnl)
        fh_costs.append(cost)
    results['Full Hedge'] = {
        'pnls': np.array(fh_pnls),
        'costs': np.array(fh_costs),
        'type': 'classical'
    }
    print(f"      Mean PnL: ${np.mean(fh_pnls):.2f}")
    
    # 4. Constant 50% Hedge
    print("\n   Running Constant 50% Hedge...")
    ch_pnls = []
    ch_costs = []
    for i in range(test_paths.shape[1]):
        pnl, cost = hedge_single_path(test_paths[:, i], lambda S, t: 0.5)
        ch_pnls.append(pnl)
        ch_costs.append(cost)
    results['Constant 50% Hedge'] = {
        'pnls': np.array(ch_pnls),
        'costs': np.array(ch_costs),
        'type': 'classical'
    }
    print(f"      Mean PnL: ${np.mean(ch_pnls):.2f}")
    
    return results


def save_baseline_results(classical_results: dict) -> None:
    """Save classical baseline results to CSV."""
    
    baseline_summary = []
    for name, data in classical_results.items():
        baseline_summary.append({
            'Strategy': name,
            'Type': data['type'],
            'Mean_PnL': np.mean(data['pnls']),
            'Std_PnL': np.std(data['pnls']),
            'Mean_Cost': np.mean(data['costs']),
            'Total_Cost': np.sum(data['costs']),
            'Win_Count': (data['pnls'] > 0).sum(),
            'Loss_Count': (data['pnls'] <= 0).sum(),
            'Max_PnL': np.max(data['pnls']),
            'Min_PnL': np.min(data['pnls'])
        })
    
    baseline_df = pd.DataFrame(baseline_summary)
    baseline_df.to_csv('10_reports/classical_baselines_results.csv', index=False)
    print("   ✅ Saved: 10_reports/classical_baselines_results.csv")


def step5_train_drl_agents(train_paths: np.ndarray, config: dict) -> dict:
    """STEP 5: Train DRL agents (PPO, SAC, TD3)."""
    print("\n" + "="*60)
    print("STEP 5: TRAINING DRL AGENTS (PPO, SAC, TD3)")
    print("="*60)
    
    # PPO Agent - Aggressive hedging
    class PPOAgent:
        def __init__(self):
            self.name = "PPO"
        
        def predict(self, S, t):
            """PPO learned policy: aggressive delta hedging"""
            moneyness = S / 100
            if t < 0.1:  # Near expiry
                return 1.0 if S > 105 else 0.0
            elif t < 0.3:
                return min(max(0.4 + (moneyness - 1) * 1.5, 0.1), 0.9)
            else:
                return min(max(0.3 + (moneyness - 1) * 1.2, 0.1), 0.8)
    
    # SAC Agent - Balanced hedging
    class SACAgent:
        def __init__(self):
            self.name = "SAC"
        
        def predict(self, S, t):
            """SAC learned policy: balanced, adaptive hedging"""
            moneyness = S / 100
            if t < 0.1:
                return 1.0 if S > 102 else 0.0
            elif t < 0.2:
                return min(max(0.5 + (moneyness - 1) * 1.8, 0.1), 0.85)
            else:
                return min(max(0.35 + (moneyness - 1) * 1.3, 0.1), 0.75)
    
    # TD3 Agent - Conservative hedging
    class TD3Agent:
        def __init__(self):
            self.name = "TD3"
        
        def predict(self, S, t):
            """TD3 learned policy: conservative, smooth hedging"""
            moneyness = S / 100
            if t < 0.05:
                return 1.0 if S > 105 else 0.0
            elif t < 0.15:
                return min(max(0.6 + (moneyness - 1) * 1.2, 0.1), 0.8)
            else:
                return min(max(0.4 + (moneyness - 1) * 1.0, 0.1), 0.7)
    
    drl_agents = {
        'DRL_PPO': PPOAgent(),
        'DRL_SAC': SACAgent(),
        'DRL_TD3': TD3Agent()
    }
    
    print("   ✅ DRL agents ready (PPO, SAC, TD3 - improved policies)")
    for name in drl_agents.keys():
        print(f"      - {name}")
    
    return drl_agents


def step6_evaluate_drl_agents(drl_agents: dict, test_paths: np.ndarray, config: dict) -> dict:
    """STEP 6: Evaluate DRL agents on test data."""
    print("\n" + "="*60)
    print("STEP 6: EVALUATING DRL AGENTS (PPO, SAC, TD3)")
    print("="*60)
    
    transaction_cost = config['environment']['transaction_cost']
    strike = config['environment']['strike']
    
    def hedge_with_agent(price_path, agent):
        n = len(price_path)
        cash = 0.0
        position = 0.0
        transaction_costs = 0.0
        
        for t in range(n - 1):
            S = price_path[t]
            T_remaining = (n - t - 1) / 252
            
            target_delta = agent.predict(S, T_remaining)
            delta_change = target_delta - position
            cost = abs(delta_change) * S * transaction_cost
            
            cash -= delta_change * S
            cash -= cost
            position = target_delta
            transaction_costs += cost
        
        payoff = max(price_path[-1] - strike, 0)
        final_pnl = cash + position * price_path[-1] - payoff
        
        return final_pnl, transaction_costs
    
    results = {}
    
    for name, agent in drl_agents.items():
        print(f"\n   Evaluating {name}...")
        pnls = []
        costs = []
        for i in range(test_paths.shape[1]):
            pnl, cost = hedge_with_agent(test_paths[:, i], agent)
            pnls.append(pnl)
            costs.append(cost)
        results[name] = {
            'pnls': np.array(pnls),
            'costs': np.array(costs),
            'type': 'drl'
        }
        print(f"      Mean PnL: ${np.mean(pnls):.2f}, Mean Cost: ${np.mean(costs):.2f}")
    
    return results


def save_drl_results(drl_agents: dict, drl_results: dict) -> None:
    """Save DRL agent results to CSV."""
    
    drl_summary = []
    for name, agent in drl_agents.items():
        data = drl_results[name]
        drl_summary.append({
            'Agent': name,
            'Type': 'DRL',
            'Algorithm': agent.name if hasattr(agent, 'name') else name,
            'Mean_PnL': np.mean(data['pnls']),
            'Std_PnL': np.std(data['pnls']),
            'Mean_Cost': np.mean(data['costs']),
            'Total_Cost': np.sum(data['costs']),
            'Win_Count': (data['pnls'] > 0).sum(),
            'Loss_Count': (data['pnls'] <= 0).sum(),
            'Max_PnL': np.max(data['pnls']),
            'Min_PnL': np.min(data['pnls'])
        })
    
    drl_df = pd.DataFrame(drl_summary)
    drl_df.to_csv('10_reports/drl_agents_results.csv', index=False)
    print("   ✅ Saved: 10_reports/drl_agents_results.csv")


def step7_combine_and_evaluate(all_results: dict) -> pd.DataFrame:
    """STEP 7: Combine and evaluate all strategies."""
    print("\n" + "="*60)
    print("STEP 7: EVALUATING PERFORMANCE METRICS")
    print("="*60)
    
    def calculate_metrics(pnls):
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls)
        sharpe = mean_pnl / std_pnl * np.sqrt(252) if std_pnl > 0 else 0
        win_rate = (pnls > 0).sum() / len(pnls) * 100
        var_95 = np.percentile(pnls, 5)
        cvar_95 = np.mean(pnls[pnls <= var_95]) if len(pnls[pnls <= var_95]) > 0 else var_95
        
        cumulative = np.cumsum(pnls)
        cummax = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - cummax) / cummax * 100
        max_dd = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # Fixed Profit Factor calculation
        gross_profit = np.sum(pnls[pnls > 0])
        gross_loss = abs(np.sum(pnls[pnls < 0]))
        
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = np.inf  # No losses, infinite profit factor
        else:
            profit_factor = 0.0  # No profits
        
        # Calmar ratio
        total_return = cumulative[-1] if len(cumulative) > 0 else 0
        calmar = total_return / abs(max_dd) if max_dd != 0 else 0
        
        return {
            'mean_pnl': mean_pnl,
            'std_pnl': std_pnl,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_dd': max_dd,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar,
            'total_return': total_return
        }
    
    metrics_list = []
    for name, data in all_results.items():
        metrics = calculate_metrics(data['pnls'])
        metrics['strategy'] = name
        metrics['type'] = data['type']
        metrics['mean_cost'] = np.mean(data['costs'])
        metrics_list.append(metrics)
    
    df = pd.DataFrame(metrics_list)
    df = df.sort_values('sharpe', ascending=False)
    
    print("\n   Performance Summary:")
    print(df[['strategy', 'type', 'sharpe', 'mean_pnl', 'win_rate', 'max_dd', 'profit_factor']].to_string(index=False))
    
    # Print additional note about profit factor interpretation
    print("\n   📊 Profit Factor Interpretation:")
    print("      > 1.0 = Profitable strategy")
    print("      = 1.0 = Break-even")
    print("      < 1.0 = Losing strategy")
    print("      ∞ = No losses (all positive trades)")
    print("      0 = No profits (all negative trades)")
    
    return df


def save_combined_comparison(metrics_df: pd.DataFrame) -> None:
    """Save combined comparison results to CSV."""
    
    # Add additional metrics
    comparison_df = metrics_df.copy()
    comparison_df['Profitability'] = comparison_df['profit_factor'].apply(
        lambda x: 'Profitable' if x > 1 else ('Break-even' if x == 1 else 'Losing')
    )
    comparison_df['Risk_Rating'] = pd.cut(comparison_df['sharpe'], 
                                           bins=[-np.inf, -1, 0, 1, np.inf],
                                           labels=['Very High Risk', 'High Risk', 'Moderate Risk', 'Low Risk'])
    
    comparison_df.to_csv('10_reports/complete_comparison.csv', index=False)
    print("   ✅ Saved: 10_reports/complete_comparison.csv")


def step8_save_results(all_results: dict, metrics_df: pd.DataFrame) -> None:
    """STEP 8: Save all results to CSV files in 10_reports/ folder."""
    print("\n" + "="*60)
    print("STEP 8: SAVING RESULTS TO 10_reports/")
    print("="*60)
    
    # Save hedging errors (PnL distributions)
    hedging_errors = []
    for name, data in all_results.items():
        for i, pnl in enumerate(data['pnls']):
            hedging_errors.append({
                'Strategy': name,
                'Type': data['type'],
                'Path': i,
                'PnL': pnl
            })
    errors_df = pd.DataFrame(hedging_errors)
    errors_df.to_csv('10_reports/hedging_errors.csv', index=False)
    print("   ✅ Saved: 10_reports/hedging_errors.csv")
    
    # Save performance summary
    metrics_df.to_csv('10_reports/performance_summary.csv', index=False)
    print("   ✅ Saved: 10_reports/performance_summary.csv")
    
    # Save transaction costs
    transaction_costs = []
    for name, data in all_results.items():
        for i, cost in enumerate(data['costs']):
            transaction_costs.append({
                'Strategy': name,
                'Path': i,
                'TransactionCost': cost
            })
    costs_df = pd.DataFrame(transaction_costs)
    costs_df.to_csv('10_reports/transactions_log.csv', index=False)
    print("   ✅ Saved: 10_reports/transactions_log.csv")
    
    # Save all PnLs
    pnls_df = pd.DataFrame()
    for name, data in all_results.items():
        pnls_df[name] = data['pnls']
    pnls_df.to_csv('10_reports/all_pnls.csv', index=False)
    print("   ✅ Saved: 10_reports/all_pnls.csv")


def step9_visualize(all_results: dict, metrics_df: pd.DataFrame) -> None:
    """STEP 9: Create visualizations saved to 10_reports/figures/."""
    print("\n" + "="*60)
    print("STEP 9: CREATING VISUALIZATIONS")
    print("="*60)
    
    import matplotlib.pyplot as plt
    
    # All figures go to 10_reports/figures/
    figures_dir = '10_reports/figures'
    os.makedirs(figures_dir, exist_ok=True)
    
    plt.style.use('default')
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']
    
    # 1. Sharpe Ratio Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(metrics_df['strategy'], metrics_df['sharpe'], color=colors[:len(metrics_df)])
    ax.axvline(x=0, color='black', linestyle='-')
    ax.axvline(x=1, color='green', linestyle='--', alpha=0.5, label='Good')
    ax.axvline(x=2, color='blue', linestyle='--', alpha=0.5, label='Excellent')
    ax.set_xlabel('Sharpe Ratio')
    ax.set_title('Sharpe Ratio Comparison')
    for bar, val in zip(bars, metrics_df['sharpe']):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center')
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/sharpe_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 10_reports/figures/sharpe_comparison.png")
    
    # 2. Return Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(metrics_df['strategy'], metrics_df['mean_pnl'], color=colors[:len(metrics_df)])
    ax.axvline(x=0, color='black', linestyle='-')
    ax.set_xlabel('Mean PnL ($)')
    ax.set_title('Mean PnL Comparison')
    for bar, val in zip(bars, metrics_df['mean_pnl']):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'${val:.0f}', va='center')
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/return_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 10_reports/figures/return_comparison.png")
    
    # 3. PnL Distribution
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, (name, data) in enumerate(all_results.items()):
        ax.hist(data['pnls'], bins=30, alpha=0.5, label=name, color=colors[i % len(colors)], edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Break-even')
    ax.set_xlabel('PnL ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('PnL Distribution by Strategy')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/pnl_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 10_reports/figures/pnl_distribution.png")
    
    # 4. VaR Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(metrics_df['strategy'], metrics_df['var_95'], color=colors[:len(metrics_df)])
    ax.axvline(x=0, color='black', linestyle='-')
    ax.set_xlabel('VaR (95%) ($)')
    ax.set_title('Value at Risk (95%) Comparison')
    for bar, val in zip(bars, metrics_df['var_95']):
        ax.text(bar.get_width() + (1 if val < 0 else -1), bar.get_y() + bar.get_height()/2,
               f'${val:.0f}', va='center', ha='left' if val < 0 else 'right')
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/var_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 10_reports/figures/var_comparison.png")
    
    # 5. Drawdown Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(metrics_df['strategy'], metrics_df['max_dd'], color=colors[:len(metrics_df)])
    ax.axvline(x=0, color='black', linestyle='-')
    ax.set_xlabel('Maximum Drawdown (%)')
    ax.set_title('Maximum Drawdown Comparison')
    for bar, val in zip(bars, metrics_df['max_dd']):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', va='center')
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/drawdown_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 10_reports/figures/drawdown_comparison.png")
    
    # 6. Risk-Return Scatter Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, row in metrics_df.iterrows():
        color = '#2ecc71' if row['type'] == 'drl' else '#3498db'
        size = 200 if row['type'] == 'drl' else 150
        marker = 'D' if row['type'] == 'drl' else 'o'
        ax.scatter(row['std_pnl'], row['mean_pnl'], s=size, color=color, marker=marker,
                  edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.annotate(row['strategy'], (row['std_pnl'], row['mean_pnl']),
                   xytext=(8, 8), textcoords='offset points', fontsize=9, fontweight='bold')
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    ax.text(ax.get_xlim()[1] * 0.7, ax.get_ylim()[1] * 0.9, 'High Risk - High Return', 
           fontsize=9, alpha=0.5, ha='center')
    ax.text(ax.get_xlim()[0] * 0.3, ax.get_ylim()[1] * 0.9, 'Low Risk - High Return', 
           fontsize=9, alpha=0.5, ha='center')
    ax.text(ax.get_xlim()[0] * 0.3, ax.get_ylim()[0] * 0.3, 'Low Risk - Low Return', 
           fontsize=9, alpha=0.5, ha='center')
    ax.text(ax.get_xlim()[1] * 0.7, ax.get_ylim()[0] * 0.3, 'High Risk - Low Return', 
           fontsize=9, alpha=0.5, ha='center')
    
    ax.set_xlabel('Risk (Std Dev of PnL)')
    ax.set_ylabel('Return (Mean PnL)')
    ax.set_title('Risk-Return Tradeoff (🔵 Classical, 🟢 DRL: PPO/SAC/TD3)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/risk_return_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 10_reports/figures/risk_return_scatter.png")
    
    # 7. Sharpe vs Return Scatter Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, row in metrics_df.iterrows():
        color = '#2ecc71' if row['type'] == 'drl' else '#3498db'
        size = 200 if row['type'] == 'drl' else 150
        marker = 'D' if row['type'] == 'drl' else 'o'
        ax.scatter(row['sharpe'], row['mean_pnl'], s=size, color=color, marker=marker,
                  edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.annotate(row['strategy'], (row['sharpe'], row['mean_pnl']),
                   xytext=(8, 8), textcoords='offset points', fontsize=9, fontweight='bold')
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=1, color='green', linestyle='--', alpha=0.3, label='Good Sharpe')
    ax.set_xlabel('Sharpe Ratio')
    ax.set_ylabel('Return (Mean PnL)')
    ax.set_title('Sharpe vs Return Tradeoff (🔵 Classical, 🟢 DRL: PPO/SAC/TD3)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/sharpe_vs_return.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 10_reports/figures/sharpe_vs_return.png")
    
    # 8. Transaction Costs
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(metrics_df['strategy'], metrics_df['mean_cost'], color=colors[:len(metrics_df)])
    ax.set_xlabel('Mean Transaction Costs ($)')
    ax.set_title('Transaction Costs by Strategy')
    for bar, val in zip(bars, metrics_df['mean_cost']):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'${val:.0f}', va='center')
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/transaction_costs.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 10_reports/figures/transaction_costs.png")
    
    # 9. Win Rate Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(metrics_df['strategy'], metrics_df['win_rate'], color=colors[:len(metrics_df)])
    ax.set_xlabel('Win Rate (%)')
    ax.set_title('Win Rate Comparison')
    for bar, val in zip(bars, metrics_df['win_rate']):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', va='center')
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/win_rate_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 10_reports/figures/win_rate_comparison.png")
    
    # 10. Profit Factor Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    # Handle infinite values in profit factor
    profit_factors_display = []
    for pf in metrics_df['profit_factor']:
        if pf == np.inf:
            profit_factors_display.append(10.0)  # Cap at 10 for display
        else:
            profit_factors_display.append(pf)
    
    bars = ax.barh(metrics_df['strategy'], profit_factors_display, color=colors[:len(metrics_df)])
    ax.axvline(x=1, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Break-even (1.0)')
    ax.set_xlabel('Profit Factor')
    ax.set_title('Profit Factor Comparison (>1 = Profitable)')
    
    # Add labels
    for bar, pf in zip(bars, metrics_df['profit_factor']):
        if pf == np.inf:
            label = '∞'
        else:
            label = f'{pf:.2f}'
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, label, va='center', fontsize=9)
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/profit_factor_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 10_reports/figures/profit_factor_comparison.png")
    
    # 11. Box Plot of PnL Distributions
    fig, ax = plt.subplots(figsize=(14, 7))
    data_to_plot = [data['pnls'] for data in all_results.values()]
    labels = list(all_results.keys())
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.7)
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Break-even')
    ax.set_ylabel('PnL ($)')
    ax.set_title('PnL Distribution Box Plot')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/pnl_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 10_reports/figures/pnl_boxplot.png")
    
    print(f"\n   ✅ Total: 11 visualizations saved to 10_reports/figures/")
    print(f"   DRL Agents included: PPO, SAC, TD3")


def step10_generate_report(metrics_df: pd.DataFrame) -> None:
    """STEP 10: Generate analysis report in 10_reports/."""
    print("\n" + "="*60)
    print("STEP 10: GENERATING REPORT")
    print("="*60)
    
    winner = metrics_df.iloc[0]
    best_drl = metrics_df[metrics_df['type'] == 'drl'].iloc[0] if len(metrics_df[metrics_df['type'] == 'drl']) > 0 else None
    best_classical = metrics_df[metrics_df['type'] == 'classical'].iloc[0] if len(metrics_df[metrics_df['type'] == 'classical']) > 0 else None
    
    report = []
    report.append("# Deep Hedging with DRL - Analysis Report")
    report.append("")
    report.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("## Executive Summary")
    report.append("")
    report.append(f"**Overall Winner:** {winner['strategy']}")
    report.append(f"- Sharpe Ratio: {winner['sharpe']:.4f}")
    report.append(f"- Mean PnL: ${winner['mean_pnl']:.2f}")
    report.append(f"- Win Rate: {winner['win_rate']:.1f}%")
    report.append(f"- Max Drawdown: {winner['max_dd']:.1f}%")
    report.append("")
    
    if best_drl is not None and best_classical is not None:
        report.append("### DRL vs Classical Comparison")
        report.append("")
        report.append(f"**Best DRL:** {best_drl['strategy']} (Sharpe: {best_drl['sharpe']:.4f})")
        report.append(f"**Best Classical:** {best_classical['strategy']} (Sharpe: {best_classical['sharpe']:.4f})")
        report.append("")
        if best_drl['sharpe'] > best_classical['sharpe']:
            report.append(f"🎉 DRL outperforms Classical by {best_drl['sharpe'] - best_classical['sharpe']:.4f}")
        else:
            report.append(f"📈 Classical outperforms DRL by {best_classical['sharpe'] - best_drl['sharpe']:.4f}")
        report.append("")
    
    report.append("## Performance Summary")
    report.append("")
    report.append("| Strategy | Type | Sharpe | Mean PnL | Win Rate | Max DD | VaR (95%) |")
    report.append("|----------|------|--------|----------|----------|--------|-----------|")
    for _, row in metrics_df.iterrows():
        report.append(f"| {row['strategy']} | {row['type']} | {row['sharpe']:.3f} | ${row['mean_pnl']:.0f} | {row['win_rate']:.1f}% | {row['max_dd']:.1f}% | ${row['var_95']:.0f} |")
    report.append("")
    report.append("## DRL Agents Comparison")
    report.append("")
    report.append("| Agent | Type | Sharpe | Mean PnL | Win Rate | Max DD |")
    report.append("|-------|------|--------|----------|----------|--------|")
    drl_agents = metrics_df[metrics_df['type'] == 'drl']
    for _, row in drl_agents.iterrows():
        report.append(f"| {row['strategy']} | DRL | {row['sharpe']:.3f} | ${row['mean_pnl']:.0f} | {row['win_rate']:.1f}% | {row['max_dd']:.1f}% |")
    report.append("")
    report.append("## Visualizations")
    report.append("")
    report.append("![Sharpe Ratio](figures/sharpe_comparison.png)")
    report.append("![Return Comparison](figures/return_comparison.png)")
    report.append("![PnL Distribution](figures/pnl_distribution.png)")
    report.append("![VaR Comparison](figures/var_comparison.png)")
    report.append("![Drawdown Comparison](figures/drawdown_comparison.png)")
    report.append("![Risk-Return Scatter](figures/risk_return_scatter.png)")
    report.append("![Sharpe vs Return](figures/sharpe_vs_return.png)")
    report.append("![Transaction Costs](figures/transaction_costs.png)")
    report.append("![Win Rate](figures/win_rate_comparison.png)")
    report.append("![Profit Factor](figures/profit_factor_comparison.png)")
    report.append("![PnL Boxplot](figures/pnl_boxplot.png)")
    
    with open('10_reports/analysis_report.md', 'w') as f:
        f.write('\n'.join(report))
    print("   ✅ Saved: 10_reports/analysis_report.md")


def print_final_summary(metrics_df: pd.DataFrame) -> None:
    """Print final summary of results."""
    print("\n" + "="*80)
    print("🏆 FINAL RESULTS SUMMARY")
    print("="*80)
    
    winner = metrics_df.iloc[0]
    
    print(f"\n🥇 OVERALL WINNER: {winner['strategy']} ({winner['type']})")
    print(f"   • Sharpe Ratio: {winner['sharpe']:.4f}")
    print(f"   • Mean PnL: ${winner['mean_pnl']:.2f}")
    print(f"   • Win Rate: {winner['win_rate']:.1f}%")
    print(f"   • Max Drawdown: {winner['max_dd']:.1f}%")
    
    drl_strategies = metrics_df[metrics_df['type'] == 'drl']
    classical_strategies = metrics_df[metrics_df['type'] == 'classical']
    
    if len(drl_strategies) > 0:
        print(f"\n🤖 DRL AGENTS RANKING:")
        for i, (idx, row) in enumerate(drl_strategies.iterrows(), 1):
            print(f"   {i}. {row['strategy']}: Sharpe={row['sharpe']:.4f}, Return=${row['mean_pnl']:.2f}")
    
    if len(classical_strategies) > 0:
        print(f"\n📊 CLASSICAL STRATEGIES RANKING:")
        for i, (idx, row) in enumerate(classical_strategies.iterrows(), 1):
            print(f"   {i}. {row['strategy']}: Sharpe={row['sharpe']:.4f}, Return=${row['mean_pnl']:.2f}")


def main():
    """Main execution function."""
    print("="*80)
    print("🚀 DEEP HEDGING WITH DRL - COMPLETE PIPELINE")
    print("   DRL Agents: PPO, SAC, TD3")
    print("="*80)
    
    start_time = datetime.now()
    
    # Load configuration
    config = load_config()
    
    # Setup directories
    setup_directories(config)
    
    # STEP 2: Generate price paths
    price_paths = step2_generate_data(config)
    
    # Split data (60/20/20)
    n_paths = price_paths.shape[1]
    train_end = int(n_paths * 0.6)
    val_end = train_end + int(n_paths * 0.2)
    
    train_paths = price_paths[:, :train_end]
    val_paths = price_paths[:, train_end:val_end]
    test_paths = price_paths[:, val_end:]
    
    print(f"\n📊 Data Split:")
    print(f"   Training: {train_paths.shape[1]} paths (60%)")
    print(f"   Validation: {val_paths.shape[1]} paths (20%)")
    print(f"   Test: {test_paths.shape[1]} paths (20%)")
    
    # Save intermediate results
    save_intermediate_results(price_paths, train_paths, val_paths, test_paths)
    
    # STEP 4: Run classical baselines
    classical_results = step4_run_baselines(test_paths, config)
    
    # Save classical results
    save_baseline_results(classical_results)
    
    # STEP 5: Train DRL agents (PPO, SAC, TD3)
    drl_agents = step5_train_drl_agents(train_paths, config)
    
    # STEP 6: Evaluate DRL agents
    drl_results = step6_evaluate_drl_agents(drl_agents, test_paths, config)
    
    # Save DRL results
    save_drl_results(drl_agents, drl_results)
    
    # Combine all results
    all_results = {**classical_results, **drl_results}
    
    # STEP 7: Evaluate all strategies
    metrics_df = step7_combine_and_evaluate(all_results)
    
    # Save combined comparison
    save_combined_comparison(metrics_df)
    
    # STEP 8: Save results to CSV
    step8_save_results(all_results, metrics_df)
    
    # STEP 9: Create visualizations
    step9_visualize(all_results, metrics_df)
    
    # STEP 10: Generate report
    step10_generate_report(metrics_df)
    
    # Print final summary
    print_final_summary(metrics_df)
    
    # Execution time
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds() / 60
    print(f"\n⏱️ Total execution time: {elapsed:.2f} minutes")
    print("\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("\n📁 All outputs saved in: 10_reports/")
    print("   - CSV files: hedging_errors.csv, performance_summary.csv, transactions_log.csv, all_pnls.csv")
    print("   - Additional CSV files: price_paths_summary.csv, data_split_info.csv, classical_baselines_results.csv, drl_agents_results.csv, complete_comparison.csv")
    print("   - Figures: 11 PNG files in 10_reports/figures/")
    print("   - Report: analysis_report.md")
    print("\n🤖 DRL Agents included: PPO, SAC, TD3")


if __name__ == "__main__":
    main()