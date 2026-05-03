# Deep Hedging with DRL - Analysis Report

**Date:** 2026-05-03 23:52:20

---

##Executive Summary

This report presents the results of a comprehensive comparison between 
classical hedging strategies (Black-Scholes Delta, Delta-Gamma) and 
Deep Reinforcement Learning agents (PPO, SAC, TD3) for European call option hedging.

### Overall Winner: **DRL_PPO**

- **Sharpe Ratio:** 13.697
- **Mean PnL:** $72.72
- **Win Rate:** 80.5%
- **Max Drawdown:** -9.11%

---

##Performance Summary

| Rank | Strategy | Sharpe | Return ($) | Win Rate | Max DD |
|------|----------|--------|------------|----------|--------|
| 3 | DRL_PPO | 13.697 | $73 | 80.5% | -9.1% |
| 2 | Delta-Gamma | 12.134 | $68 | 77.0% | -17.2% |
| 4 | DRL_SAC | 11.641 | $71 | 76.5% | -18.1% |
| 1 | Black-Scholes | 7.850 | $46 | 68.5% | -36.1% |

---

## Best Performing Strategies

| Rank | Strategy | Sharpe | Return ($) | Win Rate | Max DD |
|------|----------|--------|------------|----------|--------|
| 3 | DRL_PPO | 13.697 | $73 | 80.5% | -9.1% |
| 2 | Delta-Gamma | 12.134 | $68 | 77.0% | -17.2% |
| 4 | DRL_SAC | 11.641 | $71 | 76.5% | -18.1% |

## Worst Performing Strategies

| Rank | Strategy | Sharpe | Return ($) | Win Rate | Max DD |
|------|----------|--------|------------|----------|--------|
| 3 | Delta-Gamma | 12.134 | $68 | 77.0% | -17.2% |
| 1 | DRL_SAC | 11.641 | $71 | 76.5% | -18.1% |
| 4 | Black-Scholes | 7.850 | $46 | 68.5% | -36.1% |

---

## Risk Metrics Analysis

| Strategy | VaR (95%) | CVaR (95%) | VaR (99%) | CVaR (99%) | Sortino |
|----------|-----------|------------|-----------|------------|---------|
| DRL_PPO | $-62 | $-92 | $-108 | $-123 | 32.982 |
| Delta-Gamma | $-69 | $-110 | $-122 | $-181 | 24.113 |
| DRL_SAC | $-87 | $-119 | $-128 | $-184 | 27.017 |
| Black-Scholes | $-98 | $-136 | $-146 | $-180 | 16.535 |

---

## DRL vs Classical Comparison

### Best DRL vs Best Classical

| Metric | Best DRL | Best Classical | Difference |
|--------|----------|----------------|------------|
| **Strategy** | DRL_PPO | Delta-Gamma | - |
| **Sharpe Ratio** | 13.697 | 12.134 | +1.563 |
| **Mean PnL** | $73 | $68 | $+5 |
| **Win Rate** | 80.5% | 77.0% | +3.5% |
| **Max Drawdown** | -9.1% | -17.2% | +8.1% |

---

## Transaction Cost Analysis

| Strategy | Transaction Costs ($) | Profit Factor | Cost Efficiency |
|----------|---------------------|---------------|-----------------|
| DRL_PPO | $0 | 9.50 | ∞ |
| Delta-Gamma | $0 | 7.66 | ∞ |
| DRL_SAC | $0 | 6.32 | ∞ |
| Black-Scholes | $0 | 3.46 | ∞ |

---

## Conclusions and Recommendations

### Key Findings

1. **Best Overall Strategy:** DRL_PPO achieved the highest Sharpe ratio (13.697)
2. **Classical vs DRL:** Delta-Gamma outperformed in terms of Sharpe ratio
3. **Risk Management:** Strategies with lower transaction costs generally performed better
4. **Drawdown Control:** DRL agents showed different drawdown characteristics

### Recommendations

1. **For production deployment:** Consider the top-performing strategy based on your risk tolerance
2. **For further improvement:**
   - Increase training timesteps for DRL agents
   - Add more market regime features
   - Implement ensemble methods combining multiple strategies

---

## Appendix: Complete Metrics

```
     Strategy  Mean PnL ($)  Std PnL ($)  Sharpe Ratio  Sortino Ratio  Calmar Ratio  Win Rate (%)  VaR (95%) ($)  CVaR (95%) ($)  VaR (99%) ($)  CVaR (99%) ($)  Max Drawdown (%)  Transaction Costs ($)  Profit Factor
      DRL_PPO       72.7193      84.2821       13.6967        32.9819        7.9804          80.5       -62.0439        -92.2720      -108.0237       -122.8765           -9.1122                      0         9.4979
  Delta-Gamma       67.7281      88.6080       12.1338        24.1130        3.9459          77.0       -68.7385       -110.4494      -122.3517       -181.4323          -17.1644                      0         7.6602
      DRL_SAC       70.8518      96.6171       11.6412        27.0169        3.9169          76.5       -87.2404       -118.9450      -128.3239       -184.0232          -18.0889                      0         6.3161
Black-Scholes       45.9229      92.8673        7.8499        16.5348        1.2709          68.5       -98.0338       -135.5999      -145.9949       -180.3657          -36.1337                      0         3.4644
```

## Visualizations

### Portfolio Value Over Time

![Portfolio Values](figures/portfolio_values.png)

### Sharpe Ratio Comparison

![Sharpe Ratio](figures/sharpe_ratio.png)

### VaR Comparison

![VaR Comparison](figures/var_comparison.png)

### Risk-Return Tradeoff

![Risk-Return](figures/risk_return_scatter.png)

---

*Report generated on 2026-05-03 23:52:20*

---