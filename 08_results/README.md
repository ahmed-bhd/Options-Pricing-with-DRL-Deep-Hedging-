# 📁 08_results - Results Storage and Analysis

## Overview

This module handles storing, loading, and analyzing backtest results.

## Generated CSV Files

| File | Description |
|------|-------------|
| `hedging_errors.csv` | PnL for each strategy and path |
| `performance_summary.csv` | Performance metrics table |
| `transactions_log.csv` | Detailed trading activity |

## Usage Example

```python
from src.results import ResultsManager, ResultsAnalyzer

# Save results
manager = ResultsManager()
manager.save_all_results(backtest_results, comparison_df)

# Load and analyze
analyzer = ResultsAnalyzer()
analyzer.load_results("08_results/hedging_errors.csv")
summary = analyzer.get_summary()
ranking = analyzer.get_ranking(metric='Sharpe')
report = analyzer.generate_report()