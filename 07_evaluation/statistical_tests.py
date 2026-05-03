# -*- coding: utf-8 -*-
"""
================================================================================
STATISTICAL TESTS FOR HEDGING STRATEGIES
================================================================================

This module provides statistical significance tests to compare hedging strategies:
    - Paired t-test (for normally distributed differences)
    - Wilcoxon signed-rank test (non-parametric)
    - Mann-Whitney U test (independent samples)
    - Bootstrap confidence intervals
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class StatisticalTests:
    """
    Statistical significance tests for comparing hedging strategies.
    
    Tests available:
        - paired_t_test: For paired samples (same paths, different strategies)
        - wilcoxon_test: Non-parametric alternative to t-test
        - mann_whitney_test: For independent samples
        - bootstrap_ci: Confidence intervals via bootstrap
    """
    
    @staticmethod
    def paired_t_test(pnls1: np.ndarray, pnls2: np.ndarray) -> Dict:
        """
        Perform paired t-test to compare two strategies.
        
        Null hypothesis: mean difference = 0 (strategies perform equally)
        
        Args:
            pnls1: PnL array for strategy 1
            pnls2: PnL array for strategy 2
        
        Returns:
            Dictionary with t-statistic, p-value, and conclusion
        """
        diff = pnls1 - pnls2
        
        if len(diff) < 2 or np.std(diff) == 0:
            return {
                't_statistic': 0,
                'p_value': 1.0,
                'significant': False,
                'conclusion': "Insufficient data"
            }
        
        t_stat, p_value = stats.ttest_rel(pnls1, pnls2)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'conclusion': f"Strategy 1 {'performs differently' if p_value < 0.05 else 'performs similarly'} to Strategy 2 (p={p_value:.4f})"
        }
    
    @staticmethod
    def wilcoxon_test(pnls1: np.ndarray, pnls2: np.ndarray) -> Dict:
        """
        Perform Wilcoxon signed-rank test (non-parametric).
        
        Use when differences are not normally distributed.
        
        Args:
            pnls1: PnL array for strategy 1
            pnls2: PnL array for strategy 2
        
        Returns:
            Dictionary with statistic, p-value, and conclusion
        """
        diff = pnls1 - pnls2
        
        if len(diff) < 2:
            return {
                'statistic': 0,
                'p_value': 1.0,
                'significant': False,
                'conclusion': "Insufficient data"
            }
        
        try:
            statistic, p_value = stats.wilcoxon(diff)
        except:
            return {
                'statistic': 0,
                'p_value': 1.0,
                'significant': False,
                'conclusion': "Test failed"
            }
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'conclusion': f"Strategies {'significantly differ' if p_value < 0.05 else 'do not significantly differ'} (p={p_value:.4f})"
        }
    
    @staticmethod
    def mann_whitney_test(pnls1: np.ndarray, pnls2: np.ndarray) -> Dict:
        """
        Perform Mann-Whitney U test (independent samples).
        
        Args:
            pnls1: PnL array for strategy 1
            pnls2: PnL array for strategy 2
        
        Returns:
            Dictionary with statistic, p-value, and conclusion
        """
        if len(pnls1) == 0 or len(pnls2) == 0:
            return {
                'statistic': 0,
                'p_value': 1.0,
                'significant': False,
                'conclusion': "Insufficient data"
            }
        
        statistic, p_value = stats.mannwhitneyu(pnls1, pnls2, alternative='two-sided')
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'conclusion': f"Strategy 1 {'outperforms' if np.mean(pnls1) > np.mean(pnls2) else 'underperforms'} Strategy 2 (p={p_value:.4f})"
        }
    
    @staticmethod
    def bootstrap_ci(pnls: np.ndarray, 
                     n_bootstrap: int = 10000, 
                     confidence: float = 0.95) -> Dict:
        """
        Calculate bootstrap confidence interval for mean PnL.
        
        Args:
            pnls: PnL array
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
        
        Returns:
            Dictionary with confidence interval
        """
        if len(pnls) == 0:
            return {'lower': 0, 'upper': 0, 'mean': 0}
        
        np.random.seed(42)
        
        bootstrap_means = []
        n = len(pnls)
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(pnls, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, alpha/2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
        
        return {
            'lower': lower,
            'upper': upper,
            'mean': np.mean(pnls),
            'std': np.std(bootstrap_means),
            'confidence': confidence
        }
    
    @staticmethod
    def compare_all_strategies(results: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Perform pairwise statistical tests between all strategies.
        
        Args:
            results: Dictionary of {strategy_name: pnls_array}
        
        Returns:
            DataFrame with pairwise comparison results
        """
        strategy_names = list(results.keys())
        n_strategies = len(strategy_names)
        
        comparisons = []
        
        for i in range(n_strategies):
            for j in range(i + 1, n_strategies):
                name1 = strategy_names[i]
                name2 = strategy_names[j]
                pnls1 = results[name1]
                pnls2 = results[name2]
                
                # Perform t-test
                ttest = StatisticalTests.paired_t_test(pnls1, pnls2)
                
                # Compute mean difference
                mean_diff = np.mean(pnls1) - np.mean(pnls2)
                
                comparisons.append({
                    'Strategy 1': name1,
                    'Strategy 2': name2,
                    'Mean Difference': mean_diff,
                    't-statistic': ttest['t_statistic'],
                    'p-value': ttest['p_value'],
                    'Significant': ttest['significant'],
                    'Better Strategy': name1 if mean_diff > 0 else name2
                })
        
        df = pd.DataFrame(comparisons)
        return df.sort_values('p-value')
    
    @staticmethod
    def compare_to_baseline(strategy_pnls: np.ndarray, 
                            baseline_pnls: np.ndarray,
                            strategy_name: str,
                            baseline_name: str = "Baseline") -> Dict:
        """
        Compare a single strategy to a baseline with multiple tests.
        
        Args:
            strategy_pnls: PnL array for the strategy
            baseline_pnls: PnL array for the baseline
            strategy_name: Name of the strategy
            baseline_name: Name of the baseline
        
        Returns:
            Dictionary with comprehensive comparison results
        """
        # Calculate mean difference
        mean_diff = np.mean(strategy_pnls) - np.mean(baseline_pnls)
        
        # Paired t-test
        ttest = StatisticalTests.paired_t_test(strategy_pnls, baseline_pnls)
        
        # Wilcoxon test
        wilcoxon = StatisticalTests.wilcoxon_test(strategy_pnls, baseline_pnls)
        
        # Bootstrap CI for difference
        diff = strategy_pnls - baseline_pnls
        bootstrap_ci = StatisticalTests.bootstrap_ci(diff)
        
        return {
            'strategy_name': strategy_name,
            'baseline_name': baseline_name,
            'mean_difference': mean_diff,
            'mean_difference_percent': (mean_diff / abs(np.mean(baseline_pnls))) * 100 if np.mean(baseline_pnls) != 0 else 0,
            't_test': ttest,
            'wilcoxon_test': wilcoxon,
            'bootstrap_ci_95': bootstrap_ci,
            'strategy_better': mean_diff > 0 and ttest['p_value'] < 0.05,
            'conclusion': f"{strategy_name} {'significantly outperforms' if mean_diff > 0 and ttest['p_value'] < 0.05 else 'does not significantly outperform'} {baseline_name}"
        }


if __name__ == "__main__":
    print("="*70)
    print("STATISTICAL TESTS TEST")
    print("="*70)
    
    # Generate sample data
    np.random.seed(42)
    baseline_pnls = np.random.normal(0, 100, 100)
    strategy_pnls = np.random.normal(20, 100, 100)  # Slightly better
    
    # Compare to baseline
    comparison = StatisticalTests.compare_to_baseline(
        strategy_pnls, baseline_pnls, 
        "DRL_PPO", "Black-Scholes"
    )
    
    print("\nComparison Results:")
    print(f"{comparison['conclusion']}")
    print(f"Mean difference: ${comparison['mean_difference']:.2f}")
    print(f"t-test p-value: {comparison['t_test']['p_value']:.4f}")
