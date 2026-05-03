# -*- coding: utf-8 -*-
"""
================================================================================
RESULTS ANALYZER - ANALYZE STORED RESULTS
================================================================================

This module provides tools for analyzing stored backtest results:
    - Summary statistics
    - Performance ranking
    - Distribution analysis
    - Comparative visualization data
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class ResultsAnalyzer:
    """
    Analyzer for backtest results.
    
    Provides methods to:
        - Generate summary statistics
        - Rank strategies by performance
        - Analyze PnL distributions
        - Compare strategy performance
    """
    
    def __init__(self, results_df: Optional[pd.DataFrame] = None):
        """
        Initialize results analyzer.
        
        Args:
            results_df: DataFrame with results (optional)
        """
        self.results_df = results_df
        self.summary = None
    
    def load_results(self, filepath: str) -> pd.DataFrame:
        """
        Load results from CSV file.
        
        Args:
            filepath: Path to CSV file
        
        Returns:
            DataFrame with results
        """
        self.results_df = pd.read_csv(filepath)
        print(f"Loaded results: {filepath}")
        print(f"{len(self.results_df)} records")
        return self.results_df
    
    def get_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of results.
        
        Returns:
            DataFrame with summary statistics
        """
        if self.results_df is None:
            raise ValueError("No results loaded. Call load_results() first.")
        
        # Calculate summary statistics
        summary_list = []
        
        for strategy in self.results_df['Strategy'].unique():
            pnls = self.results_df[self.results_df['Strategy'] == strategy]['PnL'].values
            
            summary_list.append({
                'Strategy': strategy,
                'Mean': np.mean(pnls),
                'Std': np.std(pnls),
                'Min': np.min(pnls),
                'Max': np.max(pnls),
                'Count': len(pnls),
                'Win_Rate': (pnls > 0).sum() / len(pnls) * 100
            })
        
        summary = pd.DataFrame(summary_list)
        summary = summary.sort_values('Mean', ascending=False).reset_index(drop=True)
        
        self.summary = summary
        return summary
    
    def get_ranking(self, metric: str = 'Mean') -> pd.DataFrame:
        """
        Get ranking of strategies by a specific metric.
        
        Args:
            metric: Metric to rank by ('Mean', 'Sharpe', 'Win_Rate', etc.)
        
        Returns:
            DataFrame with ranking
        """
        if self.results_df is None:
            raise ValueError("No results loaded. Call load_results() first.")
        
        ranking_list = []
        
        for strategy in self.results_df['Strategy'].unique():
            pnls = self.results_df[self.results_df['Strategy'] == strategy]['PnL'].values
            
            if metric == 'Sharpe':
                value = np.mean(pnls) / np.std(pnls) * np.sqrt(252) if np.std(pnls) > 0 else 0
            elif metric == 'Win_Rate':
                value = (pnls > 0).sum() / len(pnls) * 100
            elif metric == 'Mean':
                value = np.mean(pnls)
            elif metric == 'Std':
                value = np.std(pnls)
            elif metric == 'VaR':
                value = np.percentile(pnls, 5)
            else:
                value = np.mean(pnls)
            
            ranking_list.append({
                'Strategy': strategy,
                'Value': value,
                'Rank': 0
            })
        
        ranking = pd.DataFrame(ranking_list)
        ranking['Rank'] = ranking['Value'].rank(ascending=False).astype(int)
        ranking = ranking.sort_values('Rank').reset_index(drop=True)
        
        return ranking
    
    def get_distribution_stats(self) -> pd.DataFrame:
        """
        Get distribution statistics for each strategy.
        
        Returns:
            DataFrame with distribution statistics
        """
        if self.results_df is None:
            raise ValueError("No results loaded. Call load_results() first.")
        
        stats_list = []
        
        for name in self.results_df['Strategy'].unique():
            pnls = self.results_df[self.results_df['Strategy'] == name]['PnL'].values
            
            stats_list.append({
                'Strategy': name,
                'Mean': np.mean(pnls),
                'Median': np.median(pnls),
                'Std': np.std(pnls),
                'Skewness': pd.Series(pnls).skew(),
                'Kurtosis': pd.Series(pnls).kurtosis(),
                'VaR_95': np.percentile(pnls, 5),
                'CVaR_95': np.mean(pnls[pnls <= np.percentile(pnls, 5)]) if len(pnls[pnls <= np.percentile(pnls, 5)]) > 0 else np.percentile(pnls, 5),
                'Max_Loss': np.min(pnls),
                'Max_Gain': np.max(pnls)
            })
        
        return pd.DataFrame(stats_list).sort_values('Mean', ascending=False).reset_index(drop=True)
    
    def get_worst_performers(self, n: int = 5) -> pd.DataFrame:
        """
        Get worst performing paths for each strategy.
        
        Args:
            n: Number of worst performers to return
        
        Returns:
            DataFrame with worst performers
        """
        if self.results_df is None:
            raise ValueError("No results loaded. Call load_results() first.")
        
        worst = self.results_df.groupby('Strategy').apply(
            lambda x: x.nsmallest(n, 'PnL')
        ).reset_index(drop=True)
        
        return worst
    
    def get_best_performers(self, n: int = 5) -> pd.DataFrame:
        """
        Get best performing paths for each strategy.
        
        Args:
            n: Number of best performers to return
        
        Returns:
            DataFrame with best performers
        """
        if self.results_df is None:
            raise ValueError("No results loaded. Call load_results() first.")
        
        best = self.results_df.groupby('Strategy').apply(
            lambda x: x.nlargest(n, 'PnL')
        ).reset_index(drop=True)
        
        return best
    
    def compare_strategies(self, strategy1: str, strategy2: str) -> Dict:
        """
        Compare two strategies in detail.
        
        Args:
            strategy1: Name of first strategy
            strategy2: Name of second strategy
        
        Returns:
            Dictionary with comparison results
        """
        if self.results_df is None:
            raise ValueError("No results loaded. Call load_results() first.")
        
        pnls1 = self.results_df[self.results_df['Strategy'] == strategy1]['PnL'].values
        pnls2 = self.results_df[self.results_df['Strategy'] == strategy2]['PnL'].values
        
        if len(pnls1) == 0 or len(pnls2) == 0:
            return {'error': 'Strategy not found'}
        
        from scipy import stats
        
        # Statistical tests
        t_stat, p_value = stats.ttest_ind(pnls1, pnls2)
        
        return {
            'strategy1': strategy1,
            'strategy2': strategy2,
            'mean1': np.mean(pnls1),
            'mean2': np.mean(pnls2),
            'mean_difference': np.mean(pnls1) - np.mean(pnls2),
            'std1': np.std(pnls1),
            'std2': np.std(pnls2),
            'win_rate1': (pnls1 > 0).sum() / len(pnls1) * 100,
            'win_rate2': (pnls2 > 0).sum() / len(pnls2) * 100,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'better_strategy': strategy1 if np.mean(pnls1) > np.mean(pnls2) else strategy2,
            'conclusion': f"{strategy1} {'significantly outperforms' if p_value < 0.05 and np.mean(pnls1) > np.mean(pnls2) else 'does not significantly outperform'} {strategy2} (p={p_value:.4f})"
        }
    
    def generate_report(self) -> str:
        """
        Generate a text report of the results.
        
        Returns:
            String with the report
        """
        if self.results_df is None:
            raise ValueError("No results loaded. Call load_results() first.")
        
        summary = self.get_summary()
        distribution = self.get_distribution_stats()
        
        report = []
        report.append("="*80)
        report.append("RESULTS ANALYSIS REPORT")
        report.append("="*80)
        report.append("")
        
        report.append("PERFORMANCE SUMMARY")
        report.append("-"*40)
        for idx, row in summary.iterrows():
            report.append(f"{idx+1}. {row['Strategy']}:")
            report.append(f"Mean PnL: ${row['Mean']:.2f}")
            report.append(f"Std Dev: ${row['Std']:.2f}")
            report.append(f"Win Rate: {row['Win_Rate']:.1f}%")
            report.append("")
        
        report.append("DISTRIBUTION STATISTICS")
        report.append("-"*40)
        for _, row in distribution.iterrows():
            report.append(f"{row['Strategy']}:")
            report.append(f"Mean: ${row['Mean']:.2f}, Median: ${row['Median']:.2f}")
            report.append(f"VaR(95%): ${row['VaR_95']:.2f}, CVaR: ${row['CVaR_95']:.2f}")
            report.append(f"Skewness: {row['Skewness']:.3f}, Kurtosis: {row['Kurtosis']:.3f}")
            report.append("")
        
        # Best and worst
        best_strategy = summary.iloc[0]['Strategy']
        worst_strategy = summary.iloc[-1]['Strategy']
        
        report.append("BEST STRATEGY")
        report.append("-"*40)
        report.append(f"{best_strategy}: ${summary.iloc[0]['Mean']:.2f} mean PnL")
        report.append("")
        
        report.append("WORST STRATEGY")
        report.append("-"*40)
        report.append(f"{worst_strategy}: ${summary.iloc[-1]['Mean']:.2f} mean PnL")
        report.append("")
        
        report.append("="*80)
        
        return "\n".join(report)
    
    def save_report(self, filepath: str) -> None:
        """
        Save report to text file.
        
        Args:
            filepath: Path to save the report
        """
        report = self.generate_report()
        with open(filepath, 'w') as f:
            f.write(report)
        print(f"Report saved to {filepath}")


if __name__ == "__main__":
    print("="*70)
    print("RESULTS ANALYZER TEST")
    print("="*70)
    
    # Create sample data
    np.random.seed(42)
    sample_data = []
    
    strategies = ['Black-Scholes', 'Delta-Gamma', 'DRL_PPO', 'DRL_SAC', 'DRL_TD3']
    for strategy in strategies:
        for i in range(100):
            if 'DRL' in strategy:
                pnl = np.random.normal(80, 90)
            else:
                pnl = np.random.normal(50, 100)
            sample_data.append({'Strategy': strategy, 'Path': i, 'PnL': pnl})
    
    df = pd.DataFrame(sample_data)
    
    # Initialize analyzer
    analyzer = ResultsAnalyzer(df)
    
    # Get summary
    print("\nSummary:")
    print(analyzer.get_summary())
    
    # Get ranking
    print("\nRanking by Sharpe:")
    print(analyzer.get_ranking(metric='Sharpe'))
    
    # Compare strategies
    print("\nComparison:")
    comparison = analyzer.compare_strategies('DRL_PPO', 'Black-Scholes')
    print(f"{comparison['conclusion']}")
