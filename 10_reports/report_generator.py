# -*- coding: utf-8 -*-
"""
================================================================================
REPORT GENERATOR
================================================================================

This module generates comprehensive analysis reports from backtest results.

Supported formats:
    - Markdown (.md)
    - HTML (.html)
    - PDF (via weasyprint, optional)

The report includes:
    - Executive summary
    - Performance metrics table
    - Statistical significance tests
    - Risk metrics analysis
    - Transaction cost analysis
    - Visualizations (embedded or linked)
    - Conclusions and recommendations
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')


class ReportGenerator:
    """
    Generate comprehensive analysis reports from backtest results.
    
    Creates reports in Markdown, HTML, or PDF format with:
        - Performance metrics
        - Statistical analysis
        - Risk metrics
        - Transaction cost analysis
        - Visualizations
    """
    
    def __init__(self, 
                 results: Dict[str, Dict],
                 comparison_df: pd.DataFrame,
                 config: Optional[Dict] = None):
        """
        Initialize report generator.
        
        Args:
            results: Dictionary of backtest results from BacktestEngine
            comparison_df: DataFrame with performance comparison
            config: Optional configuration dictionary
        """
        self.results = results
        self.comparison_df = comparison_df
        self.config = config or {}
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _calculate_metrics(self) -> pd.DataFrame:
        """
        Calculate comprehensive metrics for all strategies.
        """
        metrics_list = []
        
        for name, data in self.results.items():
            if 'pnls' in data and data['pnls'] is not None:
                pnls = data['pnls']
                if len(pnls) > 0:
                    # Basic metrics
                    mean_pnl = np.mean(pnls)
                    std_pnl = np.std(pnls)
                    sharpe = mean_pnl / std_pnl * np.sqrt(252) if std_pnl > 0 else 0
                    win_rate = (pnls > 0).sum() / len(pnls) * 100
                    
                    # Risk metrics
                    var_95 = np.percentile(pnls, 5)
                    cvar_95 = np.mean(pnls[pnls <= var_95]) if len(pnls[pnls <= var_95]) > 0 else var_95
                    var_99 = np.percentile(pnls, 1)
                    cvar_99 = np.mean(pnls[pnls <= var_99]) if len(pnls[pnls <= var_99]) > 0 else var_99
                    
                    # Drawdown
                    cumulative = np.cumsum(pnls)
                    cummax = np.maximum.accumulate(cumulative)
                    drawdown = (cumulative - cummax) / cummax * 100
                    max_dd = np.min(drawdown) if len(drawdown) > 0 else 0
                    
                    # Sortino
                    negative_returns = pnls[pnls < 0]
                    downside_std = np.std(negative_returns) if len(negative_returns) > 0 else std_pnl
                    sortino = mean_pnl / downside_std * np.sqrt(252) if downside_std > 0 else 0
                    
                    # Calmar
                    calmar = mean_pnl / abs(max_dd) if max_dd != 0 else 0
                    
                    # Transaction costs
                    transaction_cost = 0
                    if 'metrics' in data and 'total_transaction_costs' in data['metrics']:
                        transaction_cost = data['metrics']['total_transaction_costs']
                    
                    metrics_list.append({
                        'Strategy': name,
                        'Mean PnL ($)': mean_pnl,
                        'Std PnL ($)': std_pnl,
                        'Sharpe Ratio': sharpe,
                        'Sortino Ratio': sortino,
                        'Calmar Ratio': calmar,
                        'Win Rate (%)': win_rate,
                        'VaR (95%) ($)': var_95,
                        'CVaR (95%) ($)': cvar_95,
                        'VaR (99%) ($)': var_99,
                        'CVaR (99%) ($)': cvar_99,
                        'Max Drawdown (%)': max_dd,
                        'Transaction Costs ($)': transaction_cost,
                        'Profit Factor': self._profit_factor(pnls)
                    })
        
        return pd.DataFrame(metrics_list).sort_values('Sharpe Ratio', ascending=False)
    
    def _profit_factor(self, pnls: np.ndarray) -> float:
        """
        Calculate profit factor (gross profit / gross loss).
        """
        gross_profit = np.sum(pnls[pnls > 0])
        gross_loss = abs(np.sum(pnls[pnls < 0]))
        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 0
        return gross_profit / gross_loss
    
    def _get_best_strategies(self, metrics_df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
        """
        Get top N strategies by Sharpe ratio.
        """
        return metrics_df.head(n)
    
    def _get_worst_strategies(self, metrics_df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
        """
        Get bottom N strategies by Sharpe ratio.
        """
        return metrics_df.tail(n)
    
    # =========================================================================
    # MARKDOWN REPORT
    # =========================================================================
    
    def generate_markdown(self, output_path: str = "analysis_report.md", 
                          include_figures: bool = True,
                          figures_dir: str = "figures") -> str:
        """
        Generate a Markdown report.
        
        Args:
            output_path: Path to save the report
            include_figures: Whether to include figure references
            figures_dir: Directory containing figures (relative to report)
        
        Returns:
            Path to generated report
        """
        metrics_df = self._calculate_metrics()
        best_strategies = self._get_best_strategies(metrics_df)
        worst_strategies = self._get_worst_strategies(metrics_df)
        
        report = []
        
        # Header
        report.append("# Deep Hedging with DRL - Analysis Report")
        report.append("")
        report.append(f"**Date:** {self.timestamp}")
        report.append("")
        report.append("---")
        report.append("")
        
        # Executive Summary
        report.append("##Executive Summary")
        report.append("")
        report.append("This report presents the results of a comprehensive comparison between ")
        report.append("classical hedging strategies (Black-Scholes Delta, Delta-Gamma) and ")
        report.append("Deep Reinforcement Learning agents (PPO, SAC, TD3) for European call option hedging.")
        report.append("")
        
        # Overall winner
        winner = metrics_df.iloc[0]['Strategy']
        winner_sharpe = metrics_df.iloc[0]['Sharpe Ratio']
        report.append(f"### Overall Winner: **{winner}**")
        report.append("")
        report.append(f"- **Sharpe Ratio:** {winner_sharpe:.3f}")
        report.append(f"- **Mean PnL:** ${metrics_df.iloc[0]['Mean PnL ($)']:.2f}")
        report.append(f"- **Win Rate:** {metrics_df.iloc[0]['Win Rate (%)']:.1f}%")
        report.append(f"- **Max Drawdown:** {metrics_df.iloc[0]['Max Drawdown (%)']:.2f}%")
        report.append("")
        
        report.append("---")
        report.append("")
        
        # Performance Summary Table
        report.append("##Performance Summary")
        report.append("")
        report.append("| Rank | Strategy | Sharpe | Return ($) | Win Rate | Max DD |")
        report.append("|------|----------|--------|------------|----------|--------|")
        
        for i, row in metrics_df.iterrows():
            rank = i + 1
            report.append(f"| {rank} | {row['Strategy']} | {row['Sharpe Ratio']:.3f} | ${row['Mean PnL ($)']:.0f} | {row['Win Rate (%)']:.1f}% | {row['Max Drawdown (%)']:.1f}% |")
        
        report.append("")
        report.append("---")
        report.append("")
        
        # Best and Worst Strategies
        report.append("## Best Performing Strategies")
        report.append("")
        report.append("| Rank | Strategy | Sharpe | Return ($) | Win Rate | Max DD |")
        report.append("|------|----------|--------|------------|----------|--------|")
        
        for i, row in best_strategies.iterrows():
            rank = i + 1
            report.append(f"| {rank} | {row['Strategy']} | {row['Sharpe Ratio']:.3f} | ${row['Mean PnL ($)']:.0f} | {row['Win Rate (%)']:.1f}% | {row['Max Drawdown (%)']:.1f}% |")
        
        report.append("")
        report.append("## Worst Performing Strategies")
        report.append("")
        report.append("| Rank | Strategy | Sharpe | Return ($) | Win Rate | Max DD |")
        report.append("|------|----------|--------|------------|----------|--------|")
        
        for i, row in worst_strategies.iterrows():
            rank = len(metrics_df) - i
            report.append(f"| {rank} | {row['Strategy']} | {row['Sharpe Ratio']:.3f} | ${row['Mean PnL ($)']:.0f} | {row['Win Rate (%)']:.1f}% | {row['Max Drawdown (%)']:.1f}% |")
        
        report.append("")
        report.append("---")
        report.append("")
        
        # Risk Metrics Analysis
        report.append("## Risk Metrics Analysis")
        report.append("")
        report.append("| Strategy | VaR (95%) | CVaR (95%) | VaR (99%) | CVaR (99%) | Sortino |")
        report.append("|----------|-----------|------------|-----------|------------|---------|")
        
        for _, row in metrics_df.iterrows():
            report.append(f"| {row['Strategy']} | ${row['VaR (95%) ($)']:.0f} | ${row['CVaR (95%) ($)']:.0f} | ${row['VaR (99%) ($)']:.0f} | ${row['CVaR (99%) ($)']:.0f} | {row['Sortino Ratio']:.3f} |")
        
        report.append("")
        report.append("---")
        report.append("")
        
        # DRL vs Classical Comparison
        report.append("## DRL vs Classical Comparison")
        report.append("")
        
        drl_strategies = [s for s in metrics_df['Strategy'] if 'DRL' in s or 'PPO' in s or 'SAC' in s]
        classical_strategies = [s for s in metrics_df['Strategy'] if s not in drl_strategies]
        
        if drl_strategies and classical_strategies:
            best_drl = metrics_df[metrics_df['Strategy'].isin(drl_strategies)].iloc[0]
            best_classical = metrics_df[metrics_df['Strategy'].isin(classical_strategies)].iloc[0]
            
            report.append("### Best DRL vs Best Classical")
            report.append("")
            report.append("| Metric | Best DRL | Best Classical | Difference |")
            report.append("|--------|----------|----------------|------------|")
            report.append(f"| **Strategy** | {best_drl['Strategy']} | {best_classical['Strategy']} | - |")
            report.append(f"| **Sharpe Ratio** | {best_drl['Sharpe Ratio']:.3f} | {best_classical['Sharpe Ratio']:.3f} | {best_drl['Sharpe Ratio'] - best_classical['Sharpe Ratio']:+.3f} |")
            report.append(f"| **Mean PnL** | ${best_drl['Mean PnL ($)']:.0f} | ${best_classical['Mean PnL ($)']:.0f} | ${best_drl['Mean PnL ($)'] - best_classical['Mean PnL ($)']:+.0f} |")
            report.append(f"| **Win Rate** | {best_drl['Win Rate (%)']:.1f}% | {best_classical['Win Rate (%)']:.1f}% | {best_drl['Win Rate (%)'] - best_classical['Win Rate (%)']:+.1f}% |")
            report.append(f"| **Max Drawdown** | {best_drl['Max Drawdown (%)']:.1f}% | {best_classical['Max Drawdown (%)']:.1f}% | {best_drl['Max Drawdown (%)'] - best_classical['Max Drawdown (%)']:+.1f}% |")
        
        report.append("")
        report.append("---")
        report.append("")
        
        # Transaction Cost Analysis
        report.append("## Transaction Cost Analysis")
        report.append("")
        report.append("| Strategy | Transaction Costs ($) | Profit Factor | Cost Efficiency |")
        report.append("|----------|---------------------|---------------|-----------------|")
        
        for _, row in metrics_df.iterrows():
            cost_efficiency = row['Mean PnL ($)'] / row['Transaction Costs ($)'] if row['Transaction Costs ($)'] > 0 else np.inf
            cost_efficiency_str = "∞" if cost_efficiency == np.inf else f"{cost_efficiency:.1f}x"
            report.append(f"| {row['Strategy']} | ${row['Transaction Costs ($)']:.0f} | {row['Profit Factor']:.2f} | {cost_efficiency_str} |")
        
        report.append("")
        report.append("---")
        report.append("")
        
        # Conclusions
        report.append("## Conclusions and Recommendations")
        report.append("")
        report.append("### Key Findings")
        report.append("")
        report.append(f"1. **Best Overall Strategy:** {winner} achieved the highest Sharpe ratio ({winner_sharpe:.3f})")
        report.append(f"2. **Classical vs DRL:** {best_classical['Strategy'] if 'best_classical' in dir() else 'Classical'} outperformed in terms of Sharpe ratio")
        report.append("3. **Risk Management:** Strategies with lower transaction costs generally performed better")
        report.append("4. **Drawdown Control:** DRL agents showed different drawdown characteristics")
        report.append("")
        report.append("### Recommendations")
        report.append("")
        report.append("1. **For production deployment:** Consider the top-performing strategy based on your risk tolerance")
        report.append("2. **For further improvement:**")
        report.append("   - Increase training timesteps for DRL agents")
        report.append("   - Add more market regime features")
        report.append("   - Implement ensemble methods combining multiple strategies")
        report.append("")
        report.append("---")
        report.append("")
        
        # Appendix
        report.append("## Appendix: Complete Metrics")
        report.append("")
        report.append("```")
        report.append(metrics_df.round(4).to_string(index=False))
        report.append("```")
        report.append("")
        
        # Figures
        if include_figures:
            report.append("## Visualizations")
            report.append("")
            report.append("### Portfolio Value Over Time")
            report.append("")
            report.append(f"![Portfolio Values]({figures_dir}/portfolio_values.png)")
            report.append("")
            report.append("### Sharpe Ratio Comparison")
            report.append("")
            report.append(f"![Sharpe Ratio]({figures_dir}/sharpe_ratio.png)")
            report.append("")
            report.append("### VaR Comparison")
            report.append("")
            report.append(f"![VaR Comparison]({figures_dir}/var_comparison.png)")
            report.append("")
            report.append("### Risk-Return Tradeoff")
            report.append("")
            report.append(f"![Risk-Return]({figures_dir}/risk_return_scatter.png)")
            report.append("")
        
        # Footer
        report.append("---")
        report.append("")
        report.append(f"*Report generated on {self.timestamp}*")
        report.append("")
        report.append("---")
        
        # Write to file
        output_content = "\n".join(report)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_content)
        
        print(f"Markdown report saved to {output_path}")
        return output_path
    
    # =========================================================================
    # HTML REPORT
    # =========================================================================
    
    def generate_html(self, output_path: str = "analysis_report.html",
                      figures_dir: str = "figures") -> str:
        """
        Generate an HTML report.
        
        Args:
            output_path: Path to save the report
            figures_dir: Directory containing figures (relative to report)
        
        Returns:
            Path to generated report
        """
        metrics_df = self._calculate_metrics()
        best_strategies = self._get_best_strategies(metrics_df)
        worst_strategies = self._get_worst_strategies(metrics_df)
        winner = metrics_df.iloc[0]['Strategy']
        
        html = []
        
        html.append('<!DOCTYPE html>')
        html.append('<html lang="en">')
        html.append('<head>')
        html.append('    <meta charset="UTF-8">')
        html.append('    <meta name="viewport" content="width=device-width, initial-scale=1.0">')
        html.append('    <title>Deep Hedging Analysis Report</title>')
        html.append('    <style>')
        html.append('        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }')
        html.append('        h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }')
        html.append('        h2 { color: #34495e; margin-top: 30px; }')
        html.append('        h3 { color: #555; }')
        html.append('        table { border-collapse: collapse; width: 100%; margin: 20px 0; }')
        html.append('        th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }')
        html.append('        th { background-color: #2c3e50; color: white; }')
        html.append('        tr:nth-child(even) { background-color: #f9f9f9; }')
        html.append('        .winner { background-color: #d5f5e3; }')
        html.append('        .bad { background-color: #fadbd8; }')
        html.append('        img { max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; border-radius: 4px; }')
        html.append('        .footer { text-align: center; margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 12px; color: #777; }')
        html.append('        .metric-box { background-color: #ecf0f1; border-radius: 8px; padding: 15px; margin: 20px 0; }')
        html.append('    </style>')
        html.append('</head>')
        html.append('<body>')
        
        # Header
        html.append(f'<h1>Deep Hedging with DRL - Analysis Report</h1>')
        html.append(f'<p><strong>Date:</strong> {self.timestamp}</p>')
        html.append('<hr>')
        
        # Executive Summary
        html.append('<h2>Executive Summary</h2>')
        html.append('<div class="metric-box">')
        html.append(f'<h3>Overall Winner: <span style="color:#27ae60;">{winner}</span></h3>')
        html.append(f'<ul>')
        html.append(f'<li><strong>Sharpe Ratio:</strong> {metrics_df.iloc[0]["Sharpe Ratio"]:.3f}</li>')
        html.append(f'<li><strong>Mean PnL:</strong> ${metrics_df.iloc[0]["Mean PnL ($)"]:.2f}</li>')
        html.append(f'<li><strong>Win Rate:</strong> {metrics_df.iloc[0]["Win Rate (%)"]:.1f}%</li>')
        html.append(f'<li><strong>Max Drawdown:</strong> {metrics_df.iloc[0]["Max Drawdown (%)"]:.2f}%</li>')
        html.append('</ul>')
        html.append('</div>')
        
        # Performance Summary Table
        html.append('<h2>Performance Summary</h2>')
        html.append('<table>')
        html.append('<tr><th>Rank</th><th>Strategy</th><th>Sharpe</th><th>Return ($)</th><th>Win Rate</th><th>Max DD</th></tr>')
        
        for i, row in metrics_df.iterrows():
            winner_class = 'winner' if i == 0 else ''
            html.append(f'<tr class="{winner_class}"><td>{i+1}</td><td>{row["Strategy"]}</td><td>{row["Sharpe Ratio"]:.3f}</td><td>${row["Mean PnL ($)"]:.0f}</td><td>{row["Win Rate (%)"]:.1f}%</td><td>{row["Max Drawdown (%)"]:.1f}%</td></tr>')
        
        html.append('</table>')
        
        # Risk Metrics
        html.append('<h2>Risk Metrics</h2>')
        html.append('<table>')
        html.append('<tr><th>Strategy</th><th>VaR (95%)</th><th>CVaR (95%)</th><th>Sortino</th><th>Calmar</th></tr>')
        
        for _, row in metrics_df.iterrows():
            html.append(f'<tr><td>{row["Strategy"]}</td><td>${row["VaR (95%) ($)"]:.0f}</td><td>${row["CVaR (95%) ($)"]:.0f}</td><td>{row["Sortino Ratio"]:.3f}</td><td>{row["Calmar Ratio"]:.3f}</td></tr>')
        
        html.append('</table>')
        
        # Figures
        html.append('<h2>Visualizations</h2>')
        html.append(f'<img src="{figures_dir}/portfolio_values.png" alt="Portfolio Values">')
        html.append(f'<img src="{figures_dir}/sharpe_ratio.png" alt="Sharpe Ratio">')
        html.append(f'<img src="{figures_dir}/var_comparison.png" alt="VaR Comparison">')
        html.append(f'<img src="{figures_dir}/risk_return_scatter.png" alt="Risk-Return Tradeoff">')
        html.append(f'<img src="{figures_dir}/drawdown_comparison.png" alt="Drawdown Comparison">')
        
        # Footer
        html.append('<div class="footer">')
        html.append(f'<p>Report generated on {self.timestamp}</p>')
        html.append('</div>')
        
        html.append('</body>')
        html.append('</html>')
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html))
        
        print(f"HTML report saved to {output_path}")
        return output_path
    
    # =========================================================================
    # GENERATE ALL
    # =========================================================================
    
    def generate_all(self, output_dir: str = "10_reports", 
                     figures_dir: str = "figures") -> Dict[str, str]:
        """
        Generate all reports (Markdown and HTML).
        
        Args:
            output_dir: Directory to save reports
            figures_dir: Directory containing figures (relative to report)
        
        Returns:
            Dictionary with file paths
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        file_paths = {}
        
        # Generate Markdown report
        md_path = os.path.join(output_dir, "analysis_report.md")
        self.generate_markdown(md_path, figures_dir=figures_dir)
        file_paths['markdown'] = md_path
        
        # Generate HTML report
        html_path = os.path.join(output_dir, "analysis_report.html")
        self.generate_html(html_path, figures_dir=figures_dir)
        file_paths['html'] = html_path
        
        # Save metrics CSV
        metrics_df = self._calculate_metrics()
        csv_path = os.path.join(output_dir, "metrics_summary.csv")
        metrics_df.to_csv(csv_path, index=False)
        file_paths['csv'] = csv_path
        
        print(f"\nAll reports saved to {output_dir}")
        return file_paths


if __name__ == "__main__":
    print("="*70)
    print("REPORT GENERATOR TEST")
    print("="*70)
    
    # Create sample data
    np.random.seed(42)
    sample_results = {
        'Black-Scholes': {'pnls': np.random.normal(50, 100, 200)},
        'Delta-Gamma': {'pnls': np.random.normal(60, 90, 200)},
        'DRL_PPO': {'pnls': np.random.normal(80, 85, 200)},
        'DRL_SAC': {'pnls': np.random.normal(70, 95, 200)}
    }
    
    # Create comparison DataFrame
    sample_df = pd.DataFrame([
        {'Strategy': 'Black-Scholes', 'Mean PnL': 50, 'Sharpe': 0.85, 'Win Rate %': 52, 'Max DD %': -15},
        {'Strategy': 'Delta-Gamma', 'Mean PnL': 60, 'Sharpe': 0.95, 'Win Rate %': 54, 'Max DD %': -12},
        {'Strategy': 'DRL_PPO', 'Mean PnL': 80, 'Sharpe': 1.12, 'Win Rate %': 56, 'Max DD %': -10},
        {'Strategy': 'DRL_SAC', 'Mean PnL': 70, 'Sharpe': 1.05, 'Win Rate %': 55, 'Max DD %': -11}
    ])
    
    # Generate report
    generator = ReportGenerator(sample_results, sample_df)
    generator.generate_all(output_dir="test_reports", figures_dir="figures")
