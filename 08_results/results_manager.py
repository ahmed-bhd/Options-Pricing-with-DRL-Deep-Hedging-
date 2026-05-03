# -*- coding: utf-8 -*-
"""
================================================================================
RESULTS MANAGER - SAVE/LOAD RESULTS TO CSV
================================================================================

This module handles saving and loading backtest results to CSV files.
Generated files:
    - hedging_errors.csv: PnL distributions for each strategy
    - performance_summary.csv: Metrics table
    - transactions_log.csv: Trading activity (if available)
================================================================================
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ResultsManager:
    """
    Manager for saving and loading backtest results.
    
    Output files:
        - hedging_errors.csv: PnL distributions
        - performance_summary.csv: Performance metrics
        - transactions_log.csv: Trading activity
    """
    
    def __init__(self, output_dir: str = "08_results"):
        """
        Initialize results manager.
        
        Args:
            output_dir: Directory to save results (default: "08_results")
        """
        self.output_dir = output_dir
        self._ensure_directory()
        
        print(f"\nResults Manager initialized")
        print(f"Output directory: {self.output_dir}")
    
    def _ensure_directory(self):
        """
        Ensure output directory exists.
        """
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _get_timestamp(self) -> str:
        """
        Get current timestamp for unique filenames.
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # =========================================================================
    # SAVE METHODS
    # =========================================================================
    
    def save_hedging_errors(self, 
                            results: Dict[str, Dict],
                            filename: Optional[str] = None) -> str:
        """
        Save hedging errors (PnL distributions) to CSV.
        
        Args:
            results: Dictionary of backtest results from BacktestEngine
            filename: Optional custom filename
        
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"hedging_errors_{self._get_timestamp()}.csv"
        else:
            filename = filename if filename.endswith('.csv') else f"{filename}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Prepare data
        data = []
        for strategy_name, strategy_data in results.items():
            if 'pnls' in strategy_data:
                pnls = strategy_data['pnls']
                for i, pnl in enumerate(pnls):
                    data.append({
                        'Strategy': strategy_name,
                        'Path': i,
                        'PnL': pnl
                    })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        
        print(f"Saved hedging errors: {filepath}")
        print(f"{len(df)} records, {df['Strategy'].nunique()} strategies")
        
        return filepath
    
    def save_performance_summary(self, 
                                  comparison_df: pd.DataFrame,
                                  filename: Optional[str] = None) -> str:
        """
        Save performance summary to CSV.
        
        Args:
            comparison_df: DataFrame from BacktestEngine.run_all()
            filename: Optional custom filename
        
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"performance_summary_{self._get_timestamp()}.csv"
        else:
            filename = filename if filename.endswith('.csv') else f"{filename}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        comparison_df.to_csv(filepath, index=False)
        
        print(f"Saved performance summary: {filepath}")
        print(f"{len(comparison_df)} strategies")
        
        return filepath
    
    def save_transactions_log(self, 
                              transactions: Dict[str, List],
                              filename: Optional[str] = None) -> str:
        """
        Save transactions log to CSV.
        
        Args:
            transactions: Dictionary of transaction data
            filename: Optional custom filename
        
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"transactions_log_{self._get_timestamp()}.csv"
        else:
            filename = filename if filename.endswith('.csv') else f"{filename}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        df = pd.DataFrame(transactions) if transactions else pd.DataFrame()
        df.to_csv(filepath, index=False)
        
        print(f"Saved transactions log: {filepath}")
        
        return filepath
    
    def save_all_results(self,
                         results: Dict[str, Dict],
                         comparison_df: pd.DataFrame,
                         prefix: Optional[str] = None) -> Dict[str, str]:
        """
        Save all results to CSV files.
        
        Args:
            results: Dictionary of backtest results
            comparison_df: DataFrame with performance comparison
            prefix: Optional prefix for filenames
        
        Returns:
            Dictionary with file paths
        """
        print("\nSaving Results:")
        print("-" * 40)
        
        file_paths = {}
        
        # Add prefix if provided
        if prefix:
            hedging_file = f"{prefix}_hedging_errors.csv"
            summary_file = f"{prefix}_performance_summary.csv"
            transactions_file = f"{prefix}_transactions_log.csv"
        else:
            hedging_file = None
            summary_file = None
            transactions_file = None
        
        # Save hedging errors
        file_paths['hedging_errors'] = self.save_hedging_errors(results, hedging_file)
        
        # Save performance summary
        file_paths['performance_summary'] = self.save_performance_summary(comparison_df, summary_file)
        
        # Save transactions log (if available)
        transactions = self._extract_transactions(results)
        if transactions:
            file_paths['transactions_log'] = self.save_transactions_log(transactions, transactions_file)
        
        return file_paths
    
    def _extract_transactions(self, results: Dict[str, Dict]) -> Dict[str, List]:
        """
        Extract transaction data from results.
        
        Args:
            results: Dictionary of backtest results
        
        Returns:
            Dictionary with transaction data
        """
        transactions = {
            'strategy': [],
            'path': [],
            'step': [],
            'action': [],
            'price': [],
            'cost': []
        }
        
        for strategy_name, strategy_data in results.items():
            if 'daily_actions' in strategy_data:
                actions = strategy_data['daily_actions']
                prices = strategy_data.get('prices', [])
                costs = strategy_data.get('transaction_costs', [])
                
                for path_idx, path_actions in enumerate(actions):
                    for step, action in enumerate(path_actions):
                        transactions['strategy'].append(strategy_name)
                        transactions['path'].append(path_idx)
                        transactions['step'].append(step)
                        transactions['action'].append(action)
                        transactions['price'].append(prices[path_idx][step] if prices else 0)
                        transactions['cost'].append(costs[path_idx][step] if costs else 0)
        
        return transactions
    
    # =========================================================================
    # LOAD METHODS
    # =========================================================================
    
    def load_hedging_errors(self, filename: str) -> pd.DataFrame:
        """
        Load hedging errors from CSV.
        
        Args:
            filename: Name of the CSV file
        
        Returns:
            DataFrame with hedging errors
        """
        filepath = os.path.join(self.output_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath)
        print(f"Loaded hedging errors: {filepath}")
        print(f"{len(df)} records")
        
        return df
    
    def load_performance_summary(self, filename: str) -> pd.DataFrame:
        """
        Load performance summary from CSV.
        
        Args:
            filename: Name of the CSV file
        
        Returns:
            DataFrame with performance summary
        """
        filepath = os.path.join(self.output_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath)
        print(f"Loaded performance summary: {filepath}")
        print(f"{len(df)} strategies")
        
        return df
    
    def load_transactions_log(self, filename: str) -> pd.DataFrame:
        """
        Load transactions log from CSV.
        
        Args:
            filename: Name of the CSV file
        
        Returns:
            DataFrame with transactions log
        """
        filepath = os.path.join(self.output_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath)
        print(f"Loaded transactions log: {filepath}")
        print(f"{len(df)} transactions")
        
        return df
    
    def list_results_files(self) -> List[str]:
        """
        List all result files in the output directory.
        
        Returns:
            List of filenames
        """
        files = [f for f in os.listdir(self.output_dir) if f.endswith('.csv')]
        return sorted(files)


if __name__ == "__main__":
    print("="*70)
    print("RESULTS MANAGER TEST")
    print("="*70)
    
    # Create sample data
    np.random.seed(42)
    sample_results = {
        'Black-Scholes': {
            'pnls': np.random.normal(50, 100, 50),
            'type': 'classical'
        },
        'DRL_PPO': {
            'pnls': np.random.normal(80, 90, 50),
            'type': 'drl'
        }
    }
    
    sample_df = pd.DataFrame([
        {'Strategy': 'Black-Scholes', 'Mean PnL': 50, 'Sharpe': 0.8},
        {'Strategy': 'DRL_PPO', 'Mean PnL': 80, 'Sharpe': 1.2}
    ])
    
    # Initialize manager
    manager = ResultsManager(output_dir="test_results")
    
    # Save results
    manager.save_all_results(sample_results, sample_df, prefix="test")
    
    # List files
    print("\nAvailable result files:")
    for f in manager.list_results_files():
        print(f"- {f}")
