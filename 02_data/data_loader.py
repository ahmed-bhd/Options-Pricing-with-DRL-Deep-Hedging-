# -*- coding: utf-8 -*-
"""
================================================================================
REAL MARKET DATA LOADER
================================================================================

Loads real market data from Yahoo Finance and caches to CSV in 02_data/ folder.
- First run: Downloads from Yahoo Finance and saves to CSV
- Subsequent runs: Loads from CSV (much faster)

Data Sources:
    - Stock prices (yfinance)
    - VIX volatility index
================================================================================
"""

import numpy as np
import pandas as pd
import os
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


class MarketDataLoader:
    """
    Loader for real market data with CSV caching.
    
    Data Sources:
        - stocks: Yahoo Finance (yfinance)
        - volatility: CBOE VIX (yfinance)
    """
    
    def __init__(self, ticker: str = 'SPY', data_dir: str = None):
        """
        Initialize data loader.
        
        Args:
            ticker: Ticker symbol for the underlying asset
            data_dir: Directory to store CSV files (default: current directory)
        """
        self.ticker = ticker
        
        # Set data directory to current folder (02_data)
        if data_dir is None:
            self.data_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self.data_dir = data_dir
        
        self.stock_data = None
    
    def _get_cache_path(self, data_type: str, start_date: str, end_date: str) -> str:
        """
        Get cache file path in 02_data/ folder.
        """
        
        filename = f"{self.ticker}_{data_type}_{start_date}_to_{end_date}.csv"
        return os.path.join(self.data_dir, filename)
    
    # =========================================================================
    # STOCK PRICES (with CSV caching)
    # =========================================================================
    
    def download_stock_prices(self, 
                              start_date: str, 
                              end_date: str,
                              force_download: bool = False) -> pd.DataFrame:
        """
        Download historical stock prices with CSV caching.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force_download: If True, ignore cache and download fresh data
        
        Returns:
            DataFrame with stock price data
        """
        
        cache_path = self._get_cache_path("prices", start_date, end_date)
        
        # Try to load from cache
        if not force_download and os.path.exists(cache_path):
            print(f"Loading cached data from: {cache_path}")
            data = pd.read_csv(cache_path)
            data['Date'] = pd.to_datetime(data['Date'])
            self.stock_data = data
            print(f"Loaded {len(data)} days from cache")
            return data
        
        # Download fresh data
        print(f"Downloading {self.ticker} from {start_date} to {end_date}...")
        
        try:
            import yfinance as yf
            
            data = yf.download(self.ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                raise ValueError("No data downloaded")
            
            # Reset index to make Date a column
            data = data.reset_index()
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
            
            # Save to cache in 02_data/
            data.to_csv(cache_path, index=False)
            print(f"Downloaded and saved to: {cache_path}")
            
            self.stock_data = data
            return data
            
        except ImportError:
            print("yfinance not installed. Run: pip install yfinance")
            return self._generate_sample_data(start_date, end_date)
        except Exception as e:
            print(f"Error: {e}. Using sample data...")
            return self._generate_sample_data(start_date, end_date)
    
    def _generate_sample_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generate sample data when real data unavailable.
        """
        
        print("Generating sample data...")
                
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(dates)
        
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.015, n_days)
        prices = 100 * (1 + returns).cumprod()
        
        df = pd.DataFrame({'Date': dates, 'Close': prices})
        
        # Save sample data to cache
        cache_path = self._get_cache_path("sample_prices", start_date, end_date)
        df.to_csv(cache_path, index=False)
        
        return df
    
    def get_stock_path(self) -> np.ndarray:
        """
        Extract closing prices as numpy array.
        """
        
        if self.stock_data is not None and 'Close' in self.stock_data.columns:
            return self.stock_data['Close'].values
        return np.array([])
    
    def get_scalar_close(self, idx: int = -1) -> float:
        """
        Get closing price as scalar value.
        """
        if self.stock_data is not None and 'Close' in self.stock_data.columns:
            val = self.stock_data['Close'].iloc[idx]
            if isinstance(val, (pd.Series, np.ndarray)):
                return float(val.iloc[0]) if len(val) > 0 else 0.0
            return float(val)
        return 0.0
    
    # =========================================================================
    # VIX VOLATILITY INDEX (with CSV caching)
    # =========================================================================
    
    def download_vix_data(self, start_date: str, end_date: str, force_download: bool = False) -> pd.DataFrame:
        """
        Download VIX volatility index data with CSV caching.
        """
        
        cache_path = self._get_cache_path("vix", start_date, end_date)
        
        # Try to load from cache
        if not force_download and os.path.exists(cache_path):
            print(f"Loading cached VIX data from: {cache_path}")
            vix = pd.read_csv(cache_path)
            vix['Date'] = pd.to_datetime(vix['Date'])
            return vix
        
        # Download fresh data
        print(f"Downloading VIX from {start_date} to {end_date}...")
        
        try:
            import yfinance as yf
            
            vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
            
            if vix.empty:
                raise ValueError("No VIX data")
            
            # Clean up DataFrame
            vix = vix.reset_index()
            vix.columns = [col[0] if isinstance(col, tuple) else col for col in vix.columns]
            
            # Save to cache in 02_data/
            vix.to_csv(cache_path, index=False)
            print(f"Downloaded and saved to: {cache_path}")
            
            return vix
            
        except Exception as e:
            print(f"Error: {e}. Using sample VIX...")
            return self._generate_sample_vix(start_date, end_date)
    
    def _generate_sample_vix(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generate sample VIX data.
        """
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(dates)
        
        np.random.seed(42)
        vix_values = 15 + np.random.randn(n_days) * 3
        vix_values = np.clip(vix_values, 8, 50)
        
        df = pd.DataFrame({'Date': dates, 'Close': vix_values})
        
        cache_path = self._get_cache_path("sample_vix", start_date, end_date)
        df.to_csv(cache_path, index=False)
        
        return df


# =============================================================================
# CONVENIENCE FUNCTION: Load Market Data from Cache
# =============================================================================

def load_cached_market_data(ticker: str = 'SPY',
                            start_date: str = '2006-01-01',
                            end_date: str = '2023-12-31') -> Optional[pd.DataFrame]:
    """
    Load cached market data from CSV file in 02_data/ folder.
    
    This function reads the CSV file created by data_loader.py.
    If the file doesn't exist, it returns None (run data_loader.py first).
    
    Args:
        ticker: Ticker symbol
        start_date: Start date
        end_date: End date
    
    Returns:
        DataFrame with market data or None if not found
    """
    
    data_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(data_dir, f"{ticker}_prices_{start_date}_to_{end_date}.csv")
    
    if os.path.exists(cache_path):
        print(f"Loading cached data from: {cache_path}")
        data = pd.read_csv(cache_path)
        data['Date'] = pd.to_datetime(data['Date'])
        return data
    else:
        print(f"Cache file not found: {cache_path}")
        print("Run data_loader.py first to download and cache data")
        return None


# =============================================================================
# MAIN EXECUTION - TESTING
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("MARKET DATA LOADER WITH CSV CACHING - TEST")
    print("="*70)
    print(f"CSV files will be saved in: {os.path.dirname(os.path.abspath(__file__))}")
    
    # First run: Downloads from Yahoo Finance
    print("\nTEST 1: First Run (Download from Yahoo Finance)")
    print("-"*50)
    loader = MarketDataLoader(ticker='SPY')
    data1 = loader.download_stock_prices('2006-01-01', '2023-12-31', force_download=False)
    
    if len(data1) > 0:
        start_price = data1['Close'].iloc[0]
        end_price = data1['Close'].iloc[-1]
        
        # Convert to scalar
        if isinstance(start_price, (pd.Series, np.ndarray)):
            start_price = float(start_price.iloc[0])
        if isinstance(end_price, (pd.Series, np.ndarray)):
            end_price = float(end_price.iloc[0])
        
        print(f"\nData Summary:")
        print(f"Start price: ${start_price:.2f}")
        print(f"End price: ${end_price:.2f}")
        print(f"Total return: {(end_price/start_price - 1)*100:.2f}%")
    
    # Second run: Loads from cache (should be instant)
    print("\nTEST 2: Second Run (Load from Cache)")
    print("-"*50)
    data2 = loader.download_stock_prices('2006-01-01', '2023-12-31', force_download=False)
    print(f"Loaded {len(data2)} days from cache")
    
    # Test loading cached data function
    print("\nTEST 3: Load Cached Data Directly")
    print("-"*50)
    cached_data = load_cached_market_data('SPY', '2006-01-01', '2023-12-31')
    if cached_data is not None:
        print(f"Successfully loaded {len(cached_data)} days from cache")
    
    print("\n" + "="*70)
    print("Market data loader with CSV caching ready!")
    print("="*70)
    print("\nCSV files saved in: 02_data/ folder")
    print("File pattern: SPY_prices_2006-01-01_to_2023-12-31.csv")