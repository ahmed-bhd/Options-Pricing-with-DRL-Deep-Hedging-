# -*- coding: utf-8 -*-
"""
================================================================================
OPTIONS DATA SIMULATOR
================================================================================

Generates realistic synthetic options chains for training and testing.
Includes:
    - Option prices (Black-Scholes)
    - Greeks (Delta, Gamma, Vega, Theta)
    - Implied volatility smile
    - Multiple maturities and strikes

This simulator is useful for:
    - Testing the hedging environment before using real data
    - Generating training data with known parameters
    - Creating volatility surface visualizations
================================================================================
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import List, Optional, Dict, Tuple


# =============================================================================
# CLASS: Options Data Simulator
# =============================================================================

class OptionsDataSimulator:
    """
    Simulator for synthetic options data.
    Generates realistic option chains with volatility smile.
    
    The volatility smile models the observation that implied volatility
    is higher for deep in-the-money and deep out-of-the-money options,
    creating a "smile" shape when plotted against strike price.
    """
    
    def __init__(self, S0: float = 100.0, r: float = 0.02, 
                 base_iv: float = 0.20, smile_strength: float = 0.08):
        """
        Initialize options simulator.
        
        Args:
            S0: Current underlying price
            r: Risk-free rate (annualized)
            base_iv: Base implied volatility at-the-money (default: 20%)
            smile_strength: Strength of volatility smile (default: 0.08)
        """
        self.S0 = S0
        self.r = r
        self.base_iv = base_iv
        self.smile_strength = smile_strength
    
    # =========================================================================
    # VOLATILITY SMILE
    # =========================================================================
    
    def get_implied_volatility(self, strike: float, T: float) -> float:
        """
        Calculate implied volatility with smile effect.
        
        The smile is modeled as a quadratic function of moneyness:
        IV(K) = base_iv + smile_strength * (K/S0 - 1)^2
        
        Args:
            strike: Option strike price
            T: Time to expiry in years
        
        Returns:
            Implied volatility (annualized)
        """
        moneyness = strike / self.S0
        iv = self.base_iv + self.smile_strength * (moneyness - 1)**2
        return max(iv, 0.05)  # Minimum 5% volatility
    
    # =========================================================================
    # BLACK-SCHOLES PRICING
    # =========================================================================
    
    def _calculate_d1_d2(self, S: float, K: float, T: float, sigma: float) -> Tuple[float, float]:
        """
        Calculate Black-Scholes d1 and d2 parameters.
        """
        
        if T <= 0 or sigma <= 0:
            return 0, 0
        d1 = (np.log(S / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    
    def call_price(self, S: float, K: float, T: float, sigma: float) -> float:
        """
        Calculate European call option price.
        """
        
        if T <= 0:
            return max(S - K, 0)
        if sigma <= 0:
            return max(S - K * np.exp(-self.r * T), 0)
        d1, d2 = self._calculate_d1_d2(S, K, T, sigma)
        return S * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
    
    def put_price(self, S: float, K: float, T: float, sigma: float) -> float:
        """
        Calculate European put option price.
        """
        
        if T <= 0:
            return max(K - S, 0)
        if sigma <= 0:
            return max(K * np.exp(-self.r * T) - S, 0)
        d1, d2 = self._calculate_d1_d2(S, K, T, sigma)
        return K * np.exp(-self.r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    def call_delta(self, S: float, K: float, T: float, sigma: float) -> float:
        """
        Calculate call option delta.
        """
        
        if T <= 0:
            return 1.0 if S > K else 0.0
        if sigma <= 0:
            return 1.0 if S > K * np.exp(-self.r * T) else 0.0
        d1, _ = self._calculate_d1_d2(S, K, T, sigma)
        return norm.cdf(d1)
    
    def put_delta(self, S: float, K: float, T: float, sigma: float) -> float:
        """
        Calculate put option delta.
        """
        
        if T <= 0:
            return -1.0 if S < K else 0.0
        if sigma <= 0:
            return -1.0 if S < K * np.exp(-self.r * T) else 0.0
        d1, _ = self._calculate_d1_d2(S, K, T, sigma)
        return norm.cdf(d1) - 1
    
    def gamma(self, S: float, K: float, T: float, sigma: float) -> float:
        """
        Calculate option gamma (same for calls and puts).
        """
        
        if T <= 0 or sigma <= 0:
            return 0.0
        d1, _ = self._calculate_d1_d2(S, K, T, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    def vega(self, S: float, K: float, T: float, sigma: float) -> float:
        """
        Calculate option vega (sensitivity to 1% change in volatility).
        """
        
        if T <= 0 or sigma <= 0:
            return 0.0
        d1, _ = self._calculate_d1_d2(S, K, T, sigma)
        return S * norm.pdf(d1) * np.sqrt(T) / 100
    
    def theta_call(self, S: float, K: float, T: float, sigma: float) -> float:
        """
        Calculate call option theta (daily).
        """
        
        if T <= 0 or sigma <= 0:
            return 0.0
        d1, d2 = self._calculate_d1_d2(S, K, T, sigma)
        term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        term2 = -self.r * K * np.exp(-self.r * T) * norm.cdf(d2)
        return (term1 + term2) / 365  # Daily theta
    
    def theta_put(self, S: float, K: float, T: float, sigma: float) -> float:
        """
        Calculate put option theta (daily).
        """
        
        if T <= 0 or sigma <= 0:
            return 0.0
        d1, d2 = self._calculate_d1_d2(S, K, T, sigma)
        term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        term2 = self.r * K * np.exp(-self.r * T) * norm.cdf(-d2)
        return (term1 + term2) / 365  # Daily theta
    
    # =========================================================================
    # GENERATE OPTION CHAIN
    # =========================================================================
    
    def generate_option_chain(self,
                              strikes: Optional[List[float]] = None,
                              maturities: Optional[List[int]] = None,
                              add_greeks: bool = True) -> pd.DataFrame:
        """
        Generate a realistic option chain with volatility smile.
        
        Args:
            strikes: List of strike prices (default: 70 to 130 by 5)
            maturities: List of maturities in days (default: 7, 30, 60, 90, 180, 365)
            add_greeks: If True, include Greeks in the output
        
        Returns:
            DataFrame with option chain data
        """
        if strikes is None:
            strikes = np.arange(70, 131, 5)  # 70 to 130 by 5
        if maturities is None:
            maturities = [7, 30, 60, 90, 180, 365]  # 1 week to 1 year
        
        options = []
        
        for T_days in maturities:
            T_years = T_days / 365
            
            for K in strikes:
                # Implied volatility with smile
                iv = self.get_implied_volatility(K, T_years)
                
                # Prices
                call_price = self.call_price(self.S0, K, T_years, iv)
                put_price = self.put_price(self.S0, K, T_years, iv)
                
                option_data = {
                    'strike': float(K),
                    'maturity_days': int(T_days),
                    'maturity_years': float(T_years),
                    'implied_vol': float(iv),
                    'call_price': float(call_price),
                    'put_price': float(put_price),
                }
                
                # Add Greeks if requested
                if add_greeks:
                    option_data.update({
                        'call_delta': float(self.call_delta(self.S0, K, T_years, iv)),
                        'put_delta': float(self.put_delta(self.S0, K, T_years, iv)),
                        'gamma': float(self.gamma(self.S0, K, T_years, iv)),
                        'vega': float(self.vega(self.S0, K, T_years, iv)),
                        'call_theta': float(self.theta_call(self.S0, K, T_years, iv)),
                        'put_theta': float(self.theta_put(self.S0, K, T_years, iv)),
                    })
                
                options.append(option_data)
        
        df = pd.DataFrame(options)
        
        # Add moneyness column
        df['moneyness'] = df['strike'] / self.S0
        
        return df
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_atm_options(self, option_chain: pd.DataFrame) -> pd.DataFrame:
        """
        Get at-the-money options (closest strike to current price).
        
        Args:
            option_chain: DataFrame from generate_option_chain()
        
        Returns:
            DataFrame with ATM options only
        """
        
        atm_strike = option_chain['strike'].iloc[
            (option_chain['strike'] - self.S0).abs().argsort()[:1]
        ].values[0]
        return option_chain[option_chain['strike'] == atm_strike]
    
    def get_volatility_smile(self, option_chain: pd.DataFrame) -> Dict[int, Dict]:
        """
        Extract volatility smile for each maturity.
        
        Args:
            option_chain: DataFrame from generate_option_chain()
        
        Returns:
            Dictionary with maturity as key, containing strikes and IVs
        """
        smiles = {}
        
        for maturity in option_chain['maturity_days'].unique():
            subset = option_chain[option_chain['maturity_days'] == maturity]
            smiles[int(maturity)] = {
                'strikes': subset['strike'].values,
                'iv': subset['implied_vol'].values,
                'call_prices': subset['call_price'].values,
                'put_prices': subset['put_price'].values
            }
        
        return smiles
    
    def plot_volatility_smile(self, option_chain: pd.DataFrame, save_path: Optional[str] = None):
        """
        Plot the volatility smile for visualization.
        
        Args:
            option_chain: DataFrame from generate_option_chain()
            save_path: Optional path to save the figure
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for maturity in option_chain['maturity_days'].unique():
                subset = option_chain[option_chain['maturity_days'] == maturity]
                ax.plot(subset['strike'], subset['implied_vol'] * 100, 
                       'o-', label=f'{int(maturity)} days', linewidth=1.5, markersize=4)
            
            ax.axvline(x=self.S0, color='red', linestyle='--', alpha=0.5, label='ATM')
            ax.set_xlabel('Strike Price ($)')
            ax.set_ylabel('Implied Volatility (%)')
            ax.set_title('Implied Volatility Smile')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Install with: pip install matplotlib")


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def generate_sample_option_chain(S0: float = 100.0, 
                                  r: float = 0.02,
                                  base_iv: float = 0.20) -> pd.DataFrame:
    """
    Quick function to generate a sample option chain.
    
    Args:
        S0: Current underlying price
        r: Risk-free rate
        base_iv: Base implied volatility
    
    Returns:
        DataFrame with option chain
    """
    simulator = OptionsDataSimulator(S0=S0, r=r, base_iv=base_iv)
    return simulator.generate_option_chain()


# =============================================================================
# MAIN EXECUTION - TESTING
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("OPTIONS DATA SIMULATOR TEST")
    print("="*70)
    
    # Create simulator
    sim = OptionsDataSimulator(S0=100, r=0.02, base_iv=0.20, smile_strength=0.08)
    
    # Generate option chain
    print("\nGenerating option chain...")
    option_chain = sim.generate_option_chain()
    
    print(f"\nOption Chain Summary:")
    print(f"Total contracts: {len(option_chain)}")
    print(f"Strikes: {option_chain['strike'].unique()}")
    print(f"Maturities: {option_chain['maturity_days'].unique()} days")
    
    print(f"\nSample ATM Options (Strike ≈ {sim.S0}):")
    print("-"*60)
    
    atm = sim.get_atm_options(option_chain)
    for _, opt in atm.iterrows():
        # Convert to proper types for printing
        maturity = int(opt['maturity_days'])
        call_price = float(opt['call_price'])
        put_price = float(opt['put_price'])
        iv = float(opt['implied_vol']) * 100
        delta_c = float(opt['call_delta'])
        
        print(f"{maturity:3d} days: Call=${call_price:7.2f}, "
              f"Put=${put_price:7.2f}, IV={iv:5.1f}%, "
              f"Delta_C={delta_c:.3f}")
    
    print(f"\nVolatility Smile (90-day):")
    print("-"*60)
    
    smile = sim.get_volatility_smile(option_chain)
    for strike, iv in zip(smile[90]['strikes'][:5], smile[90]['iv'][:5]):
        print(f"Strike ${strike:.0f}: IV = {iv*100:.1f}%")
    
    # Plot volatility smile
    print("\nGenerating volatility smile plot...")
    sim.plot_volatility_smile(option_chain, save_path='volatility_smile.png')
    