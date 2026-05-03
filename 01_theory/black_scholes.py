# -*- coding: utf-8 -*-
"""
@author: Ahmed Baahmed


================================================================================
BLACK-SCHOLES OPTION PRICING MODEL
================================================================================
Theory Foundation for Deep Hedging Project
Reference: Black & Scholes (1973) - "The Pricing of Options and Corporate Liabilities"

This module implements the Black-Scholes-Merton option pricing model and
provides tools for calculating option prices and Greeks.
================================================================================
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


# =============================================================================
# CLASS 1: BlackScholes - Main Option Pricing Class
# =============================================================================
# This class implements the complete Black-Scholes option pricing model.
# It calculates:
#   - Option prices (Call and Put)
#   - All Greeks (Delta, Gamma, Vega, Theta, Rho)
#   - Implied volatility from market prices
#
# Assumptions:
#   - European options (exercise only at expiry)
#   - No dividends
#   - Constant volatility
#   - Lognormal returns
#   - No transaction costs
#   - Continuous trading
# =============================================================================

class BlackScholes:
    """
    Black-Scholes-Merton option pricing model.
    
    This class provides methods to price European call and put options,
    calculate option Greeks (sensitivities), and compute implied volatility.
    """
    
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float):
        """
        Initialize option parameters.
        
        Args:
            S: Current underlying price (Spot price)
            K: Strike price
            T: Time to expiry (in years)
            r: Risk-free rate (annualized)
            sigma: Implied volatility (annualized)
        """
        self.S = S
        self.K = K
        self.T = max(T, 1e-6)  # Avoid division by zero
        self.r = r
        self.sigma = sigma
    
    
    # =========================================================================
    # PRIVATE METHODS - Internal Calculations
    # =========================================================================
    
    def _d1(self) -> float:
        """
        Calculate d1 parameter.
        
        Formula: d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
        This is a key intermediate value in Black-Scholes formula.
        """
        return (np.log(self.S / self.K) + 
                (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
    
    def _d2(self) -> float:
        """
        Calculate d2 parameter.
        
        Formula: d2 = d1 - σ√T
        
        This is the second key intermediate value in Black-Scholes formula.
        """
        return self._d1() - self.sigma * np.sqrt(self.T)
    
    
    # =========================================================================
    # OPTION PRICING METHODS
    # =========================================================================
    
    def call_price(self) -> float:
        """
        Calculate European call option price:
        
        Formula: C = S·N(d1) - K·e^(-rT)·N(d2)
        
        Returns:
            Call option price
        """
        d1 = self._d1()
        d2 = self._d2()
        return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
    
    def put_price(self) -> float:
        """
        Calculate European put option price.
        
        Formula: P = K·e^(-rT)·N(-d2) - S·N(-d1)
        
        Returns:
            Put option price
        """
        d1 = self._d1()
        d2 = self._d2()
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
    
    
    # =========================================================================
    # OPTION GREEKS - FIRST ORDER SENSITIVITIES
    # =========================================================================
    # Greeks measure sensitivity of option price to various parameters.
    # These are essential for risk management and hedging.
    # =========================================================================
    
    def delta_call(self) -> float:
        """
        Call option delta (Δ).
        
        Definition: Rate of change of option price with respect to underlying price.
        Formula: Δ_call = N(d1)
        
        Delta ranges from 0 to 1 for calls.
        It represents the hedge ratio - number of shares to hedge one option.
        """
        return norm.cdf(self._d1())
    
    def delta_put(self) -> float:
        """
        Put option delta (Δ).
        
        Formula: Δ_put = N(d1) - 1
        
        Delta ranges from -1 to 0 for puts.
        """
        return norm.cdf(self._d1()) - 1
    
    def gamma(self) -> float:
        """
        Option gamma (Γ).
        
        Definition: Rate of change of delta with respect to underlying price.
        Formula: Γ = φ(d1) / (S·σ·√T)
        
        Gamma is same for calls and puts.
        It measures the convexity of the option price.
        """
        d1 = self._d1()
        return norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
    
    def vega(self) -> float:
        """
        Option vega (ν).
        
        Definition: Rate of change of option price with respect to volatility.
        Formula: ν = S·φ(d1)·√T / 100
        
        Vega is same for calls and puts.
        Divided by 100 to represent 1% change in volatility.
        """
        d1 = self._d1()
        return self.S * norm.pdf(d1) * np.sqrt(self.T) / 100
    
    def theta_call(self) -> float:
        """
        Call option theta (Θ).
        
        Definition: Rate of change of option price with respect to time.
        Formula: Θ_call = -S·φ(d1)·σ/(2√T) - r·K·e^(-rT)·N(d2)
        
        Theta is typically negative - options lose value as time passes.
        """
        d1 = self._d1()
        d2 = self._d2()
        term1 = -self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T))
        term2 = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        return term1 + term2
    
    def theta_put(self) -> float:
        """
        Put option theta (Θ).
        
        Formula: Θ_put = -S·φ(d1)·σ/(2√T) + r·K·e^(-rT)·N(-d2)
        """
        d1 = self._d1()
        d2 = self._d2()
        term1 = -self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T))
        term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
        return term1 + term2
    
    def rho_call(self) -> float:
        """
        Call option rho (ρ).
        
        Definition: Rate of change of option price with respect to risk-free rate.
        Formula: ρ_call = K·T·e^(-rT)·N(d2) / 100
        
        Divided by 100 to represent 1% change in interest rate.
        """
        d2 = self._d2()
        return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2) / 100
    
    def rho_put(self) -> float:
        """
        Put option rho (ρ).
        
        Formula: ρ_put = -K·T·e^(-rT)·N(-d2) / 100
        """
        d2 = self._d2()
        return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2) / 100
    
    
    # =========================================================================
    # IMPLIED VOLATILITY
    # =========================================================================
    # Implied volatility is the volatility that makes the model price match
    # the observed market price. It is a key input for trading decisions.
    # =========================================================================
    
    def implied_volatility(self, market_price: float, option_type: str = 'call') -> float:
        """
        Calculate implied volatility from market price using root finding.
        
        This uses the bisection method (brentq) to find the volatility
        that makes the model price equal to the observed market price.
        
        Args:
            market_price: Observed option price in the market
            option_type: 'call' or 'put'
        
        Returns:
            Implied volatility (annualized)
        """
        def objective(sigma):
            self.sigma = sigma
            if option_type == 'call':
                price = self.call_price()
            else:
                price = self.put_price()
            return price - market_price
        
        try:
            iv = brentq(objective, 1e-6, 5.0)
            return iv
        except ValueError:
            return np.nan
    
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_all_greeks(self) -> dict:
        """
        Return all Greeks as a dictionary.
        
        Returns:
            Dictionary containing delta, gamma, vega, theta, rho
        """
        return {
            'delta': self.delta_call(),
            'gamma': self.gamma(),
            'vega': self.vega(),
            'theta': self.theta_call(),
            'rho': self.rho_call()
        }


# =============================================================================
# CLASS 2: Greeks - Utility Class for Direct Greek Calculations
# =============================================================================
# This class provides static methods to calculate Greeks without
# instantiating the full BlackScholes class.
# =============================================================================

class Greeks:
    """
    Utility class for calculating option Greeks directly.
    """
    
    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate delta directly.
        
        Args:
            S: Current underlying price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
        
        Returns:
            Delta value
        """
        bs = BlackScholes(S, K, T, r, sigma)
        return bs.delta_call()
    
    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate gamma directly.
        
        Args:
            S: Current underlying price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
        
        Returns:
            Gamma value
        """
        bs = BlackScholes(S, K, T, r, sigma)
        return bs.gamma()
    
    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate vega directly.
        
        Args:
            S: Current underlying price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
        
        Returns:
            Vega value (for 1% change in volatility)
        """
        bs = BlackScholes(S, K, T, r, sigma)
        return bs.vega()


# =============================================================================
# CLASS 3: PutCallParity - Arbitrage-Free Relationship
# =============================================================================
# Put-Call Parity is a fundamental no-arbitrage relationship between
# European call and put options with the same strike and expiry.
#
# Formula: C - P = S - K·e^(-rT)
#
# If this relationship is violated, arbitrage opportunities exist.
# =============================================================================

class PutCallParity:
    """
    Put-Call Parity relationship for European options.
    
    For European options on non-dividend paying stocks:
        Call - Put = S - K * e^(-rT)
    
    This relationship must hold to prevent arbitrage.
    """
    
    @staticmethod
    def call_from_put(put_price: float, S: float, K: float, T: float, r: float) -> float:
        """
        Calculate call price from put price using put-call parity.
        
        Formula: C = P + S - K·e^(-rT)
        
        Args:
            put_price: Observed put option price
            S: Current underlying price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
        
        Returns:
            Theoretical call price
        """
        return put_price + S - K * np.exp(-r * T)
    
    @staticmethod
    def put_from_call(call_price: float, S: float, K: float, T: float, r: float) -> float:
        """
        Calculate put price from call price using put-call parity.
        
        Formula: P = C - S + K·e^(-rT)
        
        Args:
            call_price: Observed call option price
            S: Current underlying price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
        
        Returns:
            Theoretical put price
        """
        return call_price - S + K * np.exp(-r * T)
    
    @staticmethod
    def verify(S: float, K: float, T: float, r: float, sigma: float) -> bool:
        """
        Verify put-call parity holds for given parameters.
        
        Args:
            S: Current underlying price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
        
        Returns:
            True if parity holds within tolerance
        """
        bs = BlackScholes(S, K, T, r, sigma)
        call = bs.call_price()
        put = bs.put_price()
        left = call - put
        right = S - K * np.exp(-r * T)
        return np.isclose(left, right, rtol=1e-6, atol=1e-6)


# =============================================================================
# MAIN EXECUTION - TESTING THE MODULE
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("BLACK-SCHOLES OPTION PRICING MODEL")
    print("="*70)
    
    # =========================================================================
    # TEST PARAMETERS
    # =========================================================================
    S = 100      # Current price ($100)
    K = 100      # Strike price ($100)
    T = 1.0      # 1 year to expiry
    r = 0.05     # 5% risk-free rate
    sigma = 0.2  # 20% volatility
    
    print(f"\nOPTION PARAMETERS:")
    print(f"{'Underlying Price (S):':<25} ${S}")
    print(f"{'Strike Price (K):':<25} ${K}")
    print(f"{'Time to Expiry (T):':<25} {T} year")
    print(f"{'Risk-Free Rate (r):':<25} {r*100}%")
    print(f"{'Volatility (σ):':<25} {sigma*100}%")
    
    # =========================================================================
    # CREATE BLACK-SCHOLES INSTANCE
    # =========================================================================
    bs = BlackScholes(S, K, T, r, sigma)
    
    # =========================================================================
    # TEST OPTION PRICES
    # =========================================================================
    print(f"\nOPTION PRICES:")
    print(f"{'Call Option Price:':<25} ${bs.call_price():.4f}")
    print(f"{'Put Option Price:':<25} ${bs.put_price():.4f}")
    
    # =========================================================================
    # TEST OPTION GREEKS
    # =========================================================================
    print(f"\nOPTION GREEKS (CALL):")
    greeks = bs.get_all_greeks()
    for name, value in greeks.items():
        print(f"{name.capitalize() + ':':<25} {value:.6f}")
    
    # =========================================================================
    # TEST PUT-CALL PARITY
    # =========================================================================
    print(f"\nPUT-CALL PARITY VERIFICATION:")
    is_valid = PutCallParity.verify(S, K, T, r, sigma)
    print(f"{'Parity holds:':<25} {is_valid}")
    
    # =========================================================================
    # TEST IMPLIED VOLATILITY
    # =========================================================================
    print(f"\nIMPLIED VOLATILITY TEST:")
    market_price = bs.call_price()  # Use model price as market price
    iv = bs.implied_volatility(market_price, 'call')
    print(f"{'Model price:':<25} ${market_price:.4f}")
    print(f"{'Implied volatility:':<25} {iv*100:.2f}%")
    
    # =========================================================================
    # TEST GREEKS UTILITY CLASS
    # =========================================================================
    print(f"\nGREEKS UTILITY CLASS TEST:")
    delta_direct = Greeks.delta(S, K, T, r, sigma)
    gamma_direct = Greeks.gamma(S, K, T, r, sigma)
    vega_direct = Greeks.vega(S, K, T, r, sigma)
    print(f"{'Delta (direct):':<25} {delta_direct:.6f}")
    print(f"{'Gamma (direct):':<25} {gamma_direct:.6f}")
    print(f"{'Vega (direct):':<25} {vega_direct:.6f}")
    
    print("\n" + "="*70)
    print("Black-Scholes module ready for use!")
    print("="*70)