# -*- coding: utf-8 -*-
"""
@author: Ahmed Baahmed

Module initializer {__init__.py} (makes it a Python package)

================================================================================
01_THEORY MODULE INITIALIZER
================================================================================

This module provides theoretical foundations for option pricing and hedging.

Exports:
    - BlackScholes: Main option pricing class
    - Greeks: Utility class for direct Greek calculations
    - PutCallParity: Put-call parity verification

Usage:
    from src.theory import BlackScholes, Greeks, PutCallParity
================================================================================
"""

from .black_scholes import BlackScholes, Greeks, PutCallParity
# Calling the class {BlackScholes, Greeks, PutCallParity} from the black_sholes.py file

__all__ = [
    'BlackScholes',
    'Greeks', 
    'PutCallParity'
]

__version__ = '1.0.0'
__author__ = 'Ahmed Baahmed'
__doc__ = "Theory module for option pricing and hedging"