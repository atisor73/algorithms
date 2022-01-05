import numpy as np
import pandas as pd
import scipy.stats
import scipy.special

sqrt2 = np.sqrt(2)

def find_normal(L, U, bulk=0.99, precision=4):
    """
    pass in lower and upper bounds, along with center mass for desired distribution
    returns desired μ and σ
    
    Arguments
    -----------
    L: lower value (float)
    U: upper value (float)
    bulk: center mass (float, (0, 1))
    precision: integer to np.round() μ, σ (int > 2)
    
    Returns:
    ------------
    μ: center, rounded to `precision` (float)
    σ: spread, rounded to `precision` (float)
    """
    Lppf = (1-bulk)/2
    Uppf = 1-(1-bulk)/2
    
    μ = (U - L)/2 + L
    σ = np.abs((L - μ) / (sqrt2 * scipy.special.erfinv(2*Lppf-1)))
    
    return np.round(μ, precision), np.round(σ, precision)


def find_exponential(U, Uppf=0.99, precision=4):
    """
    pass in upper bound (integrate from lower=0) and upper ppf (default 99%)
    returns desired β
    
    Arguments
    -----------
    U: upper value (float)
    Uppf: upper ppf (float, (0, 1))
    precision: integer to np.round() β (int)
    
    Returns:
    ------------
    β: rate, rounded to `precision` (float)
    """
    β = -1/U * np.log(1-Uppf)
    return np.round(β, precision)

