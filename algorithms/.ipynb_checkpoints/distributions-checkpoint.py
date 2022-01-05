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


def find_gamma(L, U, Lppf=0.005, Uppf=0.995, bulk=None, precision=4):
    """
    Pass in lower and upper values & ppf's (default 99%)
    Returns desired α, β
    
    Arguments
    -----------
    L: lower value (float)
    U: upper value (float)
    Lppf: lower ppf (float, (0, 1))
    Uppf: upper ppf (float, (0, 1))
    bulk: default None (overrides Lppf, Uppf), center mass
    precision: integer to np.round() α, β (int)
    
    Returns:
    ------------
    α: # of arrivals, rounded to `precision` (float)
    β: rate, rounded to `precision` (float)
    """
    if bulk is not None: 
        Lppf = (1-bulk)/2
        Uppf = 1-(1-bulk)/2
        
    f = lambda α, Lppf, Uppf : scipy.special.gammaincinv(α, Lppf) / \
                               scipy.special.gammaincinv(α, Uppf) - L/U

    # locate sign change for brentq (for α search)... 
    α_ = np.arange(100)
    arr_sgn = np.sign(f(α_, Lppf, Uppf))
    i_ = np.where(arr_sgn[:-1] + arr_sgn[1:]==0)[0][0]
    bracket_low, bracket_high = α_[i_: i_+2]

    # solving... 
    α = scipy.optimize.brentq(f, bracket_low, bracket_high, args=(Lppf, Uppf))
    β = np.average([scipy.special.gammaincinv(α, Lppf) / L, 
                    scipy.special.gammaincinv(α, Uppf) / U])
    
    return np.round(α, precision), np.round(β, precision)


def find_invgamma(L, U, Lppf=0.005, Uppf=0.995, bulk=None, precision=4):
    """
    Pass in lower and upper values & ppf's (default 99%)
    Returns desired α, β
    
    Arguments
    -----------
    L: lower value (float)
    U: upper value (float)
    Lppf: lower ppf (float, (0, 1))
    Uppf: upper ppf (float, (0, 1))
    bulk: default None (overrides Lppf, Uppf), center mass
    precision: integer to np.round() α, β (int)
    
    Returns:
    ------------
    α: # of arrivals, rounded to `precision` (float)
    β: rate, rounded to `precision` (float)
    """
    if bulk is not None: 
        Lppf = (1-bulk)/2
        Uppf = 1-(1-bulk)/2
        
    f = lambda α: scipy.special.gammainccinv(α, Lppf) / \
                  scipy.special.gammainccinv(α, Uppf) - (U/L)

    # locate sign change for brentq (for α search)... 
    α_ = np.arange(1_000)
    arr_sgn = np.sign(f(α_))
    i_ = np.where(arr_sgn[:-1] + arr_sgn[1:]==0)[0][0]
    bracket_low, bracket_high = α_[i_: i_+2]

    # solving... 
    α = scipy.optimize.brentq(f, bracket_low, bracket_high)
    β = np.average([scipy.special.gammainccinv(α, Lppf) * L, 
                    scipy.special.gammainccinv(α, Uppf) * U])
    
    return np.round(α, precision), np.round(β, precision)


