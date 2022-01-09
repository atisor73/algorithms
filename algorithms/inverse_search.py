import numpy as np
import scipy.stats
import scipy.special
import scipy.optimize

# global declarations
sqrt2 = np.sqrt(2)
Γ = scipy.special.gamma
erfinv = scipy.special.erfinv


def find_normal(L, U, bulk=0.99, precision=4, return_bounds=False):
    """
    Normal Distribution
    ----------------------
    pass in lower and upper bounds,
    along with center mass for desired distribution (default 99%),
    returns desired μ and σ

    Arguments
    ----------------------
    L: lower value (float)
    U: upper value (float)
    bulk: center mass (float, (0, 1))
    precision: integer to np.round() μ, σ (int)

    Returns:
    ----------------------
    μ: center, rounded to `precision` (float)
    σ: spread, rounded to `precision` (float)
    """
    Lppf = (1-bulk)/2
    Uppf = 1-(1-bulk)/2

    μ = (U - L)/2 + L
    σ = np.average([ np.abs((L - μ) / (sqrt2 * erfinv(2*Lppf-1))),
                     np.abs((U - μ) / (sqrt2 * erfinv(2*Uppf-1)))])

    if return_bounds:
        L_guess, U_guess = scipy.stats.norm.ppf([Lppf, Uppf], loc=μ, scale=σ)
        return np.round(μ, precision), np.round(σ, precision), L_guess, U_guess
    return np.round(μ, precision), np.round(σ, precision)


def find_lognormal(L, U, bulk=0.90, precision=4, return_bounds=False):
    """
    LogNormal Distribution
    ----------------------
    pass in lower and upper bounds,
    along with center mass for desired distribution (default 90%),
    returns desired μ and σ

    Arguments
    ----------------------
    L: lower value (float)
    U: upper value (float)
    bulk: center mass (float, (0, 1))
    precision: integer to np.round() μ, σ (int)

    Returns:
    ----------------------
    μ: center, rounded to `precision` (float)
    σ: spread, rounded to `precision` (float)
    """
    Lppf = (1-bulk)/2
    Uppf = 1-(1-bulk)/2


    star = erfinv(2*Lppf-1) / erfinv(2*Uppf-1)

    μ = (star*np.log(U) - np.log(L)) / (star - 1)
    σ = np.average([ (np.log(L)-μ) / (sqrt2*erfinv(2*Lppf-1)),
                     (np.log(U)-μ) / (sqrt2*erfinv(2*Uppf-1))  ])
    if return_bounds:
        L_guess, U_guess = scipy.stats.lognorm.ppf([Lppf, Uppf], σ, loc=0, scale=np.exp(μ))
        return np.round(μ, precision), np.round(σ, precision), L_guess, U_guess

    return np.round(μ, precision), np.round(σ, precision)


def find_exponential(U, Uppf=0.99, precision=4, return_bounds=False):
    """
    Exponential Distribution
    ------------------------
    pass in upper bound (integrate from lower=0) and upper ppf (default 99%)
    returns desired β

    Arguments
    ------------------------
    U: upper value (float)
    Uppf: upper ppf (float, (0, 1))
    precision: integer to np.round() β (int)

    Returns:
    ------------------------
    β: rate, rounded to `precision` (float)
    """
    β = -1/U * np.log(1-Uppf)

    if return_bounds:
        U_guess = scipy.stats.expon.ppf(Uppf, loc=0, scale=1/β)
        return np.round(β, precision), U_guess

    return np.round(β, precision)


def find_gamma(L, U, Lppf=0.005, Uppf=0.995, bulk=None, precision=4, return_bounds=False):
    """
    Gamma Distribution
    ----------------------
    Pass in lower and upper values & ppf's (default 99%)
    Returns desired α, β

    Arguments
    ----------------------
    L: lower value (float)
    U: upper value (float)
    Lppf: lower ppf (float, (0, 1))
    Uppf: upper ppf (float, (0, 1))
    bulk: default None (overrides Lppf, Uppf), center mass
    precision: integer to np.round() α, β (int)

    Returns:
    ----------------------
    α: # of arrivals, rounded to `precision` (float)
    β: rate, rounded to `precision` (float)
    """
    if bulk is not None:
        Lppf = (1-bulk)/2
        Uppf = 1-(1-bulk)/2

    f = lambda α: scipy.special.gammaincinv(α, Lppf) / \
                  scipy.special.gammaincinv(α, Uppf) - L/U

    # locate sign change for brentq (for α search)...
    α_ = np.arange(1000)               # might change to logspace????? how to pick upper bound???
    arr_sgn = np.sign(f(α_))
    i_ = np.where(arr_sgn[:-1] + arr_sgn[1:]==0)[0][0]
    bracket_low, bracket_high = α_[i_: i_+2]

    # solving...
    α = scipy.optimize.brentq(f, bracket_low, bracket_high)
    β = np.average([scipy.special.gammaincinv(α, Lppf) / L,
                    scipy.special.gammaincinv(α, Uppf) / U])
    if return_bounds:
        L_guess, U_guess = scipy.stats.gamma.ppf([Lppf, Uppf], α, loc=0, scale=1/β)
        return np.round(α, precision), np.round(β, precision), L_guess, U_guess

    return np.round(α, precision), np.round(β, precision)


def find_invgamma(L, U, Lppf=0.005, Uppf=0.995, bulk=None, precision=4, return_bounds=False):
    """
    Inverse Gamma Distribution
    --------------------------
    Pass in lower and upper values & ppf's (default 99%)
    Returns desired α, β

    Arguments
    --------------------------
    L: lower value (float)
    U: upper value (float)
    Lppf: lower ppf (float, (0, 1))
    Uppf: upper ppf (float, (0, 1))
    bulk: default None (overrides Lppf, Uppf), center mass
    precision: integer to np.round() α, β (int)

    Returns:
    --------------------------
    α: # of arrivals, rounded to `precision` (float)
    β: rate, rounded to `precision` (float)
    """
    if bulk is not None:
        Lppf = (1-bulk)/2
        Uppf = 1-(1-bulk)/2

    f = lambda α: scipy.special.gammainccinv(α, Lppf) / \
                  scipy.special.gammainccinv(α, Uppf) - (U/L)

    # locate sign change for brentq (for α search)...
    α_ = np.arange(1_000)           # might change to logspace????? how to pick upper bound???
    arr_sgn = np.sign(f(α_))
    i_ = np.where(arr_sgn[:-1] + arr_sgn[1:]==0)[0][0]
    bracket_low, bracket_high = α_[i_: i_+2]

    # solving...
    α = scipy.optimize.brentq(f, bracket_low, bracket_high)
    β = np.average([scipy.special.gammainccinv(α, Lppf) * L,
                    scipy.special.gammainccinv(α, Uppf) * U])

    if return_bounds:
        L_guess, U_guess = scipy.stats.invgamma.ppf([Lppf, Uppf], α, loc=0, scale=β)
        return np.round(α, precision), np.round(β, precision), L_guess, U_guess

    return np.round(α, precision), np.round(β, precision)


def find_pareto(ymin, U, Uppf=0.99, precision=4, return_bounds=False):
    """
    Pareto Distribution
    ---------------------
    Pass in ymin and upper value & its ppf (default 99%)
    Returns desired α

    Arguments
    ---------------------
    ymin: lower cutoff, ensures normalizability (float)
    U: upper value (float)
    Uppf: upper ppf (float, (0, 1))
    precision: integer to np.round() α (int)

    Returns:
    ---------------------
    α: power decay of tail, rounded to `precision` (float)
    """
    α = np.log(1 - Uppf) / np.log(ymin / U)

    if return_bounds:
        U_guess = scipy.stats.invgamma.ppf(Uppf, α, scale=ymin)
        return np.round(α, precision), U_guess

    return np.round(α, precision)


def find_weibull(L, U, Lppf=.005, Uppf=.995, bulk=None, precision=4, return_bounds=False):
    """
    Weibull Distribution
    ---------------------
    Pass in lower and upper values & ppf's (default 99%)
    Returns desired α, σ

    Arguments
    ---------------------
    L: lower value (float)
    U: upper value (float)
    Lppf: lower ppf (float, (0, 1))
    Uppf: upper ppf (float, (0, 1))
    bulk: default None (overrides Lppf, Uppf), center mass
    precision: integer to np.round() α, σ (int)

    Returns:
    ---------------------
    α: shape parameter, rounded to `precision` (float)
    σ: scale parameter, rate of arrivals, rounded to `precision` (float)
    """
    if bulk is not None:
        Lppf = (1-bulk)/2
        Uppf = 1-(1-bulk)/2

    α = np.log( np.log(1-Lppf) / np.log(1-Uppf) ) / np.log(L/U)
    σ = np.average([ L/(-np.log(1-Lppf))**(1/α),
                     U/(-np.log(1-Uppf))**(1/α)  ])

    if return_bounds:
        L_guess, U_guess = scipy.stats.weibull_min.ppf([Lppf, Uppf], α, loc=0, scale=σ)
        return np.round(α, precision), np.round(σ, precision), L_guess, U_guess

    return np.round(α, precision), np.round(σ, precision)


def find_cauchy(L, U, bulk=0.90, precision=4, return_bounds=False):
    """
    Cauchy Distribution
    ---------------------
    Pass in lower and upper values & ppf's (default 99%)
    Returns desired α, σ

    Arguments
    ---------------------
    L: lower value (float)
    U: upper value (float)
    bulk: center mass (float, (0, 1))
    precision: integer to np.round() μ, σ (int)

    Returns:
    ---------------------
    μ: shape parameter, rounded to `precision` (float)
    σ: scale parameter, rate of arrivals, rounded to `precision` (float)
    """
    Lppf = (1-bulk)/2
    Uppf = 1-(1-bulk)/2

    star = np.tan(np.pi*(Lppf-1/2)) / np.tan(np.pi*(Uppf-1/2))
    μ = (star*U - L)/(star-1)
    σ = np.average([(L-μ)/np.tan(np.pi*(Lppf-1/2)),
                    (U-μ)/np.tan(np.pi*(Uppf-1/2))])

    if return_bounds:
        L_guess, U_guess = scipy.stats.cauchy.ppf([Lppf, Uppf], μ, σ)
        return np.round(μ, precision), np.round(σ, precision), L_guess, U_guess

    return np.round(μ, precision), np.round(σ, precision)


def find_studentt(ν, L, U, bulk=0.95, precision=4, return_bounds=False):
    """
    Student-t Distribution
    ----------------------
    pass in lower and upper bounds, ν (heaviness of tail),
    along with center mass for desired distribution (default 90%),
    returns desired μ and σ

    Arguments
    ----------------------
    ν: degrees of freedom, smaller -> heavier tails (float > 0)
    L: lower value (float)
    U: upper value (float)
    bulk: center mass (float, (0, 1))
    precision: integer to np.round() μ, σ (int)

    Returns:
    ----------------------
    μ: center, rounded to `precision` (float)
    σ: spread, rounded to `precision` (float)
    """
    Lppf = (1-bulk)/2
    Uppf = 1-(1-bulk)/2

    μ = (U-L)/2 + L

    spiral = Γ((ν+1)/2) / (np.sqrt(np.pi*ν) * Γ(ν/2))
    a, b, c = 1/2, (ν+1)/2, 3/2

    # solve one side
    def f(sigma):
        U_ = (U - μ) / sigma
        z = -(U_)**2/ν
        hyper = scipy.special.hyp2f1(a, b, c, z)
        zero = 1/2 + U_*spiral*hyper - Uppf
        return zero

    # brent-q's guarantees a crossing, solving...
    sigma_ = np.linspace(0.1, 100, 1000)         # change to logspace????????????? how to pick bounds?????
    arr_sgn = np.sign(f(sigma_))
    i_ = np.where(arr_sgn[:-1] + arr_sgn[1:]==0)[0][0]
    bracket_low, bracket_high = sigma_[i_: i_+2]

    σ = scipy.optimize.brentq(f, bracket_low, bracket_high)

    if return_bounds:
        L_guess, U_guess = scipy.stats.t.ppf([Lppf, Uppf], ν, μ, σ)
        return np.round(μ, precision), np.round(σ, precision), L_guess, U_guess

    return np.round(μ, precision), np.round(σ, precision)


def find_gumbel(L, U, Lppf=0.005, Uppf=0.995, bulk=None, precision=4, return_bounds=False):
    """
    Gumbel Distribution
    ----------------------
    Pass in lower and upper values & ppf's (default 99%)
    Returns desired μ, σ

    Arguments
    ----------------------
    L: lower value (float)
    U: upper value (float)
    Lppf: lower ppf (float, (0, 1))
    Uppf: upper ppf (float, (0, 1))
    bulk: default None (overrides Lppf, Uppf), center mass
    precision: integer to np.round() μ, σ (int)

    Returns:
    ----------------------
    μ: center, rounded to `precision` (float)
    σ: scale, rounded to `precision` (float)
    """
    if bulk is not None:
        Lppf = (1-bulk)/2
        Uppf = 1-(1-bulk)/2

    star = np.log(np.log(1/Lppf)) / np.log(np.log(1/Uppf))
    μ = (star * U - L) / (star - 1)
    σ = np.average([ -(L-μ) / np.log(np.log(1/Lppf)),
                     -(U-μ) / np.log(np.log(1/Uppf))  ])

    if return_bounds:
        L_guess, U_guess = scipy.stats.gumbel_r.ppf([Lppf, Uppf], loc=μ, scale=σ)
        return np.round(μ, precision), np.round(σ, precision), L_guess, U_guess

    return np.round(μ, precision), np.round(σ, precision)
