"""
Algorithms is a local package for functions I keep using but am too tired to rewrite or can't seem to remember.
"""

from .audio import *
from .list import *
from .colors import *
from .numerics import *
from .distributions import *
from .prior_inverse_search import *
from .prior_dashboard_builder import bayesian_priors

from . import blurb_normal
from . import blurb_studentt
from . import blurb_gumbel
from . import blurb_exponential
from . import blurb_gamma
from . import blurb_invgamma
from . import blurb_weibull
from . import blurb_pareto
from . import blurb_lognormal
from . import blurb_cauchy

__author__ = "Rosita Fu"
__version__ = "0.0.1"
__license__ = "MIT"
__email__ = "rosita.fu99@gmail.com"
