"""
transport/ package — expose the public interface.
"""
from .fgw import fused_gromov_wasserstein_incent
from .conditional_gradient import cg_incent, generic_conditional_gradient_incent
from .linesearch import solve_gromov_linesearch

__all__ = [
    "fused_gromov_wasserstein_incent",
    "cg_incent",
    "generic_conditional_gradient_incent",
    "solve_gromov_linesearch",
]
