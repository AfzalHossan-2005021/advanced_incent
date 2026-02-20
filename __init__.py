"""
advanced_incent
===============
Production-quality implementation of the INCENT spatial-transcriptomics
alignment algorithm.

Quick start
-----------
>>> from advanced_incent import pairwise_align, AlignConfig
>>> config = AlignConfig(alpha=0.1, beta=0.8, gamma=0.8, radius=100.0,
...                      rho1=1.0, rho2=1.0, use_gpu=True, n_jobs=-1)
>>> pi = pairwise_align(sliceA, sliceB, config,
...                     sliceA_name="4wk", sliceB_name="24wk",
...                     filePath="outputs/")
"""

from .config import AlignConfig
from .alignment import pairwise_align
from .scoring import age_progression_score
from .initialization import cell_type_aware_init
from .neighborhood import neighborhood_distribution
from .distances import (
    jensenshannon_divergence_backend,
    cosine_distance,
    pairwise_msd,
)
from .transport import fused_gromov_wasserstein_incent

__all__ = [
    "AlignConfig",
    "pairwise_align",
    "age_progression_score",
    "cell_type_aware_init",
    "neighborhood_distribution",
    "jensenshannon_divergence_backend",
    "cosine_distance",
    "pairwise_msd",
    "fused_gromov_wasserstein_incent",
]
