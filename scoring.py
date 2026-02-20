"""
Age-progression scoring — unchanged from the modular/unbalanced versions.
"""
from __future__ import annotations

import numpy as np
from anndata import AnnData


def age_progression_score(
    sliceA: AnnData,
    sliceB: AnnData,
    data1: str,
    data2: str,
    filePath: str,
) -> tuple[AnnData, AnnData]:
    """
    Compute an age-progression score for every cell in both slices.

    The score is the transport-plan-weighted cosine distance, normalised by
    the number of cells so that values are comparable across differently sized
    slices.

    Parameters
    ----------
    sliceA, sliceB : AnnData — source and target slices
    data1, data2   : str    — slice labels used to find cached .npy files
    filePath       : str    — directory containing the .npy files

    Returns
    -------
    sliceA, sliceB with `obs['age_progression_score']` populated.
    """
    cosine_dist = np.load(f"{filePath}/cosine_dist_gene_expr_{data1}_{data2}.npy")
    pi_mat = np.load(f"{filePath}/pi_matrix_{data1}_{data2}.npy")

    score_mat = pi_mat * cosine_dist

    sliceA.obs['age_progression_score'] = (
        np.sum(score_mat, axis=1, dtype=np.float64) / (1 / sliceA.n_obs) * 100
    )
    sliceB.obs['age_progression_score'] = (
        np.sum(score_mat, axis=0, dtype=np.float64) / (1 / sliceB.n_obs) * 100
    )

    return sliceA, sliceB
