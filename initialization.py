"""
Cell-type aware initialisation for the transport plan G0.

Fills the '# todo: integrate the cell-type aware initialization' comment
that appears in the original INCENT code.
"""
from __future__ import annotations

import numpy as np
from anndata import AnnData


def cell_type_aware_init(
    sliceA: AnnData,
    sliceB: AnnData,
    a: np.ndarray,
    b: np.ndarray,
    weight: float = 10.0,
) -> np.ndarray:
    """
    Build a cell-type informed initial transport plan G0.

    Instead of the uniform outer(a, b) used in the original code, pairs of
    cells that share the same annotated type receive a `weight`-fold prior
    boost.  The matrix is then normalised to sum to 1 so it is a valid joint
    distribution.

    Math
    ----
    Let S[i,j] = weight  if cell_type(i) == cell_type(j)
               = 1       otherwise

    G0 = outer(a, b) * S
    G0 = G0 / sum(G0)          # normalise to joint distribution

    When weight == 1 this reduces exactly to outer(a, b) (uniform init).

    Parameters
    ----------
    sliceA, sliceB : AnnData
        Source and target slices. Must have `obs['cell_type_annot']`.
    a, b : np.ndarray
        Marginal distributions (shape ns and nt). Uniform by default.
    weight : float
        Prior boost for same-cell-type pairs. Default 10.0.

    Returns
    -------
    G0 : np.ndarray, shape (ns, nt)
        Normalised initial transport plan.
    """
    types_A = np.array(sliceA.obs['cell_type_annot'])   # (ns,)
    types_B = np.array(sliceB.obs['cell_type_annot'])   # (nt,)

    # Vectorised same-type mask — shape (ns, nt)
    same_type = (types_A[:, None] == types_B[None, :])  # bool broadcast

    # Score matrix: 1 everywhere, weight where types match
    S = np.where(same_type, float(weight), 1.0)

    # Initial plan weighted by marginals
    G0 = np.outer(a, b) * S

    # Normalise: G0 must sum to 1 (joint distribution)
    total = G0.sum()
    if total > 0:
        G0 /= total

    return G0
