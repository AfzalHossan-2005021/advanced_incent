from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class AlignConfig:
    """
    Single source of truth for all hyperparameters of advanced_incent.

    Pass one AlignConfig instance to pairwise_align() instead of 20+ keyword
    arguments. All parameters have documented defaults that match the original
    INCENT paper.

    Cost function weights
    ---------------------
    alpha : float
        Weight of the Gromov-Wasserstein spatial term (0 < alpha < 1).
        (1-alpha) weights the feature cost M1 + gamma*M2.
    beta : float
        Weight of the cell-type one-hot block appended to the gene expression
        matrix before computing the cosine distance M1.
    gamma : float
        Weight of the neighbourhood JSD term M2 within the feature cost.
    radius : float
        Neighbourhood radius in the same units as `obsm['spatial']`
        (micrometres for MERFISH). Cells within this radius define the
        cellular neighbourhood for each spot.

    Unbalanced OT
    -------------
    rho1 : float
        KL marginal-slack weight for the source slice (sliceA).
        Increasing this tightens the constraint pi @ 1 == a.
        Set >= balanced_fallback_threshold to recover exact balanced OT.
    rho2 : float
        KL marginal-slack weight for the target slice (sliceB).
    balanced_fallback_threshold : float
        When both rho1 and rho2 >= this value the inner solver falls back
        to exact EMD (balanced OT). Default 1e6.

    CG solver
    ---------
    numItermax : int
        Maximum Frank-Wolfe iterations.
    tol_rel : float
        Relative change tolerance for early stopping.
    tol_abs : float
        Absolute change tolerance for early stopping.

    Distances
    ---------
    neighborhood_dissimilarity : str
        How to compare neighbourhood distributions.
        One of 'jsd' (Jensen-Shannon distance), 'cosine', 'msd'
        (mean squared distance).
    jsd_chunk_size : int
        Number of rows processed per GPU batch in the vectorised JSD.
        Reduce if you get CUDA out-of-memory errors.

    Initialisation
    --------------
    cell_type_init : bool
        If True, initialise the transport plan G0 using cell-type matching
        (cells of the same type receive higher prior weight).
        Fills the `# todo` comment in the original code.
    cell_type_init_weight : float
        How much to boost same-cell-type pairs in G0 relative to
        cross-type pairs. Value of 1.0 reduces to uniform outer(a,b).

    Compute
    -------
    use_gpu : bool
        Use CUDA if available. The best GPU (by free memory) is selected
        automatically when multiple devices are present.
    n_jobs : int
        Number of parallel workers for joblib CPU-parallel steps (e.g.
        neighbourhood computation).  -1 = all available CPUs.

    I/O
    ---
    use_rep : str or None
        Use `adata.obsm[use_rep]` as the gene-expression matrix instead of
        `adata.X`. Useful for PCA-reduced representations.
    overwrite : bool
        Recompute and overwrite cached matrices even if they already exist.
    verbose : bool
        Print Frank-Wolfe iteration details.
    norm : bool
        Normalise spatial distance matrices so that the minimum non-zero
        inter-spot distance equals 1.
    """

    # ── cost function ──────────────────────────────────────────────────────
    alpha: float = 0.1
    beta: float = 0.8
    gamma: float = 0.8
    radius: float = 100.0

    # ── unbalanced OT ──────────────────────────────────────────────────────
    rho1: float = 1.0
    rho2: float = 1.0
    balanced_fallback_threshold: float = 1e6

    # ── CG solver ──────────────────────────────────────────────────────────
    numItermax: int = 6000
    tol_rel: float = 1e-9
    tol_abs: float = 1e-9

    # ── distances ──────────────────────────────────────────────────────────
    neighborhood_dissimilarity: Literal['jsd', 'cosine', 'msd'] = 'jsd'
    jsd_chunk_size: int = 64

    # ── initialisation ─────────────────────────────────────────────────────
    cell_type_init: bool = True
    cell_type_init_weight: float = 10.0

    # ── compute ────────────────────────────────────────────────────────────
    use_gpu: bool = False
    n_jobs: int = -1

    # ── I/O ────────────────────────────────────────────────────────────────
    use_rep: Optional[str] = None
    overwrite: bool = False
    verbose: bool = False
    norm: bool = False
