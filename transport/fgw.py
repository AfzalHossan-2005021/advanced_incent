"""
Fused Gromov-Wasserstein solver — top-level entry for advanced_incent.

Sets up f(G), df(G) and the line-search strategy, then delegates to
`cg_incent`.  Identical logic to unbalanced_incent with two clean-ups:
  * No commented-out debug print blocks.
  * `log.get('u', None)` prevents KeyError when mm_unbalanced is the solver.
"""
from __future__ import annotations

import ot
import numpy as np

from ot.optim import line_search_armijo

from .linesearch import solve_gromov_linesearch
from .conditional_gradient import cg_incent


def fused_gromov_wasserstein_incent(
    M1, M2, C1, C2, p, q, gamma,
    G_init=None,
    loss_fun: str = 'square_loss',
    alpha: float = 0.1,
    armijo: bool = False,
    log: bool = False,
    numItermax: int = 6000,
    tol_rel: float = 1e-9,
    tol_abs: float = 1e-9,
    use_gpu: bool = False,
    rho1: float = 1.0,
    rho2: float = 1.0,
    balanced_fallback_threshold: float = 1e6,
    **kwargs,
):
    """
    Fused Gromov-Wasserstein with KL-marginal unbalanced relaxation.

    Objective
    ---------
        min_G  (1 - alpha) * [<M1, G> + gamma * <M2, G>]
             + alpha * [<C1, GG^T> + <C2, G^T G>]
             + rho1 * KL(G1 || p)  +  rho2 * KL(G^T1 || q)

    Parameters
    ----------
    M1     : (ns, nt) cosine distance of gene expression
    M2     : (ns, nt) neighbourhood dissimilarity
    C1, C2 : (ns, ns) / (nt, nt) spatial pairwise distance matrices
    p, q   : source / target marginals
    gamma  : weight of the M2 neighbourhood term relative to M1
    G_init : (ns, nt) initial transport plan (None → outer(p,q))
    loss_fun : 'square_loss' or 'kl_loss'
    alpha  : weight of the GW (spatial structure) term; 0 = pure feature OT
    armijo : force Armijo step-size; auto-enabled for kl_loss
    log    : return dict with loss history and fgw_dist
    numItermax, tol_rel, tol_abs : convergence controls
    use_gpu : move G_init to CUDA
    rho1, rho2 : KL marginal penalty weights
    balanced_fallback_threshold : fall back to EMD when both rho ≥ this value
    """
    p, q = ot.utils.list_to_array(p, q)
    nx = ot.backend.get_backend(p, q, C1, C2, M1, M2)

    # ── initial plan ──────────────────────────────────────────────────────
    if G_init is None:
        G0 = p[:, None] * q[None, :]
    else:
        G0 = (1.0 / nx.sum(G_init)) * G_init
        if use_gpu:
            import torch
            if torch.cuda.is_available():
                G0 = G0.cuda()

    # ── GW structure functions ────────────────────────────────────────────
    def f(G):
        return nx.sum((G @ G.T) * C1) + nx.sum((G.T @ G) * C2)

    def df(G):
        # Gradient of f: 2*(C1 G + G C2)
        return 2 * (nx.dot(C1, G) + nx.dot(G, C2))

    # ── line-search strategy ──────────────────────────────────────────────
    if loss_fun == 'kl_loss':
        armijo = True

    if armijo:
        def line_search(cost, G, deltaG, Mi, cost_G, **kw):
            return line_search_armijo(cost, G, deltaG, Mi, cost_G, nx=nx, **kw)
    else:
        def line_search(cost, G, deltaG, Mi, cost_G, **kw):
            return solve_gromov_linesearch(
                G, deltaG, cost_G, C1, C2, M=0.0, reg=1.0, nx=nx, **kw
            )

    # ── solve ─────────────────────────────────────────────────────────────
    cg_kwargs = dict(
        G0=G0, line_search=line_search, log=True,
        numItermax=numItermax, stopThr=tol_rel, stopThr2=tol_abs,
        rho1=rho1, rho2=rho2,
        balanced_fallback_threshold=balanced_fallback_threshold,
        **kwargs,
    )

    res, res_log = cg_incent(
        p, q,
        (1 - alpha) * M1, (1 - alpha) * M2,
        alpha, f, df, gamma=gamma,
        **cg_kwargs,
    )

    res_log['fgw_dist'] = res_log['loss'][-1]
    # Dual variables are only produced by EMD; guard safely.
    res_log['u'] = res_log.get('u', None)
    res_log['v'] = res_log.get('v', None)

    if log:
        return res, res_log
    return res
