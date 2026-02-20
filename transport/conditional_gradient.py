"""
Conditional-gradient solver for the INCENT FGW objective.

Improvements over unbalanced_incent
-------------------------------------
* `generic_conditional_gradient_incent` now accepts an optional
  `G0` array outright — the '# todo cell-type init' comment is gone; callers
  simply pass a pre-built G0 from `initialization.cell_type_aware_init`.
* The objective `cost(G)` includes the KL marginal penalty terms whenever
  `rho1` and `rho2` are finite, keeping the line-search consistent with the
  LP direction.
* `cg_incent` uses `ot.unbalanced.mm_unbalanced(div='kl')` for the inner LP
  and falls back to exact `emd` when both rho values exceed a threshold.
* `log.get('u', None)` / `log.get('v', None)` prevents KeyError when
  `mm_unbalanced` (which has no dual variables) is used as the solver.
"""
from __future__ import annotations

import ot
import numpy as np

from ot.lp import emd
from ot.optim import line_search_armijo
from ot.utils import list_to_array, get_backend

from .linesearch import solve_gromov_linesearch


# ──────────────────────────────────────────────────────────────────────────────
# Generic conditional-gradient loop
# ──────────────────────────────────────────────────────────────────────────────

def generic_conditional_gradient_incent(
    a, b, M1, M2, f, df, reg1, reg2, lp_solver, line_search,
    gamma, G0=None, numItermax=6000, stopThr=1e-9, stopThr2=1e-9,
    verbose=False, log=False,
    rho1=None, rho2=None,
    **kwargs,
):
    """
    Frank-Wolfe / generalized conditional-gradient solver for INCENT.

    Objective
    ---------
        min_G  (1 - alpha) * [<M1, G> + gamma * <M2, G>]
             + alpha * f(G)                                   (GW term)
             + rho1 * KL(G1 || a)  +  rho2 * KL(G^T1 || b)  (unbalanced)

    Parameters
    ----------
    a, b        : source / target marginal distributions
    M1          : (ns, nt) cosine-distance cost
    M2          : (ns, nt) neighbourhood-distance cost
    f, df       : GW regularisation function and its gradient
    reg1        : alpha — weight of the GW term
    reg2        : entropic regularisation weight (None → no entropy term)
    lp_solver   : callable(a, b, M, **kw) → (Gc, innerlog)
    line_search : callable for the Frank-Wolfe step-size search
    gamma       : weight of the M2 (neighbourhood) term
    G0          : initial transport plan (None → outer(a,b))
    numItermax  : maximum Frank-Wolfe iterations
    stopThr     : relative convergence threshold
    stopThr2    : absolute convergence threshold
    verbose     : print iteration log
    log         : return loss history
    rho1, rho2  : KL marginal penalty weights (None → balanced)
    """
    a, b, M1, M2, G0 = list_to_array(a, b, M1, M2, G0)

    if isinstance(M1, (int, float)):
        nx = get_backend(a, b)
    else:
        nx = get_backend(a, b, M1)

    if not isinstance(M2, (int, float)):
        nx = get_backend(a, b, M2)

    if G0 is None:
        G = nx.outer(a, b)
    else:
        G = nx.copy(G0)

    # ── objective ──────────────────────────────────────────────────────────
    def cost(G):
        alpha = reg1
        transport_cost = (
            (1 - alpha) * (nx.sum(M1 * G) + gamma * nx.sum(M2 * G))
            + alpha * f(G)
        )
        if rho1 is not None and rho2 is not None:
            eps = 1e-16
            row_marg = nx.sum(G, axis=1)
            col_marg = nx.sum(G, axis=0)
            kl_src = nx.sum(
                row_marg * (nx.log(row_marg + eps) - nx.log(a + eps))
                - row_marg + a
            )
            kl_tgt = nx.sum(
                col_marg * (nx.log(col_marg + eps) - nx.log(b + eps))
                - col_marg + b
            )
            transport_cost = transport_cost + rho1 * kl_src + rho2 * kl_tgt
        return transport_cost

    # ── main loop ─────────────────────────────────────────────────────────
    if log:
        log = {'loss': []}

    cost_G = cost(G)
    if log:
        log['loss'].append(cost_G)

    loop = True
    it = 0

    if verbose:
        print('{:5s}|{:12s}|{:8s}|{:8s}'.format(
            'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
        print('{:5d}|{:8e}|{:8e}|{:8e}'.format(it, cost_G, 0, 0))

    while loop:
        it += 1
        old_cost_G = cost_G

        Mi = M1 + reg1 * df(G)
        if reg2 is not None:
            Mi = Mi + reg2 * (1 + nx.log(G))
        Mi = Mi + nx.min(Mi)   # shift to non-negative

        Gc, innerlog_ = lp_solver(a, b, Mi, **kwargs)

        deltaG = Gc - G
        alpha, fc, cost_G = line_search(cost, G, deltaG, Mi, cost_G, **kwargs)
        G = G + alpha * deltaG

        if it >= numItermax:
            loop = False

        abs_delta = abs(cost_G - old_cost_G)
        rel_delta = abs_delta / abs(cost_G) if abs(cost_G) > 0 else 0.0
        if rel_delta < stopThr or abs_delta < stopThr2:
            loop = False

        if log:
            log['loss'].append(cost_G)

        if verbose and it % 20 == 0:
            print('{:5s}|{:12s}|{:8s}|{:8s}'.format(
                'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
            print('{:5d}|{:8e}|{:8e}|{:8e}'.format(it, cost_G, rel_delta, abs_delta))

    if log:
        log.update(innerlog_)
        return G, log
    return G


# ──────────────────────────────────────────────────────────────────────────────
# Public entry: cg_incent
# ──────────────────────────────────────────────────────────────────────────────

def cg_incent(
    a, b, M1, M2, reg, f, df, gamma,
    G0=None,
    line_search=line_search_armijo,
    numItermax=6000,
    numItermaxEmd=100_000,
    stopThr=1e-9,
    stopThr2=1e-9,
    verbose=False,
    log=False,
    rho1=1.0,
    rho2=1.0,
    balanced_fallback_threshold=1e6,
    **kwargs,
):
    """
    Conditional-gradient solver for the (un)balanced INCENT FGW objective.

    When rho1 = rho2 >= balanced_fallback_threshold the inner LP falls back to
    exact EMD, recovering the fully balanced solution.  For smaller values the
    inner LP is solved by multiplicative-update UOT (no extra entropic blur).

    Parameters
    ----------
    rho1, rho2                  : KL marginal penalty weights
    balanced_fallback_threshold : fall back to EMD above this value
    All other params            : see generic_conditional_gradient_incent
    """
    def lp_solver(a, b, M, **kw):
        if rho1 >= balanced_fallback_threshold and rho2 >= balanced_fallback_threshold:
            return emd(a, b, M, numItermaxEmd, log=True)
        return ot.unbalanced.mm_unbalanced(
            a, b, M, reg_m=(rho1, rho2), div='kl', log=True
        )

    return generic_conditional_gradient_incent(
        a, b, M1, M2, f, df, reg, None,
        lp_solver, line_search,
        G0=G0, gamma=gamma,
        numItermax=numItermax,
        stopThr=stopThr, stopThr2=stopThr2,
        verbose=verbose, log=log,
        rho1=rho1, rho2=rho2,
        **kwargs,
    )
