"""
Linesearch for the Frank-Wolfe step in the FGW solver.

Identical to unbalanced_incent — the quadratic line-search maths is correct
and requires no change for the unbalanced extension.
"""
from __future__ import annotations

import numpy as np
import ot


def solve_gromov_linesearch(
    G,
    deltaG,
    cost_G,
    C1,
    C2,
    M,
    reg,
    alpha_min=None,
    alpha_max=None,
    nx=None,
    **kwargs,
):
    """
    Solve the quadratic linesearch in the FW iterations.

    Parameters
    ----------
    G       : (ns, nt) array — transport map at current iteration
    deltaG  : (ns, nt) array — FW direction (Gc - G)
    cost_G  : float          — current objective value
    C1      : (ns, ns) array — source structure matrix
    C2      : (nt, nt) array — target structure matrix
    M       : (ns, nt) array — linear cost matrix (pass 0. for pure GW)
    reg     : float          — GW regularisation weight
    alpha_min, alpha_max : float, optional — step-size clipping
    nx      : ot.backend, optional

    Returns
    -------
    alpha  : float — optimal step size
    fc     : int   — function calls (always 1 for the quadratic formula)
    cost_G : float — objective value at G + alpha * deltaG
    """
    if nx is None:
        G, deltaG, C1, C2, M = ot.utils.list_to_array(G, deltaG, C1, C2, M)
        if isinstance(M, (int, float)):
            nx = ot.backend.get_backend(G, deltaG, C1, C2)
        else:
            nx = ot.backend.get_backend(G, deltaG, C1, C2, M)

    dot = nx.dot(nx.dot(C1, deltaG), C2.T)
    a = -2 * reg * nx.sum(dot * deltaG)
    b = nx.sum(M * deltaG) - 2 * reg * (
        nx.sum(dot * G) + nx.sum(nx.dot(nx.dot(C1, G), C2.T) * deltaG)
    )

    alpha = ot.optim.solve_1d_linesearch_quad(a, b)
    if alpha_min is not None or alpha_max is not None:
        alpha = np.clip(alpha, alpha_min, alpha_max)

    cost_G = cost_G + a * (alpha ** 2) + b * alpha
    return alpha, 1, cost_G
