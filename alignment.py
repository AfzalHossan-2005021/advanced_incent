"""
Main alignment entry point for advanced_incent.

Design decisions
----------------
* All hyperparameters are bundled in an `AlignConfig` dataclass — no
  long positional argument lists.
* Device management (CPU / best GPU) is handled by `DeviceManager`.
* Logging uses Python's standard `logging` module instead of manual
  `open(log.txt, 'w')` file I/O.
* Cell-type aware G₀ initialisation is always attempted when the
  `cell_type_init` flag is set in config (fills the original `# todo`).
* Neighbourhood distributions and all cost matrices are cached to disk
  so that repeated calls with `overwrite=False` skip heavy computation.
"""
from __future__ import annotations

import gc
import logging
import os
import time
from typing import Optional, Tuple, Union

import numpy as np
import ot
import torch
from anndata import AnnData
from numpy.typing import NDArray

from .config import AlignConfig
from .distances import (
    cosine_distance,
    cosine_dist_neighborhood,
    jensenshannon_divergence_backend,
    pairwise_msd,
)
from .initialization import cell_type_aware_init
from .neighborhood import neighborhood_distribution
from .transport import fused_gromov_wasserstein_incent
from .utils import DeviceManager, to_dense_array, extract_data_matrix


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _setup_file_logger(filePath: str, sliceA_name: str, sliceB_name: str) -> logging.FileHandler:
    """Return a file handler that writes to filePath/log_A_B.txt."""
    os.makedirs(filePath, exist_ok=True)
    fh = logging.FileHandler(
        os.path.join(filePath, f"log_{sliceA_name}_{sliceB_name}.txt"), mode='w'
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s  %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(fh)
    return fh


def _load_or_compute_neighborhood(
    sl: AnnData, name: str, filePath: str, config: AlignConfig
) -> np.ndarray:
    """Load cached neighbourhood distribution or compute it with joblib parallelism."""
    cache = os.path.join(filePath, f"neighborhood_distribution_{name}.npy")
    if os.path.exists(cache) and not config.overwrite:
        logger.info("Loading cached neighbourhood distribution for %s", name)
        nd = np.load(cache)
    else:
        logger.info("Computing neighbourhood distribution for %s", name)
        nd = neighborhood_distribution(sl, radius=config.radius, n_jobs=config.n_jobs)
        np.save(cache, nd)
    nd = nd + 0.01   # avoid zero-division in JSD / cosine
    return nd


def _compute_M2(
    ndA: np.ndarray, ndB: np.ndarray,
    sliceA_name: str, sliceB_name: str,
    filePath: str, config: AlignConfig, dm: DeviceManager,
) -> np.ndarray:
    """Compute neighbourhood dissimilarity matrix M2."""
    diss = config.neighborhood_dissimilarity

    if diss == 'jsd':
        cache = os.path.join(filePath, f"js_dist_neighborhood_{sliceA_name}_{sliceB_name}.npy")
        if os.path.exists(cache) and not config.overwrite:
            logger.info("Loading cached JSD neighbourhood matrix")
            return np.load(cache)
        logger.info("Computing JSD neighbourhood matrix (chunked, device=%s)", dm.device)
        M2 = jensenshannon_divergence_backend(
            ndA, ndB, chunk_size=config.jsd_chunk_size, device=dm.device
        )
        np.save(cache, M2)
        return M2

    if diss == 'cosine':
        logger.info("Computing cosine neighbourhood matrix")
        return cosine_dist_neighborhood(ndA, ndB, device=dm.device)

    if diss == 'msd':
        logger.info("Computing MSD neighbourhood matrix")
        return pairwise_msd(ndA, ndB)

    raise ValueError(
        f"Invalid neighborhood_dissimilarity {diss!r}. "
        "Expected one of {'jsd', 'cosine', 'msd'}."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def pairwise_align(
    sliceA: AnnData,
    sliceB: AnnData,
    config: AlignConfig = None,
    *,
    sliceA_name: str = "sliceA",
    sliceB_name: str = "sliceB",
    filePath: str = "outputs",
    G_init: Optional[np.ndarray] = None,
    a_distribution: Optional[np.ndarray] = None,
    b_distribution: Optional[np.ndarray] = None,
    return_obj: bool = False,
) -> Union[NDArray[np.floating], Tuple]:
    """
    Compute the optimal transport plan between two spatial transcriptomics slices.

    Parameters
    ----------
    sliceA, sliceB : AnnData
        Source and target slices.  Must have:
          * obsm['spatial'] — 2-D coordinates
          * obs['cell_type_annot'] — cell-type labels
          * X (or obsm[config.use_rep]) — gene expression matrix
    config : AlignConfig, optional
        All hyperparameters.  Defaults to AlignConfig() if not provided.
    sliceA_name, sliceB_name : str
        Labels used for cache file names and log messages.
    filePath : str
        Directory for cached matrices and the log file.
    G_init : (ns, nt) ndarray, optional
        Custom initial transport plan.  Overrides cell_type_init.
    a_distribution, b_distribution : (ns,) / (nt,) ndarray, optional
        Custom source / target marginals.  Default: uniform.
    return_obj : bool
        If True returns (pi, init_obj_neighbor, init_obj_gene,
                          final_obj_neighbor, final_obj_gene).

    Returns
    -------
    pi : (ns, nt) ndarray — optimal transport plan
    (optionally followed by objective values when return_obj=True)
    """
    if config is None:
        config = AlignConfig()

    # ── logging ─────────────────────────────────────────────────────────
    fh = _setup_file_logger(filePath, sliceA_name, sliceB_name)
    logger.setLevel(logging.DEBUG if config.verbose else logging.INFO)

    logger.info("pairwise_align  |  %s → %s", sliceA_name, sliceB_name)
    logger.info(
        "alpha=%.3f  beta=%.3f  gamma=%.3f  radius=%.1f  rho1=%.3g  rho2=%.3g  "
        "dissimilarity=%s  n_jobs=%d",
        config.alpha, config.beta, config.gamma, config.radius,
        config.rho1, config.rho2, config.neighborhood_dissimilarity, config.n_jobs,
    )

    start = time.time()

    # ── validate input ───────────────────────────────────────────────────
    for s in (sliceA, sliceB):
        if not len(s):
            raise ValueError(f"Found empty AnnData: {s}")

    # ── device manager (GPU used only for JSD preprocessing) ─────────────
    # The FGW solve keeps ALL matrices as CPU numpy to avoid:
    #   1. GPU OOM  — FGW matmuls create many (ns×nt) temporaries on VRAM
    #   2. RAM doubling — dm.to() would copy each matrix into both RAM and VRAM
    dm = DeviceManager(use_gpu=config.use_gpu)
    logger.info("Preprocessing device: %s  |  FGW solve: CPU", dm.device)

    # ── spatial distance matrices (CPU float32) ───────────────────────────
    coordsA = sliceA.obsm['spatial'].astype(np.float32)
    coordsB = sliceB.obsm['spatial'].astype(np.float32)
    D_A = ot.dist(coordsA, coordsA, metric='euclidean').astype(np.float32)
    D_B = ot.dist(coordsB, coordsB, metric='euclidean').astype(np.float32)

    if config.norm:
        D_A /= D_A[D_A > 0].min()
        D_B /= D_B[D_B > 0].min()

    # ── M1: gene-expression cosine distance (CPU float32) ─────────────────
    cosine_arr = cosine_distance(
        sliceA, sliceB, sliceA_name, sliceB_name, filePath,
        use_rep=config.use_rep, use_gpu=config.use_gpu,
        beta=config.beta, overwrite=config.overwrite,
    ).astype(np.float32)
    M1 = cosine_arr  # stays on CPU — no dm.to()

    # ── neighbourhood distributions ───────────────────────────────────────
    ndA = _load_or_compute_neighborhood(sliceA, sliceA_name, filePath, config)
    ndB = _load_or_compute_neighborhood(sliceB, sliceB_name, filePath, config)

    # ── M2: neighbourhood dissimilarity (dm.device used for chunked JSD) ──
    M2_arr = _compute_M2(ndA, ndB, sliceA_name, sliceB_name, filePath, config, dm).astype(np.float32)
    del ndA, ndB   # free histograms; no longer needed
    gc.collect()
    if config.use_gpu:
        torch.cuda.empty_cache()  # release any GPU scratch from JSD
    M2 = M2_arr  # stays on CPU — no dm.to()

    # ── marginals (CPU float32) ───────────────────────────────────────────
    ns, nt = sliceA.n_obs, sliceB.n_obs
    a = a_distribution.astype(np.float32) if a_distribution is not None else np.ones(ns, dtype=np.float32) / ns
    b = b_distribution.astype(np.float32) if b_distribution is not None else np.ones(nt, dtype=np.float32) / nt

    # ── initial transport plan (CPU float32) ─────────────────────────────
    if G_init is not None:
        G0 = G_init.astype(np.float32)
    elif config.cell_type_init:
        logger.info("Building cell-type aware G₀ (weight=%.1f)", config.cell_type_init_weight)
        G0 = cell_type_aware_init(
            sliceA, sliceB, a, b, weight=config.cell_type_init_weight
        ).astype(np.float32)
    else:
        G0 = None   # fgw will fall back to outer(a, b)

    # ── initial-objective diagnostic ─────────────────────────────────────
    # Use mean() instead of allocating a full (ns, nt) G_unif matrix:
    # sum(M * G_unif) = sum(M) / (ns*nt) = mean(M)
    init_obj_gene = float(np.mean(cosine_arr))
    init_obj_neighbor = float(np.mean(M2_arr))
    logger.info("Initial objective — gene: %.6g  |  neighbour: %.6g",
                init_obj_gene, init_obj_neighbor)

    # ── solve FGW ────────────────────────────────────────────────────────
    # All inputs are numpy float32 → POT uses NumpyBackend automatically → no GPU
    pi, log_dict = fused_gromov_wasserstein_incent(
        M1, M2, D_A, D_B, a, b,
        gamma=config.gamma,
        G_init=G0,
        loss_fun='square_loss',
        alpha=config.alpha,
        log=True,
        numItermax=config.numItermax,
        tol_rel=config.tol_rel,
        tol_abs=config.tol_abs,
        use_gpu=False,   # FGW always on CPU; GPU was already freed after JSD
        rho1=config.rho1,
        rho2=config.rho2,
        balanced_fallback_threshold=config.balanced_fallback_threshold,
    )
    # pi is already a numpy array (NumpyBackend output)
    pi = np.asarray(pi, dtype=np.float32)

    # ── save transport plan ───────────────────────────────────────────────
    np.save(os.path.join(filePath, f"pi_matrix_{sliceA_name}_{sliceB_name}.npy"), pi)

    # ── final objectives ─────────────────────────────────────────────────
    final_obj_gene = float(np.sum(cosine_arr * pi, dtype=np.float64))
    final_obj_neighbor = float(np.sum(M2_arr * pi, dtype=np.float64))

    # ── marginal violation (how far from balanced) ───────────────────────
    src_viol = float(np.sum(np.abs(pi.sum(axis=1) - a)))
    tgt_viol = float(np.sum(np.abs(pi.sum(axis=0) - b)))

    elapsed = time.time() - start
    logger.info("Final objective   — gene: %.6g  |  neighbour: %.6g", final_obj_gene, final_obj_neighbor)
    logger.info("Marginal violation — source L1: %.4e  |  target L1: %.4e", src_viol, tgt_viol)
    logger.info("Runtime: %.2f s", elapsed)

    logger.removeHandler(fh)
    fh.close()

    if config.use_gpu:
        torch.cuda.empty_cache()

    if return_obj:
        return pi, init_obj_neighbor, init_obj_gene, final_obj_neighbor, final_obj_gene
    return pi
