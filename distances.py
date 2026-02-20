"""
Distance / cost-matrix computations for advanced_incent.

Key improvements over the original code
----------------------------------------
jensenshannon_divergence_backend:
    The original uses an O(n) Python for-loop, calling a helper once per
    source cell.  This implementation processes `chunk_size` rows at a time
    as a single (chunk, m, k) tensor operation, eliminating the loop and
    making full use of GPU SIMD / tensor-core throughput.

cosine_distance:
    Unchanged logic; minor clean-up (top-level imports, no nested imports).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm

from .utils import to_dense_array, extract_data_matrix


# ──────────────────────────────────────────────────────────────────────────────
# Jensen-Shannon distance  (vectorised, chunked)
# ──────────────────────────────────────────────────────────────────────────────

def _js_chunk(P: torch.Tensor, Q: torch.Tensor, eps: float = 1e-16) -> torch.Tensor:
    """
    Vectorised JS distance between every row of P and every row of Q.

    Parameters
    ----------
    P : (cs, k) float32 tensor  — normalised chunk of source distributions
    Q : (m,  k) float32 tensor  — all target distributions

    Returns
    -------
    js : (cs, m) float32 tensor
    """
    # Broadcast: (cs, 1, k) vs (1, m, k)
    P3 = P.unsqueeze(1)   # (cs, 1, k)
    Q3 = Q.unsqueeze(0)   # (1,  m, k)
    M = (P3 + Q3) * 0.5   # (cs, m, k)

    kl_PM = (P3 * torch.log((P3 + eps) / (M + eps))).sum(dim=-1)  # (cs, m)
    kl_QM = (Q3 * torch.log((Q3 + eps) / (M + eps))).sum(dim=-1)  # (cs, m)

    js = torch.sqrt(torch.clamp((kl_PM + kl_QM) * 0.5, min=0.0))
    return js


def jensenshannon_divergence_backend(
    X: np.ndarray,
    Y: np.ndarray,
    chunk_size: int = 64,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Pairwise Jensen-Shannon distance matrix between X and Y — fully batched.

    Unlike the original row-by-row for-loop, this function:
    * Processes `chunk_size` source rows simultaneously as a 3-D tensor.
    * Runs the entire computation on `device` (GPU if available).
    * Returns a float32 numpy array to keep memory usage low.

    Parameters
    ----------
    X : (n, k) array   — source neighbourhood distributions
    Y : (m, k) array   — target neighbourhood distributions
    chunk_size : int   — rows of X processed per GPU kernel call.
                         Reduce this if you get CUDA out-of-memory.
    device : torch.device or None
             Use this device; falls back to CUDA if available, else CPU.

    Returns
    -------
    js_dist : (n, m) float32 numpy array
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Calculating JSD cost matrix on {device} "
          f"(chunk_size={chunk_size}, n={X.shape[0]}, m={Y.shape[0]})")

    # Normalise — avoid zero-rows with small epsilon before division
    eps = 1e-16
    X_arr = np.asarray(X, dtype=np.float32)
    Y_arr = np.asarray(Y, dtype=np.float32)
    X_arr = X_arr / (X_arr.sum(axis=1, keepdims=True) + eps)
    Y_arr = Y_arr / (Y_arr.sum(axis=1, keepdims=True) + eps)

    # Move full Y to device once
    Q = torch.from_numpy(Y_arr).to(device)   # (m, k)

    n, m = X_arr.shape[0], Y_arr.shape[0]
    result = np.empty((n, m), dtype=np.float32)

    for start in tqdm(range(0, n, chunk_size), desc="JSD chunks"):
        end = min(start + chunk_size, n)
        P_chunk = torch.from_numpy(X_arr[start:end]).to(device)  # (cs, k)
        js_chunk = _js_chunk(P_chunk, Q)                          # (cs, m)
        result[start:end] = js_chunk.cpu().numpy()

    print("Finished calculating JSD cost matrix")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Pairwise mean-squared distance
# ──────────────────────────────────────────────────────────────────────────────

def pairwise_msd(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Pairwise mean squared distance between rows of A (m,d) and B (n,d).

    Returns (m, n) float64 numpy array.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]  # (m, n, d)
    return np.mean(diff ** 2, axis=2)                   # (m, n)


# ──────────────────────────────────────────────────────────────────────────────
# Cosine distance of gene expression (unchanged logic)
# ──────────────────────────────────────────────────────────────────────────────

def cosine_distance(
    sliceA: AnnData,
    sliceB: AnnData,
    sliceA_name: str,
    sliceB_name: str,
    filePath: str,
    use_rep: Optional[str] = None,
    use_gpu: bool = False,
    beta: float = 0.8,
    overwrite: bool = False,
) -> np.ndarray:
    """
    Cosine distance between cells of sliceA and sliceB in the joint
    gene-expression + cell-type space.

    The cell-type one-hot encoding is appended to the expression matrix
    with weight `beta`, so that cell-type identity contributes to the
    distance without dominating it.

    Parameters
    ----------
    sliceA, sliceB : AnnData
    sliceA_name, sliceB_name : str — used for cache file names
    filePath : str — directory for cached .npy files
    use_rep : str or None — use obsm[use_rep] instead of X
    use_gpu : bool — move data to CUDA for the concatenation step
    beta : float — scale of cell-type one-hot block
    overwrite : bool — ignore cached files

    Returns
    -------
    cosine_dist_gene_expr : (ns, nt) float64 numpy array
    """
    fileName = os.path.join(filePath, f"cosine_dist_gene_expr_{sliceA_name}_{sliceB_name}.npy")

    if os.path.exists(fileName) and not overwrite:
        print("Loading cached cosine distance of gene expression")
        return np.load(fileName)

    print("Calculating cosine distance of gene expression")

    # ── raw expression matrices ──────────────────────────────────────────
    s_A = to_dense_array(extract_data_matrix(sliceA, use_rep)).astype(np.float32) + 0.01
    s_B = to_dense_array(extract_data_matrix(sliceB, use_rep)).astype(np.float32) + 0.01

    # ── cell-type one-hot ────────────────────────────────────────────────
    oh_A = pd.get_dummies(sliceA.obs['cell_type_annot']).to_numpy(dtype=np.float32)
    oh_B = pd.get_dummies(sliceB.obs['cell_type_annot']).to_numpy(dtype=np.float32)

    # ── concatenate expression + beta * one-hot ──────────────────────────
    s_A = np.concatenate([s_A, beta * oh_A], axis=1)
    s_B = np.concatenate([s_B, beta * oh_B], axis=1)

    # ── move to CUDA if available ─────────────────────────────────────────
    device = torch.device('cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu')
    if device.type == 'cuda':
        print("CUDA is available — computing on GPU.")
        s_A_np = torch.from_numpy(s_A).to(device).cpu().numpy()
        s_B_np = torch.from_numpy(s_B).to(device).cpu().numpy()
    else:
        print("CUDA is not available — computing on CPU.")
        s_A_np, s_B_np = s_A, s_B

    cosine_dist_gene_expr = cosine_distances(s_A_np, s_B_np)

    print("Saving cosine distance of gene expression")
    np.save(fileName, cosine_dist_gene_expr)

    return cosine_dist_gene_expr


# ──────────────────────────────────────────────────────────────────────────────
# Cosine distance for neighbourhood distributions
# ──────────────────────────────────────────────────────────────────────────────

def cosine_dist_neighborhood(
    ndA: np.ndarray,
    ndB: np.ndarray,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Cosine distance between neighbourhood distribution matrices.

    Parameters
    ----------
    ndA : (n, k) array
    ndB : (m, k) array
    device : torch.device or None

    Returns
    -------
    dist : (n, m) float32 numpy array
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    A = torch.from_numpy(np.asarray(ndA, dtype=np.float32)).to(device)
    B = torch.from_numpy(np.asarray(ndB, dtype=np.float32)).to(device)

    numerator = A @ B.T                                         # (n, m)
    denom = A.norm(dim=1, keepdim=True) * B.norm(dim=1, keepdim=True).T
    dist = 1.0 - numerator / (denom + 1e-16)
    return dist.cpu().numpy()
