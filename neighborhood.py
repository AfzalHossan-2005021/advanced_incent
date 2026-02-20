"""
Cellular neighbourhood distribution computation.

Memory design
-------------
The previous version called euclidean_distances(coords, coords) which builds
an O(n^2) float64 matrix (~800 MB for 10 k cells).  With joblib threads each
holding a reference that memory is replicated and can exhaust all RAM.

This version uses sklearn.neighbors.BallTree.query_radius which returns only
the *indices* of cells within the radius -- O(n*k) memory where k is the
average neighbourhood size (typically k << n).
"""
from __future__ import annotations

import numpy as np
from anndata import AnnData
from sklearn.neighbors import BallTree
from tqdm import tqdm



def neighborhood_distribution(
    curr_slice: AnnData,
    radius: float,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Compute the cellular neighbourhood type-composition distribution for
    every cell in `curr_slice`.

    Uses BallTree.query_radius to avoid materialising the full (n, n)
    pairwise distance matrix -- O(n*k) memory instead of O(n^2).

    Parameters
    ----------
    curr_slice : AnnData
        Must contain `obsm['spatial']` and `obs['cell_type_annot']`.
    radius : float
        Neighbourhood radius in the same units as `obsm['spatial']`.
    n_jobs : int
        Kept for API compatibility (BallTree query is already efficient).

    Returns
    -------
    distributions : (n_cells, n_cell_types) float32 ndarray
    """
    cell_types = np.array(curr_slice.obs['cell_type_annot'])
    unique_types = np.unique(cell_types)
    type_to_idx = {ct: i for i, ct in enumerate(unique_types)}
    n_types = len(unique_types)
    n_cells = curr_slice.n_obs

    coords = curr_slice.obsm['spatial'].astype(np.float32)

    # BallTree returns variable-length arrays of neighbour indices per cell.
    # Peak memory: O(n*k) index arrays, not O(n^2) float64 distances.
    tree = BallTree(coords, leaf_size=40)
    neighbor_lists = tree.query_radius(coords, r=radius)   # list of n arrays

    result = np.zeros((n_cells, n_types), dtype=np.float32)
    for i in tqdm(range(n_cells), desc="Neighbourhood distribution"):
        for idx in neighbor_lists[i]:
            result[i, type_to_idx[cell_types[idx]]] += 1.0

    return result
