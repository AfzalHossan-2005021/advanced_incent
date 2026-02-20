from typing import Any

import numpy as np
import ot
import scipy.sparse
import torch

from ot.utils import get_backend


def extract_data_matrix(adata, use_rep=None):
    """Return the gene-expression matrix from an AnnData object."""
    if use_rep is None:
        return adata.X
    return adata.obsm[use_rep]


def to_dense_array(X) -> np.ndarray:
    """Convert sparse or dense array to a dense numpy array."""
    if scipy.sparse.issparse(X):
        return X.toarray()
    return np.array(X)


def to_numpy(x: Any) -> np.ndarray:
    """Convert any backend array to a CPU numpy array."""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if hasattr(x, '__array__'):
        return np.asarray(x)
    nx = get_backend(x)
    return nx.to_numpy(x)
