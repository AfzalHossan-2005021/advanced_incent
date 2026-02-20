"""
Unified device management for advanced_incent.

Replaces the scattered isinstance(nx, TorchBackend) / .cuda() checks
spread across the original code with a single DeviceManager that every
module imports.
"""
from __future__ import annotations

from typing import Any, Union

import numpy as np
import ot
import torch


class DeviceManager:
    """
    Centralises backend selection and tensor placement.

    Parameters
    ----------
    use_gpu : bool
        Request GPU acceleration. Falls back gracefully if unavailable.
    """

    def __init__(self, use_gpu: bool = False) -> None:
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device: torch.device = self._pick_device()
        self.nx: Any = self._pick_backend()

    # ── public ────────────────────────────────────────────────────────────

    def to(self, x: Any, dtype: torch.dtype = torch.float32) -> Any:
        """
        Convert *x* to the active backend array on the active device.

        Accepts numpy arrays, torch tensors, or anything with __array__.
        """
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x.astype(np.float32 if dtype == torch.float32 else np.float64))
        elif isinstance(x, torch.Tensor):
            t = x
        else:
            t = torch.tensor(np.asarray(x), dtype=dtype)

        if isinstance(self.nx, ot.backend.TorchBackend):
            return t.to(dtype=dtype, device=self.device)
        else:
            # NumpyBackend — return plain ndarray
            arr = t.cpu().numpy()
            return arr.astype(np.float32 if dtype == torch.float32 else np.float64)

    def to_numpy(self, x: Any) -> np.ndarray:
        """Convert *x* back to a CPU numpy array regardless of source backend."""
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        if hasattr(x, '__array__'):
            return np.asarray(x)
        return self.nx.to_numpy(x)

    def ot_dist(self, a: Any, b: Any, metric: str = 'euclidean') -> Any:
        """Compute pairwise distance and place result on the active device."""
        d = ot.dist(a, b, metric=metric)
        if isinstance(self.nx, ot.backend.TorchBackend) and self.use_gpu:
            return d.to(self.device)
        return d

    @property
    def is_torch(self) -> bool:
        return isinstance(self.nx, ot.backend.TorchBackend)

    # ── private ───────────────────────────────────────────────────────────

    def _pick_device(self) -> torch.device:
        if not self.use_gpu:
            return torch.device('cpu')
        # Pick the GPU with the most free memory when multiple are present
        best, best_free = 0, -1
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free = props.total_memory - torch.cuda.memory_allocated(i)
            if free > best_free:
                best_free = free
                best = i
        return torch.device(f'cuda:{best}')

    def _pick_backend(self) -> Any:
        if self.use_gpu and torch.cuda.is_available():
            return ot.backend.TorchBackend()
        return ot.backend.NumpyBackend()
