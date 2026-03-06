"""
Hash + Arithmetic mapping for parameter conflation.

Each virtual parameter is computed from a small number of actual parameters
selected by deterministic hash functions, combined with cheap arithmetic:

    virtual[j] = actual[h1(j)] * actual[h2(j)] + actual[h3(j)]

The output is normalized so that when actual_params ~ N(0, 1), virtual
params also have approximately unit variance.

For actual ~ N(0, sigma):
    Var(a1*a2) = sigma^4  (product of independent zero-mean normals)
    Var(a3) = sigma^2
    Var(a1*a2 + a3) = sigma^4 + sigma^2

We normalize by 1/sqrt(sigma^4 + sigma^2) at init, but this changes during
training. For init_std=1: Var = 1 + 1 = 2, so we divide by sqrt(2).

Properties:
    - O(1) per virtual param (3 lookups + 1 multiply + 1 add + normalize)
    - Nonlinear: multiplicative interaction between two actual params
    - Differentiable: product rule gives clean gradients
    - M actual params can produce up to ~M^3 distinct virtual params
    - No extra learned parameters
"""

import torch
import math
from typing import Tuple, Dict
from ..core import VirtualParameterMap


class HashArithMap(VirtualParameterMap):
    """
    Hash + arithmetic parameter conflation.

    For each virtual param j, deterministically selects 3 actual params
    and computes: actual[h1] * actual[h2] + actual[h3], normalized to
    approximately unit variance.

    Caches hash indices per (slot_id, n_virtual) for speed.

    Args:
        num_actual: number of actual (stored) parameters
        init_std: initialization std for actual params (default 1.0)
    """

    def __init__(self, num_actual: int, init_std: float = 1.0):
        super().__init__(num_actual, init_std=init_std)
        self._cache: Dict = {}
        # Normalization factor: for init_std=s, Var = s^4 + s^2
        # We store it so it's fixed at init (doesn't track training drift)
        self._norm = math.sqrt(init_std ** 4 + init_std ** 2)

    def _get_cached(self, n_virtual: int, slot_id: int, device: torch.device):
        cache_device_key = (slot_id, n_virtual, str(device))

        if cache_device_key not in self._cache:
            M = self.num_actual
            gen = torch.Generator()
            gen.manual_seed(slot_id * 1000003 + 7)

            h1 = torch.randint(0, M, (n_virtual,), generator=gen).to(device)
            h2 = torch.randint(0, M, (n_virtual,), generator=gen).to(device)
            h3 = torch.randint(0, M, (n_virtual,), generator=gen).to(device)

            self._cache[cache_device_key] = (h1, h2, h3)

        return self._cache[cache_device_key]

    def _compute_virtual(self, shape: Tuple[int, ...], slot_id: int) -> torch.Tensor:
        n_virtual = 1
        for s in shape:
            n_virtual *= s

        h1, h2, h3 = self._get_cached(n_virtual, slot_id, self.actual_params.device)

        a1 = self.actual_params[h1]
        a2 = self.actual_params[h2]
        a3 = self.actual_params[h3]

        virtual_flat = (a1 * a2 + a3) / self._norm

        return virtual_flat.reshape(shape)

    def clear_cache(self) -> None:
        """Free cached index tensors."""
        self._cache.clear()

    def extra_repr(self) -> str:
        return f"num_actual={self.num_actual}, norm={self._norm:.4f}"
