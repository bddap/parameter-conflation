"""
Deep Hash mapping for parameter conflation.

Like HashArith but with two levels of multiplicative interaction:

    Level 1: t1 = a1*a2 + a3,  t2 = a4*a5 + a6
    Level 2: virtual[j] = (t1 * t2 + a7) / norm

This uses 7 actual params per virtual param (vs 3 for HashArith).
The deeper nonlinearity produces more diverse virtual values from
the same actual param pool.

For actual ~ N(0, 1):
    Var(t) = Var(a*b + c) = 1 + 1 = 2     (each level-1 term)
    Var(t1*t2) = Var(t1)*Var(t2) = 4       (product of independent terms)
    Var(t1*t2 + a7) = 4 + 1 = 5
    norm = sqrt(5)

Properties:
    - O(1) per virtual param (7 lookups + 3 multiply + 3 add + normalize)
    - Two levels of nonlinearity for richer combinatorics
    - Same M actual params, but richer virtual param diversity
"""

import torch
import math
from typing import Tuple, Dict
from ..core import VirtualParameterMap


class DeepHashMap(VirtualParameterMap):
    """
    Two-level hash + arithmetic parameter conflation.

    virtual[j] = ((a1*a2 + a3) * (a4*a5 + a6) + a7) / sqrt(5)

    Uses 7 hash lookups per virtual param.

    Args:
        num_actual: number of actual (stored) parameters
        init_std: initialization std for actual params (default 1.0)
    """

    def __init__(self, num_actual: int, init_std: float = 1.0):
        super().__init__(num_actual, init_std=init_std)
        self._cache: Dict = {}
        # For init_std=s:
        # Var(level1) = s^4 + s^2
        # Var(level1_a * level1_b) = Var(level1)^2 = (s^4 + s^2)^2
        # Var(result) = (s^4 + s^2)^2 + s^2
        v1 = init_std**4 + init_std**2
        self._norm = math.sqrt(v1 * v1 + init_std**2)

    def _get_cached(self, n_virtual: int, slot_id: int, device: torch.device):
        cache_key = (slot_id, n_virtual, str(device))
        if cache_key not in self._cache:
            M = self.num_actual
            gen = torch.Generator()
            gen.manual_seed(slot_id * 1000003 + 13)

            indices = []
            for _ in range(7):
                indices.append(torch.randint(0, M, (n_virtual,), generator=gen).to(device))

            self._cache[cache_key] = tuple(indices)
        return self._cache[cache_key]

    def _compute_virtual(self, shape: Tuple[int, ...], slot_id: int) -> torch.Tensor:
        n_virtual = 1
        for s in shape:
            n_virtual *= s

        h1, h2, h3, h4, h5, h6, h7 = self._get_cached(
            n_virtual, slot_id, self.actual_params.device
        )

        a1 = self.actual_params[h1]
        a2 = self.actual_params[h2]
        a3 = self.actual_params[h3]
        a4 = self.actual_params[h4]
        a5 = self.actual_params[h5]
        a6 = self.actual_params[h6]
        a7 = self.actual_params[h7]

        t1 = a1 * a2 + a3
        t2 = a4 * a5 + a6
        virtual_flat = (t1 * t2 + a7) / self._norm

        return virtual_flat.reshape(shape)

    def clear_cache(self) -> None:
        """Free cached index tensors."""
        self._cache.clear()

    def extra_repr(self) -> str:
        return f"num_actual={self.num_actual}, norm={self._norm:.4f}"
