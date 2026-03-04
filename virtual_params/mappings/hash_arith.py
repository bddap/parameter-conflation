"""
Hash + Arithmetic mapping for parameter conflation.

Each virtual parameter is computed from a small number of actual parameters
selected by deterministic hash functions, combined with cheap arithmetic:

    virtual[j] = actual[h1(j)] * actual[h2(j)] + actual[h3(j)]

This is the simplest mapping that provides nonlinear mixing of actual params.

Properties:
    - O(1) per virtual param (3 lookups + 1 multiply + 1 add)
    - Nonlinear: multiplicative interaction between two actual params
    - Differentiable: product rule gives clean gradients
    - M actual params can produce up to ~M^3 distinct virtual params
    - No extra learned parameters
"""

import torch
from typing import Tuple, Dict
from ..core import VirtualParameterMap


class HashArithMap(VirtualParameterMap):
    """
    Hash + arithmetic parameter conflation.

    For each virtual param j, deterministically selects 3 actual params
    and computes: actual[h1] * actual[h2] + actual[h3]

    Caches hash indices per (slot_id, n_virtual) for speed.

    Args:
        num_actual: number of actual (stored) parameters
        init_std: initialization std for actual params
    """

    def __init__(self, num_actual: int, init_std: float = 0.02):
        super().__init__(num_actual, init_std=init_std)
        self._cache: Dict = {}

    def _get_cached(self, n_virtual: int, slot_id: int, device: torch.device):
        key = (slot_id, n_virtual)
        if key not in self._cache:
            M = self.num_actual
            gen = torch.Generator()
            gen.manual_seed(slot_id * 1000003 + 7)

            h1 = torch.randint(0, M, (n_virtual,), generator=gen)
            h2 = torch.randint(0, M, (n_virtual,), generator=gen)
            h3 = torch.randint(0, M, (n_virtual,), generator=gen)

            self._cache[key] = (h1, h2, h3)

        h1, h2, h3 = self._cache[key]
        return h1.to(device), h2.to(device), h3.to(device)

    def _compute_virtual(self, shape: Tuple[int, ...], slot_id: int) -> torch.Tensor:
        n_virtual = 1
        for s in shape:
            n_virtual *= s

        h1, h2, h3 = self._get_cached(n_virtual, slot_id, self.actual_params.device)

        a1 = self.actual_params[h1]
        a2 = self.actual_params[h2]
        a3 = self.actual_params[h3]

        virtual_flat = a1 * a2 + a3

        return virtual_flat.reshape(shape)

    def extra_repr(self) -> str:
        return f"num_actual={self.num_actual}"
