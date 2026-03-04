"""
Sinusoidal / Fourier-basis mapping for parameter conflation.

Each virtual parameter is computed as a weighted combination of K actual
parameters, where the mixing weights depend on the virtual param's index
through sinusoidal functions.

For virtual param j, we:
  1. Use hash functions to select K actual param indices: s_1(j), ..., s_K(j)
  2. Compute a mixing weight for each via sin with per-term freq/phase
  3. Return: sum_k( actual[s_k(j)] * sin(omega_k * j + phi_k) ) / sqrt(K)

Key difference from v1: each virtual param selects its own set of actual
params (via hashing), rather than all virtual params sharing the same K indices.
This ensures decorrelation between virtual params.

Properties:
    - O(K) per virtual param
    - Decorrelated: different virtual params use different actual param subsets
    - Differentiable: gradients w.r.t. actual_params are bounded
    - No extra learned parameters
"""

import torch
import math
from typing import Tuple, Dict
from ..core import VirtualParameterMap


class SinusoidalMap(VirtualParameterMap):
    """
    Fourier-basis parameter conflation.

    Caches hash indices, frequencies, phases, and basis values per (slot_id, shape)
    so they are only generated once.

    Args:
        num_actual: number of actual (stored) parameters
        num_terms: how many actual params contribute to each virtual param (K)
        init_std: initialization std for actual params
    """

    def __init__(self, num_actual: int, num_terms: int = 8, init_std: float = 0.02):
        super().__init__(num_actual, init_std)
        self.num_terms = min(num_terms, num_actual)
        # Cache: (slot_id, n_virtual) -> (indices, basis)
        self._cache: Dict = {}

    def _get_cached(self, n_virtual: int, slot_id: int, device: torch.device):
        key = (slot_id, n_virtual)
        if key not in self._cache:
            K = self.num_terms
            M = self.num_actual

            gen = torch.Generator()
            gen.manual_seed(slot_id * 1000003 + 42)

            indices = torch.randint(0, M, (K, n_virtual), generator=gen)

            omegas = torch.rand(K, generator=gen)
            omegas = 0.5 + omegas * 10.0
            phis = torch.rand(K, generator=gen) * (2.0 * math.pi)

            positions = torch.arange(n_virtual, dtype=torch.float32)
            angles = omegas.unsqueeze(1) * positions.unsqueeze(0) + phis.unsqueeze(1)
            basis = torch.sin(angles)  # [K, n_virtual]

            self._cache[key] = (indices, basis)

        indices, basis = self._cache[key]
        return indices.to(device), basis.to(device)

    def _compute_virtual(self, shape: Tuple[int, ...], slot_id: int) -> torch.Tensor:
        n_virtual = 1
        for s in shape:
            n_virtual *= s

        K = self.num_terms
        indices, basis = self._get_cached(n_virtual, slot_id, self.actual_params.device)

        # Gather actual params: [K, n_virtual]
        selected = self.actual_params[indices]

        # Weighted sum across K terms: [n_virtual]
        virtual_flat = (selected * basis).sum(dim=0) / math.sqrt(K)

        return virtual_flat.reshape(shape)

    def extra_repr(self) -> str:
        return f"num_actual={self.num_actual}, num_terms={self.num_terms}"
