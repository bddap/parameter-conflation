"""
Generic arithmetic mapping for parameter conflation.

A flexible mapping class that takes a user-provided combine function
to define how actual parameters are combined into virtual parameters.
This replaces the need for separate classes per combination strategy.

Usage:
    import math

    # Identity (x=1)
    ArithMap(500, num_terms=1, combine_fn=lambda a: a[0])

    # Multiplicative (x=2)
    ArithMap(500, num_terms=2, combine_fn=lambda a: a[0] * a[1])

    # HashArith-style (x=3)
    ArithMap(500, num_terms=3,
             combine_fn=lambda a: (a[0]*a[1] + a[2]) / math.sqrt(2))

    # DeepHash-style (x=7)
    ArithMap(500, num_terms=7,
             combine_fn=lambda a: ((a[0]*a[1]+a[2])*(a[3]*a[4]+a[5])+a[6]) / math.sqrt(5))

The combine_fn receives a list of `num_terms` tensors, each of shape
[n_virtual], and must return a single tensor of shape [n_virtual].
Normalization (if desired) is the caller's responsibility — include it
in the combine_fn.

Properties:
    - O(num_terms) per virtual param for index generation
    - Combination complexity depends on combine_fn
    - Differentiable as long as combine_fn uses differentiable ops
    - No extra learned parameters beyond actual_params
"""

import torch
import math
from typing import Tuple, List, Callable
from ..core import VirtualParameterMap


class ArithMap(VirtualParameterMap):
    """
    Generic arithmetic parameter conflation.

    For each virtual param j, deterministically selects `num_terms` actual
    params via hash functions and combines them with `combine_fn`.

    Args:
        num_actual: number of actual (stored) parameters
        num_terms: how many actual params each virtual param draws from
        combine_fn: function (List[Tensor]) -> Tensor that combines the terms.
                    Each tensor in the list has shape [n_virtual].
                    Must return a tensor of shape [n_virtual].
                    Should include any desired normalization.
        init_std: initialization std for actual params (default 1.0)
    """

    def __init__(
        self,
        num_actual: int,
        num_terms: int,
        combine_fn: Callable[[List[torch.Tensor]], torch.Tensor],
        init_std: float = 1.0,
    ):
        super().__init__(num_actual, init_std=init_std)
        if num_terms < 1:
            raise ValueError(f"num_terms must be >= 1, got {num_terms}")
        self.num_terms = num_terms
        self.combine_fn = combine_fn

    def _make_indices(self, n_virtual: int, slot_id: int, device: torch.device):
        M = self.num_actual
        gen = torch.Generator()
        gen.manual_seed(slot_id * 1000003 + 17)
        return [
            torch.randint(0, M, (n_virtual,), generator=gen).to(device)
            for _ in range(self.num_terms)
        ]

    def _compute_virtual(self, shape: Tuple[int, ...], slot_id: int) -> torch.Tensor:
        n_virtual = math.prod(shape)
        indices = self._make_indices(n_virtual, slot_id, self.actual_params.device)
        terms = [self.actual_params[idx] for idx in indices]
        return self.combine_fn(terms).reshape(shape)

    def extra_repr(self) -> str:
        return f"num_actual={self.num_actual}, num_terms={self.num_terms}"
