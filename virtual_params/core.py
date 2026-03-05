"""
Core abstraction for Parameter Conflation.

A VirtualParameterMap holds M actual parameters and can produce N >> M virtual
parameters on the fly via a deterministic, differentiable mapping function f.

The mapping function is a swappable strategy — subclasses implement get_virtual().
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple


class VirtualParameterMap(ABC, nn.Module):
    """
    Base class for parameter conflation maps.

    Stores M actual parameters and provides an interface to materialize
    virtual parameter tensors of arbitrary shape on demand.

    Subclasses must implement:
        _compute_virtual(shape, slot_id) -> Tensor

    The slot_id is a unique identifier for each virtual parameter request,
    ensuring deterministic mapping (same slot_id always produces the same
    virtual params for the same actual params).
    """

    def __init__(self, num_actual: int, init_std: float = 0.02):
        super().__init__()
        self.num_actual = num_actual
        self.actual_params = nn.Parameter(torch.randn(num_actual) * init_std)

    @abstractmethod
    def _compute_virtual(self, shape: Tuple[int, ...], slot_id: int) -> torch.Tensor:
        """
        Compute a tensor of virtual parameters.

        Args:
            shape: desired output shape
            slot_id: deterministic identifier for this virtual param request
                     (different layers should use different slot_ids)

        Returns:
            Tensor of the requested shape, computed from self.actual_params.
            Must be differentiable w.r.t. self.actual_params.
        """
        ...

    def get_virtual(self, shape: Tuple[int, ...], slot_id: int) -> torch.Tensor:
        """Public API: materialize virtual params for a given slot."""
        return self._compute_virtual(shape, slot_id)

    # TODO: Consider returning int (or math.inf) instead of str — callers may
    # expect a numeric value.  Kept as-is to avoid breaking subclass overrides.
    def num_virtual_possible(self) -> str:
        """Human-readable description of virtual param capacity."""
        return "depends on mapping"

    def extra_repr(self) -> str:
        return f"num_actual={self.num_actual}"
