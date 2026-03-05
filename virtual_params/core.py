"""
Core abstraction for Parameter Conflation.

A VirtualParameterMap holds M actual parameters and can produce N >> M virtual
parameters on the fly via a deterministic, differentiable mapping function f.

The mapping function is a swappable strategy — subclasses implement get_virtual().

CONTRACT: subclasses should produce virtual params with approximately unit
variance when actual_params have unit variance. The layers handle Kaiming scaling.
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

    def __init__(self, num_actual: int, init_std: float = 1.0):
        super().__init__()
        if num_actual < 1:
            raise ValueError(f"num_actual must be >= 1, got {num_actual}")
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
            Should have approximately unit variance when actual_params has unit variance.
        """
        ...

    def get_virtual(self, shape: Tuple[int, ...], slot_id: int) -> torch.Tensor:
        """Public API: materialize virtual params for a given slot."""
        return self._compute_virtual(shape, slot_id)

    def extra_repr(self) -> str:
        return f"num_actual={self.num_actual}"
