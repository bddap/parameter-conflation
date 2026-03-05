"""
Drop-in replacement layers that use virtual parameters.

These layers don't own their own weight tensors — they request virtual
parameters from a shared VirtualParameterMap on each forward pass.
After the forward pass, the virtual weights are discarded.

Each layer is assigned a unique slot_id. Slot IDs are spaced by 2
internally (weight_slot = slot_id, bias_slot = slot_id + 1), so callers
should space their slot_ids by at least 2 per layer to avoid collisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from .core import VirtualParameterMap


class VirtualLinear(nn.Module):
    """
    Linear layer backed by virtual parameters.

    Functionally equivalent to nn.Linear, but weights (and optionally bias)
    are computed on-the-fly from a VirtualParameterMap.

    Scaling: weight = virtual * sqrt(gain / fan_in)
      - gain=2.0 (default) for layers followed by ReLU (Kaiming He)
      - gain=1.0 for output layers (Xavier/Glorot)
    """

    def __init__(self, vpm: VirtualParameterMap, in_features: int, out_features: int,
                 bias: bool = True, slot_id: int = 0, gain: float = 2.0):
        super().__init__()
        self.vpm = vpm
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.weight_slot = slot_id
        self.bias_slot = slot_id + 1 if bias else None

        # sqrt(gain / fan_in): gain=2 for ReLU (He), gain=1 for linear output (Xavier)
        self.weight_scale = math.sqrt(gain / in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.vpm.get_virtual(
            (self.out_features, self.in_features), self.weight_slot
        ) * self.weight_scale

        bias = None
        if self.use_bias and self.bias_slot is not None:
            # Bias is typically init'd to zero or small; scale down virtual params
            bias = self.vpm.get_virtual((self.out_features,), self.bias_slot) * 0.01

        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.use_bias}, weight_slot={self.weight_slot}, "
                f"scale={self.weight_scale:.4f}")


class VirtualConv2d(nn.Module):
    """
    Conv2d layer backed by virtual parameters.

    Functionally equivalent to nn.Conv2d, but weights (and optionally bias)
    are computed on-the-fly from a VirtualParameterMap.

    Scaling: weight = virtual * sqrt(gain / fan_in)
      - gain=2.0 (default) for layers followed by ReLU (Kaiming He)
      - gain=1.0 for output layers (Xavier/Glorot)
    """

    def __init__(self, vpm: VirtualParameterMap, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, padding: int = 0,
                 bias: bool = True, slot_id: int = 0, gain: float = 2.0):
        super().__init__()
        self.vpm = vpm
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias
        self.weight_slot = slot_id
        self.bias_slot = slot_id + 1 if bias else None

        # sqrt(gain / fan_in): gain=2 for ReLU (He), gain=1 for linear output (Xavier)
        fan_in = in_channels * kernel_size * kernel_size
        self.weight_scale = math.sqrt(gain / fan_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.vpm.get_virtual(
            (self.out_channels, self.in_channels, self.kernel_size, self.kernel_size),
            self.weight_slot
        ) * self.weight_scale

        bias = None
        if self.use_bias and self.bias_slot is not None:
            bias = self.vpm.get_virtual((self.out_channels,), self.bias_slot) * 0.01

        return F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        return (f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding}, bias={self.use_bias}, "
                f"weight_slot={self.weight_slot}")
