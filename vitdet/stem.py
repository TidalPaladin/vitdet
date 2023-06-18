#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple

import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor


class PatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: Tuple[int, int],
    ):
        super().__init__()
        self.patch = nn.Conv2d(in_channels, out_channels, patch_size, patch_size)
        self.to_sequence = Rearrange("n c h w -> n (h w) c")

    @property
    def in_channels(self) -> int:
        return self.patch.in_channels

    @property
    def dim(self) -> int:
        return self.patch.out_channels

    @property
    def patch_size(self) -> Tuple[int, int]:
        patch_size = self.patch.kernel_size
        return tuple(patch_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch(x)
        x = self.to_sequence(x)
        return x
