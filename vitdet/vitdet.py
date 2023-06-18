#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial
from typing import List, Tuple, Union

import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor

from .pos_enc import FourierLogspace
from .stem import PatchEmbed


class WindowAttention(nn.TransformerEncoderLayer):
    def __init__(
        self,
        *args,
        grid_size: Tuple[int, int],
        window_size: Tuple[int, int],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        Hw, Ww = window_size
        Hi, Wi = grid_size

        if not Hi % Hw == 0:
            raise ValueError(f"Window height {Hw} does not divide grid height {Hi}")
        if not Wi % Ww == 0:
            raise ValueError(f"Window width {Ww} does not divide grid width {Wi}")

        H, W = Hi // Hw, Wi // Ww
        self.window = Rearrange("n (h hw w ww) c -> (n h w) (hw ww) c", hw=Hw, ww=Ww, h=H, w=W)
        self.unwindow = Rearrange("(n h w) (hw ww) c -> n (h hw w ww) c", hw=Hw, ww=Ww, h=H, w=W)

    def forward(self, x: Tensor) -> Tensor:
        x = self.window(x)
        x = super().forward(x)
        x = self.unwindow(x)
        return x


class ViTDet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dim: int,
        num_heads: int,
        depth: int,
        img_size: Tuple[int, int],
        patch_size: Tuple[int, int] = (16, 16),
        window_size: Tuple[int, int] = (7, 7),
        global_attention_interval: Union[int, List[int]] = 3,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU(),
    ):
        super().__init__()
        self.img_size = img_size

        # Patch embed and position encoding
        self.patch_embed = PatchEmbed(in_channels, dim, patch_size)
        num_bands = dim
        max_freq = max(self.tokenized_size)
        self.pos_enc = FourierLogspace(2, dim, num_bands, max_freq, dropout=dropout, zero_one_norm=False)
        self.norm = nn.LayerNorm(dim)

        # Transformer
        self.transformer = nn.Sequential()
        for i in range(depth):
            # Decide if we want to use global attention based on depth of current layer
            global_attn = (
                isinstance(global_attention_interval, int)
                and i % global_attention_interval == 0
                or isinstance(global_attention_interval, list)
                and i in global_attention_interval
            )
            LayerType = (
                partial(WindowAttention, grid_size=self.tokenized_size, window_size=window_size)
                if global_attn
                else nn.TransformerEncoderLayer
            )

            # Add transformer layer
            layer = LayerType(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=4 * dim,
                dropout=dropout,
                activation=activation,
                batch_first=True,
                norm_first=True,
            )
            self.transformer.add_module(f"layer{i}", layer)

    @property
    def patch_size(self) -> Tuple[int, int]:
        return self.patch_embed.patch_size

    @property
    def tokenized_size(self) -> Tuple[int, int]:
        H = self.img_size[0] // self.patch_size[0]
        W = self.img_size[1] // self.patch_size[1]
        return H, W

    def forward(self, x: Tensor) -> Tensor:
        # Patch embed
        x = self.patch_embed(x)

        # Position encoding
        pos = self.pos_enc.from_grid(dims=self.tokenized_size, proto=x)
        x = self.norm(x + pos)

        # Transformer
        x = self.transformer(x)
        return x
