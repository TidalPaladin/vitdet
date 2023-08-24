#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import torch.nn as nn
from einops import pack, rearrange, repeat, unpack
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from .helpers import DropPath
from .pos_enc import FourierLogspace
from .stem import PatchEmbed


# There is no einops.layers.torch.Repeat
class Repeat(nn.Module):
    def __init__(self, pattern: str, **kwargs):
        super().__init__()
        self.pattern = pattern
        self.kwargs = kwargs

    def forward(self, x: Tensor) -> Tensor:
        return repeat(x, self.pattern, **self.kwargs)


class StridedConvMixer(nn.Module):
    def __init__(
        self,
        dim: int,
        tokenized_size: Tuple[int, int],
        kernel_size: Union[int, Tuple[int, int]] = 3,
    ):
        super().__init__()
        self.dim = dim
        H, W = tokenized_size
        self.img_size = tokenized_size
        self.to_img = Rearrange("n (h w) d -> n d h w", h=H, w=W, d=dim)
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        padding = tuple(k // 2 for k in kernel_size)
        self.conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
        self.to_tokens = Rearrange("n d h w -> n (h w) d", h=H, w=W, d=dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        y = self.to_img(x)
        y = self.conv(y)
        y = self.to_tokens(y)
        return x + self.norm(y)


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
        self.grid_size = grid_size
        self.window = Rearrange("n (h hw w ww) c -> (n h w) (hw ww) c", hw=Hw, ww=Ww, h=H, w=W)
        self.unwindow = Rearrange("(n h w) (hw ww) c -> n (h hw w ww) c", hw=Hw, ww=Ww, h=H, w=W)
        self.token_distribute = Repeat("n l c -> (n h w) l c", h=H, w=W)
        self.token_pool = nn.Sequential(
            Rearrange("(n h w) l c -> n (h w) l c", h=H, w=W),
            Reduce("n w l c -> n () l c", "mean"),
        )

    @property
    def num_windowable_tokens(self) -> int:
        H, W = self.grid_size
        return H * W

    def forward(self, x: Tensor) -> Tensor:
        # Unpack any extra tokens from the end
        B, N, D = x.shape
        if N != self.num_windowable_tokens:
            x, tokens = unpack(x, [[self.num_windowable_tokens], [1]], "b * d")
        else:
            tokens = None

        # Window the image
        x = self.window(x)

        # Pack the tokens with each window
        ps = None
        if tokens is not None:
            tokens = self.token_distribute(tokens)
            x, ps = pack([x, tokens], "b * d")

        # Run the transformer
        x = super().forward(x)

        # Unpack the tokens
        if tokens is not None:
            assert ps is not None
            x, tokens = unpack(x, ps, "b * d")
            tokens = self.token_pool(tokens)

        # Unwindow the grid
        x = self.unwindow(x)

        # Join grid with reduced tokens
        if tokens is not None:
            x, _ = pack([x, tokens], "b * d")

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
        drop_path_rate: float = 0.0,
        conv_mixer: bool = False,
    ):
        super().__init__()
        self.img_size = img_size

        # Patch embed and position encoding
        self.patch_embed = PatchEmbed(in_channels, dim, patch_size)
        num_bands = dim
        max_freq = max(self.tokenized_size)
        self.pos_enc = FourierLogspace(2, dim, num_bands, max_freq, dropout=dropout, zero_one_norm=False)
        self.stem_norm = nn.LayerNorm(dim)
        self.final_norm = nn.LayerNorm(dim)
        self.conv_mixer = conv_mixer

        # Transformer
        self.transformer = nn.Sequential()
        for i in range(depth):
            drop_path = DropPath(drop_path_rate, residual=True)

            # Add Conv mixer
            if conv_mixer:
                layer = StridedConvMixer(
                    dim,
                    self.tokenized_size,
                    kernel_size=7,
                )
                self.transformer.add_module(f"conv_{i}", layer)
                self.transformer.add_module(f"drop_path_conv_{i}", drop_path)

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
                norm_first=False,
            )
            self.transformer.add_module(f"transformer_{i}", layer)
            self.transformer.add_module(f"drop_path_transformer_{i}", drop_path)

    @property
    def in_channels(self) -> int:
        return self.patch_embed.in_channels

    @property
    def dim(self) -> int:
        return self.patch_embed.dim

    @property
    def patch_size(self) -> Tuple[int, int]:
        return self.patch_embed.patch_size

    @property
    def tokenized_size(self) -> Tuple[int, int]:
        H = self.img_size[0] // self.patch_size[0]
        W = self.img_size[1] // self.patch_size[1]
        return H, W

    def forward(self, x: Tensor, tokens: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        if tokens is not None and self.conv_mixer:
            raise NotImplementedError("Additional tokens not supported when using conv mixer")

        # Patch embed
        x = self.patch_embed(x)

        # Position encoding
        pos = self.pos_enc.from_grid(dims=self.tokenized_size, proto=x)
        x = self.stem_norm(x + pos)

        # Concat with tokens
        # NOTE: We always pack tokens at the end. This simplifies windowing
        ps = None
        if tokens is not None:
            x, ps = pack([x, tokens], "b * d")

        # Transformer
        x = self.transformer(x)
        x = self.final_norm(x)

        # Unpack tokens
        if tokens is not None:
            assert ps is not None
            x, tokens = unpack(x, ps, "b * d")

        return x, tokens

    def register_mask_hook(self, func: Callable, *args, **kwargs) -> RemovableHandle:
        r"""Register a token masking hook to be applied after the patch embedding step.

        Args:
            func: Callable token making hook with signature given in :func:`register_forward_hook`

        Returns:
            A handle that can be used to remove the added hook by calling ``handle.remove()``.
        """
        return self.patch_embed.register_forward_hook(func, *args, **kwargs)

    def unpatch(self, x: Tensor) -> Tensor:
        Hp, Wp = self.patch_size
        H, W = self.tokenized_size
        return rearrange(
            x,
            "n (h w) (hp wp c) -> n c (h hp) (w wp)",
            hp=Hp,
            wp=Wp,
            h=H,
            w=W,
        )
