#!/usr/bin/env python
# -*- coding: utf-8 -*-
from functools import partial
from typing import Any, Dict, Optional, Tuple, cast

import torch
import torch.nn as nn
from ssl_tasks.contrastive.task import ContrastiveEmbedding as ContrastiveEmbeddingBase
from ssl_tasks.tokens import TokenMask
from torch import Tensor

from vitdet import MODEL_REGISTRY

from .helpers import mask_fn


class ContrastiveEmbedding(ContrastiveEmbeddingBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        torch.set_float32_matmul_precision("medium")
        dim = cast(Any, self.backbone).dim
        self.query = nn.Parameter(torch.randn(1, 1, dim))

    def prepare_backbone(self, name: str) -> nn.Module:
        return MODEL_REGISTRY.get(name).instantiate_with_metadata().fn

    @property
    def img_size(self) -> Tuple[int, int]:
        return cast(Any, self.backbone).img_size

    def create_head(self) -> nn.Module:
        r"""Creates the MAE head for the model"""
        dim = cast(Any, self.backbone).dim
        return nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.LayerNorm(dim, elementwise_affine=False),
        )

    def create_token_mask(self, batch_size: int, device: torch.device = torch.device("cpu")) -> TokenMask:
        r"""Creates the MAE head for the model"""
        return TokenMask.create(self.img_size, cast(Any, self.backbone).patch_size, batch_size, device=device)

    def forward(self, x: Tensor, mask: Optional[TokenMask] = None) -> Dict[str, Tensor]:
        mask_hook = (
            self.backbone.register_mask_hook(partial(mask_fn, mask=mask), prepend=True) if mask is not None else None
        )
        N = x.shape[0]
        query = self.query.expand(N, -1, -1)
        _, cls = self.backbone(x, query)
        cls = self.embed_head(cls).view(N, -1)

        if mask_hook is not None:
            mask_hook.remove()

        return {"embed": cls}
