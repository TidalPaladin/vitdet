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

    def prepare_backbone(self, name: str) -> nn.Module:
        return MODEL_REGISTRY.get(name).instantiate_with_metadata().fn

    @property
    def img_size(self) -> Tuple[int, int]:
        return cast(Any, self.backbone).img_size

    def create_head(self) -> nn.Module:
        r"""Creates the MAE head for the model"""
        dim = cast(Any, self.backbone).dim
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim, elementwise_affine=False),
        )

    def create_token_mask(self, batch_size: int, device: torch.device = torch.device("cpu")) -> TokenMask:
        r"""Creates the MAE head for the model"""
        return TokenMask.create(self.img_size, cast(Any, self.backbone).patch_size, batch_size, device=device)

    def forward(self, x: Tensor, mask: Optional[TokenMask] = None) -> Dict[str, Tensor]:
        mask_hook = (
            self.backbone.register_mask_hook(partial(mask_fn, mask=mask), prepend=True) if mask is not None else None
        )
        x = self.backbone(x)
        x = x.mean(dim=1)
        x = self.embed_head(x)

        if mask_hook is not None:
            mask_hook.remove()

        return {"embed": x}
