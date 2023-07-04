#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Any, Dict, Optional, Tuple, cast

import torch
import torch.nn as nn
from ssl_tasks.mae.task import MAE as MAEBase
from ssl_tasks.tokens import TokenMask
from torch import Tensor

from vitdet import MODEL_REGISTRY


class MAE(MAEBase):
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
        x = self.backbone(x, mask)
        x = x.mean(dim=1)
        x = self.mae_head(x)
        return {"mae": x}
