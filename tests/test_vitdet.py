#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from vitdet import MODEL_REGISTRY, ViTDet


class TestViTDet:
    def test_forward(self):
        H, W = 256, 256
        model = ViTDet(1, 32, 2, 6, (H, W), (16, 16), (4, 4), 3)
        x = torch.randn(1, 1, H, W)
        y = model(x)
        assert isinstance(y, Tensor)
        assert y.shape == (1, 16 * 16, 32)

    @pytest.mark.parametrize("key", MODEL_REGISTRY.available_keys())
    def test_instantiate_from_registry(self, key):
        model = MODEL_REGISTRY.get(key).instantiate_with_metadata().fn
        assert isinstance(model, ViTDet)
        x = torch.rand(1, model.in_channels, *model.img_size)
        output = model(x)
        assert isinstance(output, Tensor)

    def test_unpatch(self):
        H, W = 256, 256
        Hp, Wp = 16, 16
        D = 32

        # Forward
        model = ViTDet(1, D, 2, 6, (H, W), (Hp, Wp), (4, 4), 3)
        x = torch.randn(1, 1, H, W)
        y = model(x)

        # To pixels
        mae = nn.Linear(D, Hp * Wp)
        x2 = model.unpatch(mae(y))
        assert isinstance(x2, Tensor)
        assert x2.shape == x.shape
