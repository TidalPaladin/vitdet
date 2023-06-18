#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import Tensor

from vitdet import ViTDet


class TestViTDet:
    def test_forward(self):
        H, W = 256, 256
        model = ViTDet(1, 32, 2, 6, (H, W), (16, 16), (4, 4), 3)
        x = torch.randn(1, 1, H, W)
        y = model(x)
        assert isinstance(y, Tensor)
        assert y.shape == (1, 16 * 16, 32)
