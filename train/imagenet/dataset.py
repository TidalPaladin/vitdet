#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import Tensor
from torchvision.datasets import ImageNet as TVImageNet
from typing import Dict

class ImageNet(TVImageNet):

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        img, label = super().__getitem__(index)
        return {
            "img": img,
            "label": label,
        }
