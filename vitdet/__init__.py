#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib.metadata

from registry import Registry

from .vitdet import ViTDet


MODEL_REGISTRY = Registry("vitdet-models")

MODEL_REGISTRY(
    ViTDet,
    name="vitdet_base_patch8_224",
    in_channels=3,
    dim=768,
    img_size=(224, 224),
    patch_size=(8, 8),
    window_size=(4, 4),
    depth=12,
    num_heads=12,
    dropout=0.1,
)


MODEL_REGISTRY(
    ViTDet,
    name="vitdet_base_patch16_224",
    in_channels=3,
    dim=768,
    img_size=(224, 224),
    patch_size=(16, 16),
    window_size=(4, 4),
    depth=16,
    global_attention_interval=4,
    num_heads=12,
    dropout=0.1,
)


__version__ = importlib.metadata.version("vitdet")
__all__ = ["ViTDet"]
