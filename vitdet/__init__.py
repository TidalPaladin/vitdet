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
    global_attention_interval=3,
    drop_path_rate=0.1,
    conv_mixer=False,
)

MODEL_REGISTRY(
    ViTDet,
    name="vitdet_large_patch8_224",
    in_channels=3,
    dim=1024,
    img_size=(224, 224),
    patch_size=(8, 8),
    window_size=(4, 4),
    depth=24,
    num_heads=16,
    dropout=0.1,
    global_attention_interval=4,
    drop_path_rate=0.1,
    conv_mixer=False,
)

MODEL_REGISTRY(
    ViTDet,
    name="vitdet_huge_patch8_224",
    in_channels=3,
    dim=1280,
    img_size=(224, 224),
    patch_size=(8, 8),
    window_size=(4, 4),
    depth=32,
    num_heads=20,
    dropout=0.1,
    global_attention_interval=4,
    drop_path_rate=0.1,
    conv_mixer=False,
)

__version__ = importlib.metadata.version("vitdet")
__all__ = ["ViTDet"]
