#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib.metadata

from .vitdet import ViTDet


__version__ = importlib.metadata.version("vitdet")
__all__ = ["ViTDet"]
