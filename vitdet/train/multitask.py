#!/usr/bin/env python
# -*- coding: utf-8 -*-

from deep_helpers.tasks import MultiTask as MultiTaskBase


class MultiTask(MultiTaskBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.share_attribute("backbone")
