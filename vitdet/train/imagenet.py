#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from typing import Any, Callable, Dict

import torch
import torchvision.transforms.v2 as T
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet as TVImageNet


class ImageNet(TVImageNet):
    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        img, label = super().__getitem__(index)
        return {
            "img": img,
            "label": label,
        }


class ImagenetDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        image_size: int = 224,
        num_workers: int = 0,
        batch_size: int = 32,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: path to the imagenet dataset file
            image_size: final image size
            num_workers: how many data workers
            batch_size: batch_size
            shuffle: If true shuffles the data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__(*args, **kwargs)

        self.image_size = image_size
        self.dims = (3, self.image_size, self.image_size)
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    @property
    def num_classes(self) -> int:
        return 1000

    def _verify_splits(self, data_dir: str, split: str) -> None:
        dirs = os.listdir(data_dir)

        if split not in dirs:
            raise FileNotFoundError(
                f"a {split} Imagenet split was not found in {data_dir},"
                f" make sure the folder contains a subfolder named {split}"
            )

    def prepare_data(self) -> None:
        """This method already assumes you have imagenet2012 downloaded. It validates the data using the meta.bin.

        .. warning:: Please download imagenet on your own first.
        """
        self._verify_splits(self.data_dir, "train")
        self._verify_splits(self.data_dir, "val")

        for split in ["train", "val"]:
            files = os.listdir(os.path.join(self.data_dir, split))
            assert "meta.bin" in files

    def train_dataloader(self) -> DataLoader:
        """Uses the train split of imagenet2012 and puts away a portion of it for the validation split."""
        transforms = self.train_transform()

        dataset = ImageNet(
            self.data_dir,
            split="train",
            transform=transforms,
        )
        loader: DataLoader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    # def val_dataloader(self) -> DataLoader:
    #    """Uses the part of the train split of imagenet2012  that was not used for training via
    #    `num_imgs_per_val_class`

    #    Args:
    #        batch_size: the batch size
    #        transforms: the transforms
    #    """
    #    transforms = self.val_transform()

    #    dataset = ImageNet(
    #        self.data_dir,
    #        split="val",
    #        transform=transforms,
    #    )
    #    loader: DataLoader = DataLoader(
    #        dataset,
    #        batch_size=self.batch_size,
    #        shuffle=False,
    #        num_workers=self.num_workers,
    #        drop_last=self.drop_last,
    #        pin_memory=self.pin_memory,
    #    )
    #    return loader

    # def test_dataloader(self) -> DataLoader:
    #    """Uses the validation split of imagenet2012 for testing."""
    #    transforms = self.val_transform() if self.test_transforms is None else self.test_transforms

    #    dataset = ImageNet(
    #        self.data_dir, split="test", transform=transforms
    #    )
    #    loader: DataLoader = DataLoader(
    #        dataset,
    #        batch_size=self.batch_size,
    #        shuffle=False,
    #        num_workers=self.num_workers,
    #        drop_last=self.drop_last,
    #        pin_memory=self.pin_memory,
    #    )
    #    return loader

    def train_transform(self) -> Callable:
        return T.Compose(
            [
                T.RandomResizedCrop(self.image_size),
                T.RandomHorizontalFlip(),
                T.ToImageTensor(),
                T.ConvertImageDtype(torch.float32),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def val_transform(self) -> Callable:
        return T.Compose(
            [
                T.Resize(self.image_size + 32),
                T.CenterCrop(self.image_size),
                T.ToImageTensor(),
                T.ConvertImageDtype(torch.float32),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
