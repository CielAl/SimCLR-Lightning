import pytorch_lightning as L
import torch

from simclr_lightning.dataset.transform_set import TransformSet
from typing import Callable, Optional
from simclr_lightning.dataset.base import AbstractDataset
from simclr_lightning.dataset.example import VisionDatasetWrapper
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets as tv_data

_default_transform = ToTensor()


class BaseDataModule(L.LightningDataModule):
    """
    """
    train_dataset: TransformSet
    val_dataset: TransformSet
    # generator: torch.Generator
    __seed: int
    _dataset_transforms: Callable
    __generator: torch.Generator

    _batch_size: int
    _num_workers: int
    _persistent: bool

    @property
    def batch_size(self):
        return self.batch_size

    @property
    def num_workers(self):
        return self._num_workers

    @property
    def persistent(self):
        return self._persistent

    @property
    def seed(self):
        return self.__seed

    @seed.setter
    def seed(self, x):
        self.__seed = x
        self.__generator = torch.Generator().manual_seed(x)

    @property
    def generator(self):
        return self.__generator

    def __init__(self, dataset_transforms: Callable, seed: int, batch_size, num_workers, persistent):
        super().__init__()
        self.__seed = seed
        self.__generator = torch.Generator().manual_seed(seed)
        self._dataset_transforms = dataset_transforms

        # for dataloaders
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._persistent = persistent


class ExampleDataModule(BaseDataModule):
    """An example of data module using the example torchvision.dataset.VisionDataset wrapped into AbstractDataset

    The Lightning's wrapper will handle the DistributedSampler if distributed data parallel is performed.
    Manually override the details for other manual dataset and more fine-grained controls and configuration.
    """
    _root_dir: str
    _dataset_name: str

    train_dataset: AbstractDataset
    val_dataset: AbstractDataset
    pred_dataset: AbstractDataset

    __num_classes: int

    @property
    def num_classes(self):
        return self.__num_classes

    def __init__(self, root_dir, dataset_name, batch_size, num_workers, persistent, seed=0,
                 num_classes: Optional[int] = None):
        # Actual transform functions in AugmentView are fused into the nn.Module
        # only leave ToTensor here which curates the input format
        super().__init__(dataset_transforms=_default_transform,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         persistent=persistent,
                         seed=seed)
        self._root_dir = root_dir
        self._dataset_name = dataset_name
        self.__num_classes = num_classes

    @staticmethod
    def _label_set(dataset: VisionDatasetWrapper):
        label_set = set(x for x in dataset.labels)
        return label_set

    def set_num_classes_from_dataset(self, train_dataset_origin: VisionDatasetWrapper,
                                     val_dataset_origin: VisionDatasetWrapper):
        if self.__num_classes is not None and isinstance(self.__num_classes, int):
            return
        train_label_set = ExampleDataModule._label_set(train_dataset_origin)
        val_label_set = ExampleDataModule._label_set(val_dataset_origin)
        num_classes: int = len(train_label_set.union(val_label_set))
        self.__num_classes = num_classes

    def setup(self, stage: str = "") -> None:
        train_dataset_origin = VisionDatasetWrapper.build(root=self._dataset_name,
                                                          train=True, dataset_name=self._dataset_name)
        self.train_dataset = TransformSet.build(dataset=train_dataset_origin,
                                                transforms=self._dataset_transforms)

        val_dataset_origin = VisionDatasetWrapper.build(root=self._dataset_name, train=False,
                                                        dataset_name=self._dataset_name)
        self.val_dataset = TransformSet.build(dataset=val_dataset_origin,
                                              transforms=self._dataset_transforms)

        pred_dataset_origin = VisionDatasetWrapper.build(root=self._dataset_name, train=False,
                                                         dataset_name=self._dataset_name)
        self.pred_dataset = TransformSet.build(dataset=pred_dataset_origin,
                                               transforms=self._dataset_transforms)

        self.set_num_classes_from_dataset(train_dataset_origin, val_dataset_origin)

    def _dataloader_helper(self, dataset, shuffle, drop_last):
        # the Lightning's wrapper will handle the DistributedSampler if distributed data parallel is performed.
        # override the details
        return DataLoader(dataset, batch_size=self._batch_size,
                          shuffle=shuffle,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          drop_last=drop_last,
                          generator=self.generator,
                          persistent_workers=self.persistent and self.num_workers > 0)

    def train_dataloader(self):
        return self._dataloader_helper(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self._dataloader_helper(self.val_dataset, shuffle=False, drop_last=True)

    def predict_dataloader(self):
        return self._dataloader_helper(self.pred_dataset, shuffle=False, drop_last=False)
