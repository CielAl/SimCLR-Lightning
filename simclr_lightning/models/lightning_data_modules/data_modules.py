from typing import Callable, Optional, Union, List
import numpy as np
import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader

from simclr_lightning.dataset import TransformSet, AbstractDataset
from simclr_lightning.dataset.example import VisionDatasetWrapper
from simclr_lightning.models.lightning_data_modules.example_sampling import stratified_subset
from torchvision.transforms import ToTensor

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
    train_sample_ratio: float

    @property
    def num_classes(self):
        return self.__num_classes

    @staticmethod
    def sampled_dataset(dataset: AbstractDataset, label_array: Union[List, np.ndarray],
                        ratio: Optional[float] = None, seed: int = 0):
        assert ratio is None or isinstance(ratio, float) and not isinstance(ratio, int), \
            f"ratio must be either fractional in [0.0, or 1.0] or None. Got {ratio}"
        if ratio is None or ratio >= 1.0:
            return dataset
        return stratified_subset(dataset, label_array, ratio, seed)

    def __init__(self, root_dir: str, dataset_name: str,
                 batch_size: int, num_workers: int, persistent: bool, seed: int = 0,
                 num_classes: Optional[int] = None,
                 train_sample_ratio: Optional[float] = None):
        """

        Args:
            root_dir: root_dir to save the downloaded dataset
            dataset_name: name of dataset, see `simclr_lightning.dataset.example.VisionDatasetWrapper`
            batch_size: batch size for dataloaders
            num_workers: num_workers for dataloaders
            persistent: whether workers are persistent in dataloaders
            seed: random seed
            num_classes: num_classes. If None then `ExampleDataModule` will attempt to count the unique labels from
                dataset in the given `labels` attribute
            train_sample_ratio: The ratio to sample a subset from training set, while preserving label composition if
                possible. Can be used in fine-tuning. Must be a fractional number between [0, 1] to sample. If set None
                or larger than 1 then no effect (no sampling, use full training set).
        """

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
        self._train_sample_ratio = float(train_sample_ratio) if train_sample_ratio is not None else train_sample_ratio

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

        # if sample only a ratio
        self.train_dataset = ExampleDataModule.sampled_dataset(self.train_dataset,
                                                               train_dataset_origin.labels,
                                                               self._train_sample_ratio,
                                                               self.seed)

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
