"""
Demo dataset class. We assume the dataset here are all labeled, so it can be used in finetune stage.
For simplicity, we assume the dataset can access the label of fields directly through an attribute (e.g., `targets` in
MNIST).
"""
from torchvision import datasets as tv_data
from functools import partial
from .base import AbstractDataset
from .data_class import ModelInput
from typing import Dict, Callable

# assume the following example all have __getitem__ that can be unpacked as (image, label) for each index and
# accepts root and download as argument for __init__

_dataset_builder: Dict[str, Callable[..., tv_data.VisionDataset]] = {
    'cifar10': partial(tv_data.CIFAR10, download=True),
    'mnist': partial(tv_data.MNIST, download=True)
}

_dataset_label_field: Dict[str, str] = {
    'cifar10': 'targets',
    'mnist': 'targets'
}


class VisionDatasetWrapper(AbstractDataset):

    _vision_data: tv_data.VisionDataset
    _label_field: str

    @property
    def labels(self):
        return getattr(self._vision_data, self._label_field)

    def __init__(self, v_data: tv_data.VisionDataset, label_field: str):
        self._vision_data = v_data
        self._label_field = label_field

    def __len__(self):
        return len(self._vision_data)

    def fetch(self, idx) -> ModelInput:
        pil_img, label = self._vision_data[idx]
        return ModelInput(data=pil_img, original=0, filename=str(idx), meta=idx, ground_truth=label)

    @classmethod
    def build(cls, root: str, train: bool, dataset_name: str):
        vision_dataset = _dataset_builder[dataset_name](root=root, train=train)
        label_field = _dataset_label_field[dataset_name]
        return cls(v_data=vision_dataset, label_field=label_field)
