from .data_class import ModelInput
from typing import Callable, Union
from .base import AbstractDataset
from copy import deepcopy


class TransformSet(AbstractDataset):
    """
    Wrapper dataset to add on-the-fly augmentation.
    """
    _dataset: AbstractDataset
    _transforms: Union[Callable, None]
    _copy_flag: bool
    _keep_original: bool

    def __len__(self):
        return len(self._dataset)

    def __init__(self, dataset: AbstractDataset, transforms: Callable,
                 keep_original: bool = False,
                 copy_flag: bool = False):
        """

        Args:
            dataset: associated dataset.
            transforms: augmentation functions. If no transforms then set it to None.
            keep_original: Whether to keep the copy of data into "original" field of ModelInput
            copy_flag: Whether to perform deepcopy of the data. Otherwise just keep the reference, e.g., if the
                transformation itself is not in-place and create the copy then it's not necessary to copy again.
        """
        super().__init__()
        self._dataset = dataset
        self._transforms = transforms
        self._copy_flag = copy_flag
        self._keep_original = keep_original

    def fetch(self, index) -> ModelInput:
        data: ModelInput = self._dataset[index]
        if self._keep_original:
            data['original'] = data['data'] if not self._copy_flag else deepcopy(data['data'])
        else:
            data['original'] = AbstractDataset.DEFAULT_VALUE

        if self._transforms is not None:
            data['data'] = self._transforms(data['data'])
        return data

    @classmethod
    def build(cls, dataset: AbstractDataset, transforms: Callable,
              keep_original: bool = False,
              copy_flag: bool = False):
        return cls(dataset=dataset, transforms=transforms, keep_original=keep_original, copy_flag=copy_flag)
