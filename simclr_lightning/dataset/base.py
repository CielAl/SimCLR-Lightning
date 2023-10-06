"""
Base abstract class - Dataset that returns ModelInput (TypedDict) in __getitem__
Optionally have cache for unpickleable objects supported for multiprocessing by setting it in worker_init_fn
of dataloader (directly initiate it in single processing mode)
The cache is supposed to be a dict that can be looked up from a key to return the data.

"""
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from .data_class import ModelInput
from typing import Dict, Set

SET_DATA_KEYS: Set = set(ModelInput.__annotations__.keys())


class AbstractDataset(Dataset, ABC):

    DEFAULT_VALUE: int = 0

    @staticmethod
    def _validate_type(data: ModelInput) -> ModelInput:
        """
        Validate if the data is a ModelInput type by enforcing its keys. The data must contain all necessary keys
        defined in ModelInput. Its key set should be a superset of that of ModelInput.
        Args:
            data: data to examine.

        Returns:
            data (ModelInput)

        Raises:
            AssertionError
        """
        assert isinstance(data, Dict)
        key_set = set(data.keys())
        assert key_set.issuperset(SET_DATA_KEYS)
        return data

    @abstractmethod
    def __len__(self):
        return NotImplemented

    @abstractmethod
    def fetch(self, index) -> ModelInput:
        """
        Override to fetch data from index. It's invoked in __getitem__
        Args:
            index:

        Returns:

        """
        raise NotImplementedError

    def __getitem__(self, index) -> ModelInput:
        """
        Use self.fetch to obtain the data by index but always guarantee that the data is a ModelInput
        Args:
            index: index of data (e.g., corresponding index in a list)

        Returns:

        """
        data = self.fetch(index)
        return AbstractDataset._validate_type(data)

    @classmethod
    @abstractmethod
    def build(cls, *args, **kwargs):
        """
        A factory method to build the dataset.
        Args:
            *args:
            **kwargs:

        Returns:

        """
        raise NotImplementedError
