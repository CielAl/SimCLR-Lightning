from sklearn.model_selection import StratifiedShuffleSplit
from simclr_lightning.dataset.example import VisionDatasetWrapper
from torch.utils.data.dataset import Dataset, Subset
import numpy as np
from typing import Union, List


def stratified_sampling_indices(label_array: Union[List, np.ndarray], size: float, seed: int) -> List[int]:
    """Get the indices of subset while preserve the label composition.

     See `sklearn.model_selection.StratifiedShuffleSplit`

    Args:
        label_array: List of labels of data points.
        size: int as number of samples. Float from 0 to 1 as sample ratio.
            See `sklearn.model_selection.StratifiedShuffleSplit`
        seed: Random state

    Returns:
        Indices of subset sampled.
    """
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=size, random_state=seed)
    return list(splitter.split(label_array, label_array))[0][0].tolist()


def stratified_subset(dataset: Dataset, label_array: Union[List, np.ndarray], size: float, seed: int):
    indices = stratified_sampling_indices(label_array, size=size, seed=seed)
    return Subset(dataset, indices)


def stratified_example_subset(dataset: VisionDatasetWrapper, size: float, seed: int):
    """A wrapper of stratified_subset for example VisionDatasetWrapper with defined label fields

    Args:
        dataset:
        size:
        seed:

    Returns:

    """
    return stratified_subset(dataset, dataset.labels, size, seed)
