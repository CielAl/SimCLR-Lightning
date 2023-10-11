"""
Regulate the form of the dataset
"""
from typing import TypedDict, Union, List
import torch
import numpy as np
from PIL.Image import Image

TYPE_IMG_ARRAY = Union[np.ndarray, torch.Tensor, Image]


class ModelInput(TypedDict):
    """Format the input of any nn.Modules

    data: any data to feed into the model -- must be number, numpy.ndarray, torch.Tensor, or PIL.Image.Image
    original: default 0. A field to retain the original copy of data when necessary if a transformation is performed.
    filename: the uri for the data.
    meta: Any extra numeric data (e.g., coords) that can be handled by default collate_fn of DataLoader
    ground_truth: labels or any target data. must be number, numpy.ndarray, torch.Tensor.
    """
    data: Union[float, int, TYPE_IMG_ARRAY, List[TYPE_IMG_ARRAY]]
    original: Union[float, int, np.ndarray, torch.Tensor, Image]
    filename: str
    meta:  Union[float, int, TYPE_IMG_ARRAY, List[TYPE_IMG_ARRAY]]
    ground_truth: Union[int, float, np.ndarray, torch.Tensor]


class ModelOutput(TypedDict):
    """Format the output of nn.Modules. 'loss' is mandatory for compatibility of LightningModule

    Loss: output of loss function if available. Mandatory for Lightning APIs.
    logits: output layer responses.
    ground_truth: corresponding true value.
    meta: any meta data or extra information (e.g., coordinates of images)
    filename: the uri of the data point if necessary.
    """
    loss: torch.Tensor
    logits: torch.Tensor
    ground_truth: torch.Tensor
    meta:  Union[float, int, TYPE_IMG_ARRAY, List[TYPE_IMG_ARRAY]]
    filename: Union[str, List[str]]

