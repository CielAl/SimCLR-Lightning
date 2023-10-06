from torchvision import transforms
from torch import nn
from simclr_lightning.dataset.data_class import ModelInput
import torch
from typing import Callable


def default_transform_factory(target_size, brightness: float = 0.8, contrast: float = 0.8,
                              saturation: float = 0.8, hue: float = 0.2) -> transforms.Compose:

    color_jitter = transforms.ColorJitter(brightness, contrast,
                                          saturation, hue)
    data_transforms = transforms.Compose([
                                          transforms.RandomResizedCrop(size=target_size, antialias=True),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          transforms.RandomApply(
                                              [transforms.GaussianBlur(kernel_size=int(0.1 * target_size))],
                                              p=0.5),
                                          ])
    return data_transforms


class AugmentationView(nn.Module):
    """Inspired from https://github.com/sthalles/SimCLR

    """
    _n_views: int
    _base_transform: Callable

    @property
    def base_transform(self):
        return self._base_transform

    @property
    def n_views(self):
        return self._n_views

    @n_views.setter
    def n_views(self, x):
        assert isinstance(x, int)
        self._n_views = x

    def __init__(self, base_transform: Callable, n_views: int):
        super().__init__()
        self._base_transform = base_transform
        self.n_views = n_views

    @staticmethod
    def _image_from_input(x: ModelInput | torch.Tensor):
        assert isinstance(x, (dict, torch.Tensor))

    def forward(self, x: ModelInput | torch.Tensor) -> ModelInput | torch.Tensor:

        if isinstance(x, dict):
            img_input = x['data']
        elif isinstance(x, torch.Tensor):
            img_input = x
        else:
            raise TypeError(f"{type(x)}")
        img_views = [self._base_transform(img_input) for _ in range(self.n_views)]
        output = torch.cat(img_views, dim=0)
        output = output.contiguous()

        if isinstance(x, dict):
            x['data'] = output
            output = x
        return output

    @classmethod
    def build(cls, base_transform: Callable, n_views: int):
        return cls(base_transform, n_views=n_views)

    @classmethod
    def build_from_default(cls, target_size,
                           brightness: float = 0.8, contrast: float = 0.8,
                           saturation: float = 0.8, hue: float = 0.2,
                           n_views: int = 2):
        augment = default_transform_factory(target_size, brightness, contrast, saturation, hue)
        return cls(base_transform=augment, n_views=n_views)
