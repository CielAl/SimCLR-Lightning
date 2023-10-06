from functools import partial
from typing import Tuple, get_args, Callable

from torch import nn
from torchvision.models import resnet

from simclr_lightning.models.contrast_learning.base import AbstractBaseModel, SUPPORTED_RESNET


class ResNetBaseModel(AbstractBaseModel):

    def _get_backbone_model_config(self, model_name: SUPPORTED_RESNET, **backbone_args) -> Tuple[nn.Module, int]:
        assert model_name in get_args(SUPPORTED_RESNET)

        constructor = getattr(resnet, model_name)
        assert isinstance(constructor, Callable)
        model_func = partial(constructor, num_classes=2)
        base_model = model_func()
        hidden_dim = base_model.fc.in_features
        base_model.fc = nn.Identity()
        return base_model, hidden_dim
