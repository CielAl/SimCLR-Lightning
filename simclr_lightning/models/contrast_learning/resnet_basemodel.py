from functools import partial
from typing import Tuple, get_args, Callable

from torch import nn
from torchvision.models import resnet

from simclr_lightning.models.contrast_learning.base import AbstractBaseModel, SUPPORTED_RESNET


class ResNetBaseModel(AbstractBaseModel):

    def _get_backbone_model_config(self, model_name: SUPPORTED_RESNET,
                                   **backbone_args) -> Tuple[nn.Module, nn.Module, int]:
        assert model_name in get_args(SUPPORTED_RESNET)

        constructor = getattr(resnet, model_name)
        assert isinstance(constructor, Callable)
        model_func = partial(constructor, num_classes=2)
        base_model = model_func()
        hidden_dim = base_model.fc.in_features
        # doing the below will butcher the dimensionality due to the extra x = torch.flatten(x, 1) in resnet
        # base_model.avgpool = nn.Identity()
        # base_model.fc = nn.Identity()
        base_model = nn.Sequential(*list(base_model.children())[:-2])
        flattener = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
        )
        return base_model, flattener, hidden_dim
