from typing import Dict
from torch import nn
from simclr_lightning.models.contrast_learning.base import AbstractBaseModel


class ClassificationHead(nn.Module):
    """Wrapper of classification head.
    """
    __num_classes: int
    __hidden_dim: int
    __classification_head: nn.Module

    def __init__(self, classification_head: nn.Module, num_classes: int, hidden_dim: int):
        super().__init__()
        self.__classification_head = classification_head
        self.__num_classes = num_classes
        self.__hidden_dim = hidden_dim

    @property
    def hidden_dim(self):
        return self.__hidden_dim

    @property
    def classification_head(self):
        return self.__classification_head

    @property
    def num_classes(self):
        return self.__num_classes

    def forward(self, x):
        return self.classification_head(x)

    @staticmethod
    def _default_classification(hidden_dim: int, num_classes: int, drop_rate: float) -> nn.Module:
        return nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    @classmethod
    def build_default(cls, hidden_dim: int, num_classes: int, drop_rate: float):
        class_head = cls._default_classification(hidden_dim, num_classes, drop_rate)
        return cls(classification_head=class_head, num_classes=num_classes, hidden_dim=hidden_dim)


class BaseFineTune(nn.Module):
    _simclr_base: AbstractBaseModel
    classification_head: ClassificationHead

    @property
    def simclr_base(self) -> AbstractBaseModel:
        return self._simclr_base

    @property
    def hidden_dim(self):
        return self.simclr_base.hidden_dim

    def remove_last_projection_layer(self):
        self._simclr_base.projection_head[-1] = nn.Identity()

    def __init__(self, simclr_base: AbstractBaseModel, classification_head: ClassificationHead):
        super().__init__()
        self._simclr_base = simclr_base
        self._simclr_base.return_embedding = False
        self.remove_last_projection_layer()
        self.classification_head = classification_head

    def load_base_model_state(self, state_dict: Dict, strict: bool):
        self._simclr_base.load_state_dict(state_dict=state_dict, strict=strict)

    def forward(self, x):
        feat = self.simclr_base(x)
        return self.classification_head(feat)

    def freeze_base_model(self, flag: bool):
        for params in self.simclr_base.parameters():
            params.requires_grad_(not flag)

    @classmethod
    def build_default(cls, simclr_base: AbstractBaseModel, num_classes: int, drop_rate: float):
        classification_head = ClassificationHead.build_default(simclr_base.hidden_dim, num_classes, drop_rate)
        return cls(simclr_base=simclr_base, classification_head=classification_head)
