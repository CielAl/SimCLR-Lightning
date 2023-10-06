from torch import nn
from abc import ABC, abstractmethod
from simclr_lightning.models.contrast_learning.transforms import AugmentationView
from typing import Tuple, Literal, Callable

SUPPORTED_RESNET = Literal['resnet18', 'resnet50', 'resnet34', 'resnet101', 'resnet152']


class BaseModelCore(nn.Module):
    augment_view: AugmentationView
    backbone: nn.Module
    projection_head: nn.Sequential
    _n_views: int
    _return_embedding: bool

    @property
    def n_views(self):
        return self._n_views

    @property
    def return_embedding(self):
        return self._return_embedding

    @return_embedding.setter
    def return_embedding(self, x):
        assert isinstance(x, bool)
        self._return_embedding = x

    @staticmethod
    def to_sequential(module: nn.Module):
        if isinstance(module, nn.Sequential):
            return module
        return nn.Sequential(module)

    def __init__(self,
                 augment_view: AugmentationView,
                 backbone: nn.Module,
                 projection: nn.Sequential,
                 return_embedding: bool):
        super().__init__()

        self.augment_view = augment_view
        self._n_views = self.augment_view.n_views
        self.backbone = backbone
        self.projection_head = BaseModelCore.to_sequential(projection)

        self.return_embedding = return_embedding

    def forward(self, x):
        transformed_input = self.augment_view(x)
        embedding_feat = self.backbone(transformed_input)
        if self._return_embedding:
            return embedding_feat
        return self.projection_head(embedding_feat)


class AbstractBaseModel(BaseModelCore, ABC):

    _hidden_dim: int

    @property
    def hidden_dim(self):
        return self._hidden_dim

    def __init__(self,
                 augment_view: AugmentationView,
                 model_name: str,
                 out_dim,
                 projection_bn: bool,
                 return_embedding: bool,
                 **backbone_args):
        backbone, projection_hidden_dim = self._get_backbone_model_config(model_name, **backbone_args)
        self._hidden_dim = projection_hidden_dim
        projection = self._sequential_projection(projection_hidden_dim, out_dim, projection_bn=projection_bn)
        super().__init__(augment_view, backbone, projection, return_embedding)

    @abstractmethod
    def _get_backbone_model_config(self, model_name: str, **backbone_args) -> Tuple[nn.Module, int]:
        """get the backbone and size of its embedding feature dimension (e.g., 512 for resnet18)

        Args:
            model_name: name of the arch. Can be used as
            **backbone_args: detailed keyword argument for model construction

        Returns:
            backbone model and the number of its output dim
        """

        raise NotImplementedError

    @staticmethod
    def _sequential_projection(projection_hidden_dim: int, out_dim: int, projection_bn: bool) -> nn.Sequential:
        """Default implementation of projection head in SimCLR v1. Assume the final layer is a nn.Linear.

        Args:
            projection_hidden_dim:
            out_dim:
            projection_bn:

        Returns:

        """
        # dim_mlp x dim_mlp relu dim_mlp x out_dim
        final_feature = nn.Linear(projection_hidden_dim, out_dim)
        projection = nn.Sequential(nn.Linear(projection_hidden_dim, projection_hidden_dim),
                                   nn.BatchNorm1d(projection_hidden_dim) if projection_bn else nn.Identity(),
                                   nn.LeakyReLU(negative_slope=0.01),
                                   final_feature,
                                   )
        return projection

    @classmethod
    def build(cls,
              transforms: Callable,
              n_views: int,
              model_name: str,
              out_dim: int,
              projection_bn: bool = True,
              return_embedding: bool = False,
              **backbone_args):
        view_generator = AugmentationView(transforms, n_views=n_views)
        return cls(augment_view=view_generator,
                   model_name=model_name, out_dim=out_dim,
                   projection_bn=projection_bn, return_embedding=return_embedding, **backbone_args)
