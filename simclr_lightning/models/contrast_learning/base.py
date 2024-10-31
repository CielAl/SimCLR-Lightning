from torch import nn
from abc import abstractmethod
from simclr_lightning.models.contrast_learning.transforms import AugmentationView
from simclr_lightning.models.hooks import register_output_hook, HookSimple
from typing import Tuple, Literal, Callable, Optional
import torch

SUPPORTED_RESNET = Literal['resnet18', 'resnet50',
                           'resnet34', 'resnet101', 'resnet152',
                           'densenet121', 'densenet161', 'densenet169', 'densenet201']


class BaseModelCore(nn.Module):
    augment_view: AugmentationView
    backbone: nn.Module
    flattener: nn.Module
    projection_head: nn.Sequential
    decoder: nn.Module

    _n_views: int
    _hidden_dim: int

    _reconstruct: bool

    _aug_hook: HookSimple
    _enc_hook: HookSimple
    _proj_hook: HookSimple
    _dec_hook: HookSimple
    _flatten_hook: HookSimple

    @property
    def reconstruct(self) -> bool:
        return self._reconstruct

    @reconstruct.setter
    def reconstruct(self, new_val: bool):
        self._reconstruct = new_val

    @property
    def hidden_dim(self):
        return self._hidden_dim

    @property
    def n_views(self):
        return self._n_views

    @staticmethod
    def to_sequential(module: nn.Module):
        if isinstance(module, nn.Sequential):
            return module
        return nn.Sequential(module)

    def __init__(self,
                 augment_view: AugmentationView,
                 backbone: nn.Module,
                 flattener: nn.Module,
                 projection: nn.Sequential,
                 hidden_dim: int,
                 reconstruct: bool = False,
                 decoder: Optional[nn.Module] = None,
                 return_recon: bool = False,):
        super().__init__()

        # self.augment_view = augment_view
        self.update_augment_view(augment_view)
        self._n_views = self.augment_view.n_views

        # self.backbone = backbone
        self.update_backbone(backbone)

        # self.flattener = flattener
        self.update_flattener(flattener)
        # self.projection_head = BaseModelCore.to_sequential(projection)
        self.update_projection_head(BaseModelCore.to_sequential(projection))
        self._reconstruct = reconstruct

        decoder = nn.Identity() if decoder is None else decoder
        self.update_decoder(decoder)

        self._hidden_dim = hidden_dim
        self.return_recon = return_recon

        # self._aug_hook = register_output_hook(self.augment_view)
        # self._enc_hook = register_output_hook(self.backbone)
        # self._proj_hook = register_output_hook(self.projection_head)
        # self._dec_hook = register_output_hook(self.decoder)

    def set_module_hook(self, module, module_name, hook_name):
        setattr(self, module_name, module)
        hook = register_output_hook(module)
        setattr(self, hook_name, hook)

    def update_flattener(self, module: nn.Module):
        # self.__flattener = module
        # self._flatten_hook = register_output_hook(self.__flattener)
        self.set_module_hook(module, 'flattener', '_flatten_hook')

    def update_augment_view(self, augment_view: AugmentationView):
        # self.__augment_view = augment_view
        # self._aug_hook = register_output_hook(self.__augment_view)
        self.set_module_hook(augment_view, 'augment_view', '_aug_hook')

    def update_backbone(self, backbone: nn.Module):
        # self.__backbone = backbone
        # self._enc_hook = register_output_hook(self.__backbone)
        self.set_module_hook(backbone, 'backbone', '_enc_hook')

    def update_projection_head(self, projection: nn.Sequential):
        # self.__projection_head = projection
        # self._proj_hook = register_output_hook(self.__projection_head)
        self.set_module_hook(projection, 'projection_head', '_proj_hook')

    def update_decoder(self, new_decoder: nn.Module):
        # self.__decoder = new_decoder
        # self._dec_hook = register_output_hook(self.__decoder)
        self.set_module_hook(new_decoder, 'decoder', '_dec_hook')

    @property
    def flatten_hook(self):
        return self._flatten_hook

    @property
    def aug_hook(self):
        return self._aug_hook

    @property
    def enc_hook(self):
        return self._enc_hook

    @property
    def proj_hook(self):
        return self._proj_hook

    @property
    def dec_hook(self):
        return self._dec_hook

    @property
    def aug_out(self):
        return self.aug_hook.stored

    @property
    def enc_out(self):
        return self.enc_hook.stored

    @property
    def proj_out(self):
        return self.proj_hook.stored

    @property
    def reconst_out(self):
        return self.dec_hook.stored

    @property
    def flat_out(self):
        return self.flatten_hook.stored

    def augmentation(self, x: torch.Tensor):
        return self.augment_view(x)

    def inference(self, x: torch.Tensor):
        embedding_feat = self.backbone(x)
        flattened_feat = self.flattener(embedding_feat)
        if self.reconstruct:
            # aux. stored in the hook
            self.decoder(embedding_feat)
        return flattened_feat

    def output_prediction(self, feat: torch.Tensor):
        return self.projection_head(feat)

    def forward(self, x, augment: bool = True):
        transformed_input = self.augment_view(x) if augment else x
        # embedding_feat = self.backbone(transformed_input)
        # flattened_feat = self.flattener(embedding_feat)
        # if self.reconstruct:
        #     # aux. stored in the hook
        #     self.decoder(embedding_feat)
        flattened_feat = self.inference(transformed_input)
        project_feat = self.projection_head(flattened_feat)
        if self.return_recon:
            return self.reconst_out
        return project_feat

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


class AbstractBaseModel(BaseModelCore):

    @property
    def hidden_dim(self):
        return self._hidden_dim

    def __init__(self,
                 augment_view: AugmentationView,
                 model_name: str,
                 out_dim,
                 projection_bn: bool,
                 reconstruct: bool = False,
                 decoder: Optional[nn.Module] = None,
                 return_recon: bool = False,
                 **backbone_args):
        backbone, flattener, projection_hidden_dim = self._get_backbone_model_config(model_name, **backbone_args)
        hidden_dim = projection_hidden_dim
        projection = self._sequential_projection(projection_hidden_dim, out_dim, projection_bn=projection_bn)
        super().__init__(augment_view, backbone, flattener, projection, hidden_dim, reconstruct, decoder, return_recon)

    @abstractmethod
    def _get_backbone_model_config(self, model_name: str, **backbone_args) -> Tuple[nn.Module, nn.Module, int]:
        """get the backbone, flattener (e.g., global pooling)
        and size of its embedding feature dimension (e.g., 512 for resnet18)

        Args:
            model_name: name of the arch. Can be used as
            **backbone_args: detailed keyword argument for model construction

        Returns:
            backbone model and the number of its output dim
        """

        raise NotImplementedError

    @classmethod
    def build(cls,
              transforms: Callable,
              n_views: int,
              model_name: str,
              out_dim: int,
              projection_bn: bool = True,
              reconstruct: bool = False,
              decoder: Optional[nn.Module] = None,
              return_recon: bool = False,
              **backbone_args):
        view_generator = AugmentationView(transforms, n_views=n_views)
        return cls(augment_view=view_generator,
                   model_name=model_name, out_dim=out_dim,
                   projection_bn=projection_bn, reconstruct=reconstruct, decoder=decoder, return_recon=return_recon,
                   **backbone_args)
