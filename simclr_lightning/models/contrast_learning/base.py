from torch import nn
from abc import abstractmethod
from simclr_lightning.models.contrast_learning.transforms import AugmentationView
from simclr_lightning.models.hooks import register_output_hook, HookSimple
from typing import Tuple, Literal, Callable, Optional
import torch
from .masking import feat_masking

SUPPORTED_RESNET = Literal['resnet18', 'resnet50',
                           'resnet34', 'resnet101', 'resnet152',
                           'densenet121', 'densenet161',
                           'densenet169', 'densenet201']


def set_module_hook_func(parent: nn.Module,
                         module: nn.Module, module_name: str, hook_name: str):
    # if (not hasattr(module, module_name)) or getattr(module, module_name) is not module:
    setattr(parent, module_name, module)
    hook = register_output_hook(module)
    setattr(parent, hook_name, hook)


class HookedModel(nn.Module):

    def set_module_hook(self, module, module_name, hook_name):
        set_module_hook_func(self, module, module_name, hook_name)

    def __init__(self):
        super().__init__()


class BaseDecoder(HookedModel):
    middle_blocks: nn.Module
    _mid_dec_hook: HookSimple

    def __init__(self):
        super().__init__()

    def update_mid_blocks(self, middle_blocks: nn.Module) -> None:
        # self.__augment_view = augment_view
        # self._aug_hook = register_output_hook(self.__augment_view)
        self.set_module_hook(middle_blocks, 'middle_blocks', '_mid_dec_hook')

    @property
    def mid_out(self):
        return self._mid_dec_hook.stored


class DefaultDecoder(BaseDecoder):

    def __init__(self):
        super().__init__()
        self.update_mid_blocks(nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.middle_blocks(x)


class BaseModelCore(HookedModel):
    augment_view: AugmentationView
    backbone: nn.Module
    flattener: nn.Module
    projection_head: nn.Sequential
    decoder: BaseDecoder
    dec_shortcut: nn.Module
    aux_classifier: nn.Module

    _n_views: int
    _hidden_dim: int

    _reconstruct: bool
    add_shortcut: bool
    return_recon: bool
    do_prediction: bool
    detach_aux_input: bool

    _aug_hook: HookSimple
    _enc_hook: HookSimple
    _proj_hook: HookSimple
    _dec_hook: HookSimple
    _flatten_hook: HookSimple
    _dec_shortcut_hook: HookSimple
    _aux_class_hook: HookSimple

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

    @classmethod
    def _default_identity(cls, module: Optional[nn.Module]):
        if module is None:
            return nn.Identity()
        assert isinstance(module, nn.Module)
        return module

    @classmethod
    def validate_decoder(cls, module: Optional[nn.Module]):
        if module is None:
            return DefaultDecoder()
        assert isinstance(module, BaseDecoder)
        return module

    def __init__(self,
                 augment_view: AugmentationView,
                 backbone: nn.Module,
                 flattener: nn.Module,
                 projection: nn.Sequential,
                 hidden_dim: int,
                 reconstruct: bool = False,
                 decoder: Optional[BaseDecoder] = None,
                 return_recon: bool = False,
                 dec_shortcut: Optional[nn.Module] = None,
                 add_shortcut: bool = False,
                 classifier: Optional[nn.Module] = None,
                 do_prediction: bool = False,
                 dynamic_scaling: bool = True,
                 detach_aux_input: bool = False,
                 ):
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

        decoder = self.__class__.validate_decoder(decoder)
        # decoder =nn.Identity() if decoder is None else decoder
        self.update_decoder(decoder)

        classifier = self.__class__._default_identity(classifier)
        self.update_classifier(classifier)

        dec_shortcut = self.__class__._default_identity(dec_shortcut)
        # dec_shortcut = nn.Identity() if dec_shortcut is None else dec_shortcut
        self.update_dec_shortcut(dec_shortcut)

        self._hidden_dim = hidden_dim
        self.return_recon = return_recon

        self.add_shortcut = add_shortcut
        self.do_prediction = do_prediction
        self.dynamic_scaling = dynamic_scaling

        # self._aug_hook = register_output_hook(self.augment_view)
        # self._enc_hook = register_output_hook(self.backbone)
        # self._proj_hook = register_output_hook(self.projection_head)
        # self._dec_hook = register_output_hook(self.decoder)
        scaling = torch.ones(self._hidden_dim, 1, 1)
        self.scaling = scaling if not self.dynamic_scaling else nn.Parameter(scaling)
        self.detach_aux_input = detach_aux_input

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

    def update_decoder(self, new_decoder: BaseDecoder):
        # self.__decoder = new_decoder
        # self._dec_hook = register_output_hook(self.__decoder)
        self.set_module_hook(new_decoder, 'decoder', '_dec_hook')

    def update_classifier(self, new_classifier: nn.Module):
        self.set_module_hook(new_classifier, 'aux_classifier', '_aux_class_hook')

    def update_dec_shortcut(self, new_dec_shortcut: nn.Module):
        self.set_module_hook(new_dec_shortcut, 'dec_shortcut', '_dec_shortcut_hook')

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
    def dec_shortcut_hook(self):
        return self._dec_shortcut_hook

    @property
    def aux_class_hook(self):
        return self._aux_class_hook

    @property
    def aux_class_out(self):
        return self._aux_class_hook.stored

    @property
    def aug_out(self):
        return self.aug_hook.stored

    @property
    def short_out(self):
        return self._dec_shortcut_hook.stored

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

    @property
    def mid_out(self):
        return self.decoder.mid_out

    def augmentation(self, x: torch.Tensor):
        return self.augment_view(x)

    def reconstruct_path(self, x: torch.Tensor, embedding_feat: torch.Tensor):
        if not self.reconstruct:
            # aux. stored in the hook
            return
        if self.add_shortcut:
            embedding_feat = self.dec_shortcut(embedding_feat, x)
        self.decoder(embedding_feat)

    def class_path(self, mask: Optional[torch.Tensor]):
        if not self.reconstruct:
            # use reconstruct path feature
            return
        if not self.do_prediction:
            return
        # todo? where does this happen
        feat = self.decoder.mid_out
        assert feat is not None
        if self.detach_aux_input:
            feat = feat.detach()
        feat_masked = feat_masking(feat, mask)
        self.aux_classifier(feat_masked)

    def inference(self, x: torch.Tensor, mask: Optional[torch.Tensor]):
        embedding_feat = self.backbone(x)
        flattened_feat = self.flattener(embedding_feat)
        self.reconstruct_path(x, embedding_feat)
        self.class_path(mask)
        return flattened_feat

    def output_prediction(self, feat: torch.Tensor):
        return self.projection_head(feat)

    def forward(self, x, augment: bool = True, mask: Optional[torch.Tensor] = None):
        transformed_input = self.augment_view(x) if augment else x
        # embedding_feat = self.backbone(transformed_input)
        # flattened_feat = self.flattener(embedding_feat)
        # if self.reconstruct:
        #     # aux. stored in the hook
        #     self.decoder(embedding_feat)
        flattened_feat = self.inference(transformed_input, mask)
        project_feat = self.projection_head(flattened_feat)
        if self.return_recon:
            assert self.reconstruct
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
                 decoder: Optional[BaseDecoder] = None,
                 return_recon: bool = False,
                 dec_shortcut: Optional[nn.Module] = None,
                 add_shortcut: bool = False,
                 classifier: Optional[nn.Module] = None,
                 do_prediction: bool = False,
                 dynamic_scaling: bool = True,
                 detach_aux_input: bool = False,
                 **backbone_args):
        backbone, flattener, projection_hidden_dim = self._get_backbone_model_config(model_name, **backbone_args)
        hidden_dim = projection_hidden_dim
        projection = self._sequential_projection(projection_hidden_dim, out_dim, projection_bn=projection_bn)
        super().__init__(augment_view, backbone, flattener,
                         projection, hidden_dim, reconstruct,
                         decoder, return_recon,
                         dec_shortcut=dec_shortcut,
                         add_shortcut=add_shortcut,
                         classifier=classifier,
                         do_prediction=do_prediction,
                         dynamic_scaling=dynamic_scaling,
                         detach_aux_input=detach_aux_input)

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
              decoder: Optional[BaseDecoder] = None,
              return_recon: bool = False,
              dec_shortcut: Optional[nn.Module] = None,
              add_shortcut: bool = False,
              classifier: Optional[nn.Module] = None,
              do_prediction: bool = False,
              dynamic_scaling: bool = True,
              detach_aux_input: bool = False,
              **backbone_args):
        """

        Args:
            transforms:
            n_views:
            model_name:
            out_dim:
            projection_bn:
            reconstruct:
            decoder:
            return_recon:
            dec_shortcut:
            add_shortcut:
            classifier:
            do_prediction:
            dynamic_scaling:
            detach_aux_input:
            **backbone_args:

        Returns:

        """
        view_generator = AugmentationView(transforms, n_views=n_views)
        return cls(augment_view=view_generator,
                   model_name=model_name, out_dim=out_dim,
                   projection_bn=projection_bn, reconstruct=reconstruct, decoder=decoder,
                   return_recon=return_recon,
                   dec_shortcut=dec_shortcut,
                   add_shortcut=add_shortcut,
                   classifier=classifier,
                   do_prediction=do_prediction,
                   dynamic_scaling=dynamic_scaling,
                   detach_aux_input=detach_aux_input,
                   **backbone_args)
