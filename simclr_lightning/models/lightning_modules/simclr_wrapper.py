from typing import Tuple, Optional, Literal
import torch
import torchmetrics
from torchvision import transforms as tvtf
from simclr_lightning.models.contrast_learning.loss import ContrastLoss, ReConstLoss
from simclr_lightning.models.contrast_learning.transforms import AugmentationView

from simclr_lightning.models.lightning_modules.base import PHASE_STR, BaseLightningModule, feature_norm_penalty
from simclr_lightning.models.contrast_learning.base import AbstractBaseModel
from simclr_lightning.dataset.data_class import ModelInput, ModelOutput
from simclr_lightning.models.metrics import MetricDict


OPTIM_ADAM = Literal['adam']
# todo lars optimizer
OPTIM_LARS = Literal['lars']
OPTIM_ADAMW = Literal['adamw']
SUPPORTED_OPTIM = Literal[OPTIM_ADAM, OPTIM_ADAMW]


def _get_optim(name: SUPPORTED_OPTIM):
    match name:
        case 'adam':
            return torch.optim.Adam
        case 'adamw':
            return torch.optim.AdamW
        case _:
            raise NotImplementedError


class SimCLRLightning(BaseLightningModule):
    out_dim: int
    lr: float
    betas: Tuple[float, float]
    weight_decay: float
    return_embedding: bool

    batch_size: int
    temperature: float
    prog_bar: bool

    contrast_weight: float

    image_channels: int

    accuracy: MetricDict
    loss_avg: MetricDict
    class_avg: MetricDict
    reconst_avg: MetricDict
    psnr_meter: MetricDict
    enc_penalty_meter: MetricDict

    norm_penalty: bool
    norm_target: float
    norm_lambda: float

    @property
    def n_views(self):
        return self.model.n_views

    def __init__(self,
                 base_model: AbstractBaseModel,
                 lr: Optional[float] = 1e-3,
                 batch_size: Optional[int] = 64,
                 temperature: Optional[float] = 0.07,
                 contrast_weight: float = 1.0,
                 reconst_weight: float = 1.0,
                 max_t: Optional[int] = 90,  # length dl
                 betas=(0.5, 0.99),
                 weight_decay: float = 0,
                 prog_bar: bool = True,
                 next_line: bool = True,
                 extra_val_interval: Optional[int] = None,
                 recon_beta: float = 0.005,
                 image_channels: int = 3,
                 optim_name: SUPPORTED_OPTIM = 'adam',
                 random_erase_p: float = 0.0,
                 norm_penalty: bool = False,
                 norm_target: float = 1000,
                 norm_lambda: float = 1e-3,
                 debug: bool = False,
                 ):
        """Wrapper of LightningModule for SimCLR training.

        Args:
            base_model: The base model. The transformation is enclosed in an `AugmentationView` and prepended to the
                backbone of the base model. This allows the GPU-acceleration of transformation and flexibility as
                the `augment_view` can be replaced any time (e.g., to nn.Identity in prediction)
                as there are no learnable weights.
            lr: learning rate
            batch_size: batch size
            temperature:  temperature for Info NCE loss
            max_t: max number of steps for CosineAnnealingLR scheduler to restart.
            betas: betas for the optimizer (adams in the current implementation)
            weight_decay: weight decays for the optimizer (adams in the current implementation)
            prog_bar: whether to log results in progress bar
            next_line: whether to print a new line after each validation epoch. This enables the default tqdm progress
                bar to retain the results of previous epochs in previous lines.
            debug: whether enter debug mode for certain downstream callbacks
        """
        optim_func = _get_optim(optim_name)
        super(SimCLRLightning, self).__init__(batch_size, lr, max_t, prog_bar, next_line,
                                              optim_func=optim_func, debug=debug)

        # params
        self.temperature = temperature
        self.max_t = max_t
        self.betas = betas
        self.weight_decay = weight_decay

        # model
        self.model = base_model

        # contrastive loss
        self.contrast_loss = ContrastLoss.build(self.batch_size, self.n_views, self.temperature, contrast_weight)

        # cross entropy loss for classification of positive/negative pairs
        self.criterion = torch.nn.CrossEntropyLoss()
        self.reconst_loss = ReConstLoss.build(reconst_weight * bool(self.model.reconstruct), beta=recon_beta)

        # meters for loss and acc
        num_classes = self.n_views * (self.batch_size - 1) + 1
        top_k = min(5, num_classes)
        # self.accuracy = torchmetrics.classification.Accuracy(task="multiclass",
        #                                                      num_classes=2 * self.batch_size - 1, top_k=top_k)
        # calculate epoch-level mean cross-entropy loss
        # self.loss_avg = MetricDict.build() # torchmetrics.MeanMetric(nan_strategy='ignore')
        # self.class_avg = torchmetrics.MeanMetric(nan_strategy='ignore')
        # self.reconst_avg = torchmetrics.MeanMetric(nan_strategy='ignore')
        # self.psnr_meter = torchmetrics.image.PeakSignalNoiseRatio(data_range=(0., 1.))
        self.accuracy = MetricDict.build_all_modes(torchmetrics.classification.Accuracy,
                                                   task="multiclass",
                                                   num_classes=2 * self.batch_size - 1, top_k=top_k
                                                   )
        self.loss_avg = MetricDict.build_all_modes(torchmetrics.MeanMetric, nan_strategy='ignore')
        self.class_avg = MetricDict.build_all_modes(torchmetrics.MeanMetric, nan_strategy='ignore')
        self.reconst_avg = MetricDict.build_all_modes(torchmetrics.MeanMetric, nan_strategy='ignore')
        self.psnr_meter = MetricDict.build_all_modes(torchmetrics.image.PeakSignalNoiseRatio, data_range=(0., 1.))

        self.enc_penalty_meter = MetricDict.build_all_modes(torchmetrics.MeanMetric, nan_strategy='ignore')

        self.extra_val_interval = extra_val_interval
        self.image_channels = image_channels
        self.separate_optimizer = False

        self.random_erasing = tvtf.RandomErasing(p=random_erase_p, scale=(0.02, 0.1))

        self.norm_penalty = norm_penalty
        self.norm_target = norm_target
        self.norm_lambda = norm_lambda

    def forward(self, x, augment: bool, mask: Optional[torch.Tensor]):
        return self.model(x, augment=augment, mask=mask)

    def _step_enc_norm_penalty(self, phase_name: PHASE_STR):
        if not self.norm_penalty:
            return 0.
        if not isinstance(self.model, AbstractBaseModel):
            return 0.

        feat_map = self.model.enc_out
        if feat_map is None:
            return 0.
        norm_loss = feature_norm_penalty(feat_map, self.norm_target, self.norm_lambda)
        self.enc_penalty_meter.get_metric(phase_name).update(norm_loss)
        return norm_loss

    def compute_contrastive(self, augment_view: AugmentationView, images: torch.Tensor):
        # explicitly do augmentation outside, and use masking upon augmentation output as the model input
        augment_images = augment_view(images).clamp(0, 1)
        masked_images = self.random_erasing(augment_images)
        # recon_batch_size = batch['data'].shape[0]  # // self.n_views

        # already augmented beforehand
        logits = self(masked_images, augment=False, mask=None)  # self(images)
        # contrastive learning
        loss_contrast, (logits, labels) = self.contrast_loss(logits)
        return loss_contrast, (logits, labels)

    def _step_get_output(self, batch: ModelInput, phase_name: PHASE_STR):
        """Step helper shared by training and validation steps which computes the logits

        Args:
            batch: batch data in format of NetInput

        Returns:
            NetOutput containing loss, logits (final-layer output) and true labels.
        """
        # stacked view of original and augmented images
        images = batch['data'][:, :self.image_channels, ...]
        # explicitly do augmentation outside, and use masking upon augmentation output as the model input

        # augment_images = self.model.augment_view(images).clamp(0, 1)
        # masked_images = self.random_erasing(augment_images)
        # # recon_batch_size = batch['data'].shape[0]  # // self.n_views
        #
        # # already augmented beforehand
        # logits = self(masked_images, augment=False, mask=None)  # self(images)
        # # contrastive learning
        # loss_contrast, (logits, labels) = self.contrast_loss(logits)
        loss_contrast, (logits, labels) = self.compute_contrastive(self.model.augment_view, images)
        is_valid_class_loss = isinstance(logits, torch.Tensor) and isinstance(labels, torch.Tensor)
        if is_valid_class_loss and self.contrast_loss.weight != 0:
            self.accuracy[phase_name].update(logits, labels)
            self.class_avg[phase_name].update(loss_contrast / self.contrast_loss.weight)
        # recon
        #   # [:, :recon_batch_size, ...]  # self.model.aug_out  #
        real_img = self.model.aug_out[:, :self.image_channels, ...]
        if self.reconst_loss.weight > 0:
            reconst_out = self.model.reconst_out[:, :self.image_channels, ...]
            loss_recon = self.reconst_loss(real_img, reconst_out)
            self.reconst_avg[phase_name].update(loss_recon / self.reconst_loss.weight)
            self.psnr_meter[phase_name].update(reconst_out, real_img)
        else:
            loss_recon = 0.
            reconst_out = torch.zeros_like(real_img, device=real_img.device)
        # sum loss
        loss_norm_reg = self._step_enc_norm_penalty(phase_name)

        loss = loss_contrast + loss_recon + loss_norm_reg
        # if torch.isnan(loss).any():
        #     breakpoint()

        self.loss_avg[phase_name].update(torch.Tensor(loss))
        filenames = batch['filename']

        return ModelOutput(loss=loss, logits=self.model.flat_out,
                           ground_truth=real_img, filename=filenames, meta=reconst_out)

    def _step(self, batch: ModelInput, phase_name: PHASE_STR, batch_idx, dataloader_idx: int = 0):
        """Step function helper shared by training and validation steps which computes the logits and log the loss.

        Args:
            batch: batch data in format of NetInput
            phase_name: the name of current phase, i.e., train or validation, for purpose of loss logging.

        Returns:
            NetOutput containing loss, logits (final-layer output) and true labels.
        """
        # stacked view of original and augmented images
        out = self._step_get_output(batch, phase_name)
        self.log_metrics(phase_name, dataloader_idx)
        return out

    def _reset_on_first_batch(self, batch_idx: int, phase_name: PHASE_STR):
        if batch_idx == 0:
            # breakpoint()
            self.reset_meter_phase(phase_name)

    def reset_meter_phase(self, phase_name: PHASE_STR):
        self.accuracy.reset_by_mode(phase_name)
        self.loss_avg.reset_by_mode(phase_name)
        self.class_avg.reset_by_mode(phase_name)
        self.reconst_avg.reset_by_mode(phase_name)
        self.psnr_meter.reset_by_mode(phase_name)
        self.enc_penalty_meter.reset_by_mode(phase_name)

    def reset_meter_all(self):
        self.accuracy.reset_all()
        self.loss_avg.reset_all()
        self.class_avg.reset_all()
        self.reconst_avg.reset_all()
        self.psnr_meter.reset_all()
        self.enc_penalty_meter.reset_all()

    def training_step(self, batch: ModelInput, batch_idx, dataloader_idx: int = 0):
        self._reset_on_first_batch(batch_idx, 'fit')
        out = self._step(batch, 'fit', batch_idx, dataloader_idx)
        # todo move
        # self.scheduler_step()
        return out

    def validation_step(self, batch: ModelInput, batch_idx, dataloader_idx: int = 0):
        val_phase: PHASE_STR = 'validate'
        self._reset_on_first_batch(batch_idx, val_phase)
        out = self._step(batch, val_phase, batch_idx=batch_idx, dataloader_idx=dataloader_idx)
        return out

    def test_step(self, batch: ModelInput, batch_idx: int, dataloader_idx: int = 0):
        self._reset_on_first_batch(batch_idx, 'test')
        return self._step(batch, 'test', batch_idx, dataloader_idx)

    def predict_step(self, batch: ModelInput, batch_idx: int, dataloader_idx: int = 0):
        """In prediction, labels may not be available. Thus, loss is filled 0s as placeholders.

        n_views should be set to 1 if the goal is just to embed the input into feature space.
        Tweak the `_return_embedding` flag in self.model (AbstractBaseModel) to decide whether to get the projection
        head output (_return_embedding=False) or the hidden feature before projection head (_return_embedding=True)

        Args:
            batch:
            batch_idx:
            dataloader_idx:

        Returns:
            Prediction output.
        """
        self._reset_on_first_batch(batch_idx, 'predict')
        images = batch['data']
        labels = batch['ground_truth']
        meta = batch['meta']

        filenames_list = batch['filename']
        # obtain the projection feature representation
        logits = self(images)
        # placeholder - no loss computation as label may not be available in prediction phase
        loss_to_pred = 0 * torch.ones_like(logits)
        out = ModelOutput(loss=loss_to_pred, logits=logits,
                          ground_truth=labels, filename=filenames_list, meta=meta)
        return out

    def log_metrics(self, phase_name: PHASE_STR, dataloader_idx: int = 0):
        # breakpoint()
        self.log(f"{phase_name}_acc", self.accuracy.get_metric(phase_name),
                 batch_size=self.batch_size,
                 prog_bar=self.prog_bar,
                 logger=True,
                 sync_dist=True, on_epoch=True,
                 on_step=False)
        self.log(f"{phase_name}_loss", self.loss_avg.get_metric(phase_name),
                 batch_size=self.batch_size,
                 prog_bar=self.prog_bar, logger=True, sync_dist=True, on_epoch=True,
                 on_step=False)
        self.log(f"{phase_name}_class", self.class_avg.get_metric(phase_name),
                 batch_size=self.batch_size,
                 prog_bar=self.prog_bar, logger=True, sync_dist=True, on_epoch=True,
                 on_step=False)
        self.log(f"{phase_name}_reconst", self.reconst_avg.get_metric(phase_name),
                 batch_size=self.batch_size,
                 prog_bar=self.prog_bar, logger=True, sync_dist=True, on_epoch=True,
                 on_step=False)
        self.log(f"{phase_name}_psnr", self.psnr_meter.get_metric(phase_name),
                 batch_size=self.batch_size,
                 prog_bar=self.prog_bar, logger=True, sync_dist=True, on_epoch=True,
                 on_step=False)

        if self.norm_penalty:
            self.log(f'{phase_name}_enc_penalty',
                     self.enc_penalty_meter.get_metric(phase_name),
                     on_epoch=True, on_step=False, batch_size=self.batch_size,
                     prog_bar=self.prog_bar, sync_dist=True, logger=True)

    def on_train_epoch_start(self) -> None:
        self.reset_meter_all()

    def on_validation_epoch_end(self) -> None:
        self.print_newln()

    def on_test_epoch_end(self) -> None:
        self.print_newln()
        self.reset_meter_phase('test')

    def configure_optimizers(self):
        if not self.separate_optimizer:
            return super().configure_optimizers()
        raise NotImplementedError("manual optimization is not implemented")
        # params_base = self.param_groups(self.model.backbone, self.weight_decay)
        # params_proj = self.param_groups(self.model.projection_head, self.weight_decay)
        # params_ssl = params_base + params_proj
        #
        # params_recon = self.param_groups(self.model.decoder, self.weight_decay)
        #
        # optimizer_ssl = self.optim_func(params_ssl, lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        # optimizer_recon = self.optim_func(params_recon, lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        #
        # scheduler_ssl = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ssl, T_max=self.max_t, eta_min=0,
        #                                                            last_epoch=-1)
        # scheduler_recon = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_recon, T_max=self.max_t, eta_min=0,
        #                                                              last_epoch=-1)
        # return [optimizer_ssl, optimizer_recon], [scheduler_ssl, scheduler_recon]
