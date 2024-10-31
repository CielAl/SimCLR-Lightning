from typing import Tuple, Optional, Literal
import torch
import torchmetrics
from torchvision import transforms as tvtf
from simclr_lightning.models.contrast_learning.loss import ContrastLoss, ReConstLoss

from simclr_lightning.models.lightning_modules.base import PHASE_STR, BaseLightningModule
from simclr_lightning.models.contrast_learning.base import AbstractBaseModel
from simclr_lightning.dataset.data_class import ModelInput, ModelOutput


OPTIM_ADAM = Literal['adam']
# todo lars optimizer
OPTIM_LARS = Literal['lars']
SUPPORTED_OPTIM = Literal[OPTIM_ADAM, OPTIM_LARS]

random_erasing = tvtf.RandomErasing(p=0.5, scale=(0.02, 0.1))


class SimCLRLightning(BaseLightningModule):
    out_dim: int
    lr: float
    betas: Tuple[float, float]
    weight_decay: float
    return_embedding: bool

    batch_size: int
    temperature: float
    prog_bar: bool

    WARM_UP_EPOCH: int = 10
    contrast_weight: float

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
                 recon_beta: float = 0.005
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
        """
        super(SimCLRLightning, self).__init__(batch_size, lr, max_t, prog_bar, next_line)

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
        self.accuracy = torchmetrics.classification.Accuracy(task="multiclass",
                                                             num_classes=2 * self.batch_size - 1, top_k=top_k)
        # calculate epoch-level mean cross-entropy loss
        self.loss_avg = torchmetrics.MeanMetric()
        self.class_avg = torchmetrics.MeanMetric()
        self.reconst_avg = torchmetrics.MeanMetric()
        self.extra_val_interval = extra_val_interval
        # assert contrast_weight >= 0
        # self.contrast_weight = contrast_weight
        # assert reconst_weight >= 0
        # self.reconst_weight = reconst_weight
        # misc

    def forward(self, x, augment: bool):
        return self.model(x, augment=augment)

    def _step_get_output(self, batch: ModelInput):
        """Step helper shared by training and validation steps which computes the logits

        Args:
            batch: batch data in format of NetInput

        Returns:
            NetOutput containing loss, logits (final-layer output) and true labels.
        """
        # stacked view of original and augmented images
        images = batch['data']
        # explicitly do augmentation outside, and use masking upon augmentation output as the model input
        augment_images = self.model.augment_view(images).clip(0, 1)

        masked_images = random_erasing(augment_images)

        # recon_batch_size = batch['data'].shape[0]  # // self.n_views

        # already augmented beforehand
        logits = self(masked_images, augment=False)  # self(images)
        # contrastive learning

        loss_contrast, (logits, labels) = self.contrast_loss(logits)

        is_valid_class_loss = isinstance(logits, torch.Tensor) and isinstance(labels, torch.Tensor)
        if is_valid_class_loss and self.contrast_loss.weight != 0:
            self.accuracy.update(logits, labels)
            self.class_avg.update(loss_contrast / self.contrast_loss.weight)

        # recon
        real_img = self.model.aug_out  # [:, :recon_batch_size, ...]  # self.model.aug_out  #
        reconst_out = self.model.reconst_out  # [:, :recon_batch_size, ...]
        loss_recon = self.reconst_loss(real_img, reconst_out)
        self.reconst_avg.update(loss_recon / self.reconst_loss.weight)

        # sum loss
        loss = loss_contrast + loss_recon
        self.loss_avg.update(loss)
        filenames = batch['filename']

        return ModelOutput(loss=loss, logits=self.model.flat_out,
                           ground_truth=real_img, filename=filenames, meta=reconst_out)

    def _step(self, batch: ModelInput, phase_name: PHASE_STR):
        """Step function helper shared by training and validation steps which computes the logits and log the loss.

        Args:
            batch: batch data in format of NetInput
            phase_name: the name of current phase, i.e., train or validation, for purpose of loss logging.

        Returns:
            NetOutput containing loss, logits (final-layer output) and true labels.
        """
        # stacked view of original and augmented images
        out = self._step_get_output(batch)
        # log the TorchMetric object statistics to the logger/progbar
        self.log_on_final_batch(phase_name)
        return out

    def training_step(self, batch: ModelInput, batch_idx):
        out = self._step(batch, 'fit')
        self.scheduler_step()
        return out

    def _extra_val_every_n_epochs(self, phase_name: PHASE_STR):
        n = self.extra_val_interval
        if self.current_epoch > self.WARM_UP_EPOCH and n is not None and self.current_epoch % n == 0:
            self.log_on_final_batch(phase_name)

    def validation_step(self, batch: ModelInput, batch_idx, dataloader_idx: int = 0):
        default_phase: PHASE_STR = 'validate'
        # if dataloader_idx == 0:
        #     out = self._step(batch, default_phase)
        # else:  # if multiple validation as extra measurement
        #     out = self._step_get_output(batch)
        #     self._extra_val_every_n_epochs(default_phase)
        out = self._step(batch, default_phase)
        return out

    def test_step(self, batch: ModelInput, batch_idx):
        return self._step_get_output(batch)

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

    def _log_on_final_batch_helper(self, phase_name: PHASE_STR, dataloader_idx: int = 0):
        self.log_meter(f"{phase_name}_acc", self.accuracy, logger=True, sync_dist=True)
        self.log_meter(f"{phase_name}_loss", self.loss_avg, logger=True, sync_dist=True)
        self.log_meter(f"{phase_name}_class", self.class_avg, logger=True, sync_dist=True)
        self.log_meter(f"{phase_name}_reconst", self.reconst_avg, logger=True, sync_dist=True)
        # import matplotlib.pyplot as plt
        # import numpy as np
        # def simple_minmax(arr: np.ndarray):
        #     arr_min = arr.min()
        #     arr_max = arr.max()
        #     return 255 * (arr - arr_min) / (arr_max - arr_min)
        # debug_recon = simple_minmax(self.model.reconst_out.detach().cpu()[0].permute(1, 2, 0).numpy().astype(np.float32))
        # debug_aug = simple_minmax(self.model.aug_out.detach().cpu()[0].permute(1, 2, 0).numpy().astype(np.float32))
        #
        # plt.imshow(debug_recon)
        # plt.title('debug_recon')
        # plt.show()
        # plt.imshow(debug_aug)
        # plt.title('debug_aug')
        # plt.show()

    def _reset_meters(self):
        self.accuracy.reset()
        self.loss_avg.reset()
        self.class_avg.reset()
        self.reconst_avg.reset()

    def on_train_epoch_end(self) -> None:
        self._reset_meters()

    def on_validation_epoch_end(self) -> None:
        self._reset_meters()
        self.print_newln()

    def on_test_epoch_end(self) -> None:
        self._log_on_final_batch_helper('test')
        self._reset_meters()
        self.print_newln()
