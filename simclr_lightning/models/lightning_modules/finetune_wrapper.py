import torch
from typing import Optional, Dict, Callable
import numpy as np
from torch import nn

from torchmetrics import MeanMetric
from torchmetrics.classification import AUROC, ConfusionMatrix

from simclr_lightning.models.lightning_modules.base import PHASE_STR, BaseLightningModule
from simclr_lightning.models.contrast_learning.finetune import BaseFineTune
from simclr_lightning.models.lightning_modules.simclr_wrapper import SimCLRLightning
from simclr_lightning.dataset.data_class import ModelInput, ModelOutput


class FinetuneLightning(BaseLightningModule):
    _transforms_dict: nn.ModuleDict | Dict[PHASE_STR, Callable]
    finetune_base: BaseFineTune
    max_t: int

    @property
    def transforms_dict(self):
        return self._transforms_dict

    @transforms_dict.setter
    def transforms_dict(self, x):
        self._transforms_dict = x

    def __init__(self,
                 transforms_dict: nn.ModuleDict | Dict[PHASE_STR, Callable],
                 finetune_base: BaseFineTune,
                 freeze_weight: bool = False,
                 max_t: int = 90,
                 betas=(0.5, 0.99),
                 weight_decay: float = 1e-4,
                 lr: Optional[float] = 1e-3,
                 batch_size: Optional[int] = 64,
                 prog_bar: bool = True,
                 next_line: bool = True):
        """A simple LightningModule wrapper to finetune the pre-trained SimCLR model with labels.

        Args:
            transforms_dict: A dict (dict/ModuleDict) of callables (or better, nn.Modules) for transformations of
                input, e.g., optional augmentation of image data.
            finetune_base: Base model to finetune
            freeze_weight: whether to freeze the layers before classification layers, e.g., any convolutional blocks).
            max_t: max number of steps for CosineAnnealingLR scheduler to restart.
            betas: betas for the optimizer (adams in the current implementation)
            weight_decay: weight decays for the optimizer (adams in the current implementation)
            lr: learning rate
            batch_size: batch size
            prog_bar: whether to log results in progress bar
            next_line: whether to print a new line after each validation epoch. This enables the default tqdm progress
                bar to retain the results of previous epochs in previous lines.
        """
        super().__init__(batch_size=batch_size, lr=lr, max_t=max_t, prog_bar=prog_bar, next_line=next_line)
        self.finetune_base = finetune_base
        self._transforms_dict = transforms_dict
        self.num_classes = finetune_base.classification_head.num_classes
        self.freeze_pretrained_model(freeze_weight)

        # metrics
        self.auc_metric = AUROC("multiclass", num_classes=self.num_classes)  # BinaryAUROC(thresholds=None)
        self.loss_avg = MeanMetric()
        self.conf_mat = ConfusionMatrix('multiclass', num_classes=self.num_classes, normalize="true")

        # loss
        self.loss_func = nn.CrossEntropyLoss()
        # hparams
        self.weight_decay = weight_decay
        self.betas = betas

    def forward(self, x, mode: PHASE_STR):
        transformed_feat = self._transforms_dict[mode](x)
        return self.finetune_base(transformed_feat)

    def _step(self, batch: ModelInput, phase_name: PHASE_STR):
        """Step function helper shared by training and validation steps which computes the logits and log the loss.

        Args:
            batch: batch data in format of NetInput
            phase_name: the name of current phase, i.e., train or validation, for purpose of loss logging.

        Returns:
            NetOutput containing loss, logits (final-layer output) and true labels.
        """
        img = batch['data']
        labels = batch['ground_truth'].long()
        filenames = batch['filename']

        logits = self(img, phase_name)
        loss = self.loss_func(logits, labels)

        # update meters
        self.loss_avg.update(loss)
        self.auc_metric.update(logits, labels)
        self.conf_mat.update(logits, labels)
        out = ModelOutput(loss=loss, logits=logits, ground_truth=labels, filename=filenames, meta=batch['meta'])
        self.log_on_final_batch(phase_name)
        return out

    # noinspection PyUnusedLocal
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        img = batch['data']
        labels = batch['ground_truth'].long()
        filenames = batch['filename']
        # batch x num_class
        logits = self(img, 'predict')
        meta = batch['meta']

        loss_to_placeholder = 0 * torch.ones_like(logits)
        filenames_np = np.asarray(filenames)
        filenames_np_to_pred = filenames_np.tolist()

        out = ModelOutput(loss=loss_to_placeholder, logits=logits, ground_truth=labels, filename=filenames_np_to_pred,
                          meta=meta)
        return out

    def _log_on_final_batch_helper(self, phase_name: PHASE_STR):
        self.log_meter(f"{phase_name}_auc", self.auc_metric, logger=True, sync_dist=True)
        self.log_meter(f"{phase_name}_loss", self.loss_avg, logger=True, sync_dist=True)

    def _reset_meters(self):
        self.auc_metric.reset()
        self.loss_avg.reset()
        self.conf_mat.reset()

    def on_train_epoch_end(self) -> None:
        self._reset_meters()

    def on_validation_epoch_end(self) -> None:
        self._reset_meters()
        self.print_newln()

    def on_test_epoch_end(self) -> None:
        self._reset_meters()

    # noinspection PyUnusedLocal
    def training_step(self, batch: ModelInput, batch_idx):
        return self._step(batch, 'fit')

    # noinspection PyUnusedLocal
    def validation_step(self, batch: ModelInput, batch_idx):
        return self._step(batch, 'validate')

    # noinspection PyUnusedLocal
    def testing_step(self, batch: ModelInput, batch_idx):
        return self._step(batch, 'test')

    def load_base_state(self, state_dict):
        self.finetune_base.load_state_dict(state_dict)

    def freeze_pretrained_model(self, freeze_flag: bool):
        for params in self.finetune_base.parameters():
            params.requires_grad_(not freeze_flag)

    @classmethod
    def build_from_lightning(cls,
                             transforms_dict: nn.ModuleDict | Dict[PHASE_STR, Callable],
                             simclr_lightning: SimCLRLightning,
                             num_classes: int,
                             drop_rate: Optional[float] = 0.5,
                             freeze_weight: bool = False,
                             max_t: int = 90,
                             betas=(0.5, 0.99),
                             weight_decay: float = 1e-4,
                             lr: Optional[float] = 1e-3,
                             batch_size: Optional[int] = 64,
                             prog_bar: bool = True,
                             next_line: bool = True
                             ):
        simclr_base = simclr_lightning.model
        finetune_base = BaseFineTune.build_default(simclr_base, num_classes=num_classes, drop_rate=drop_rate)
        return cls(transforms_dict=transforms_dict, finetune_base=finetune_base, freeze_weight=freeze_weight,
                   betas=betas, weight_decay=weight_decay,
                   lr=lr, batch_size=batch_size, prog_bar=prog_bar, next_line=next_line)
