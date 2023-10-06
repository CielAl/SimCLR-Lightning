from typing import Tuple, Optional, Literal
import torch
import torchmetrics
from simclr_lightning.models.contrast_learning.loss import InfoNCELoss
from simclr_lightning.models.lightning_modules.base import PHASE_STR, BaseLightningModule
from simclr_lightning.models.contrast_learning.base import AbstractBaseModel
from simclr_lightning.dataset.data_class import ModelInput, ModelOutput


OPTIM_ADAM = Literal['adam']
# todo lars optimizer
OPTIM_LARS = Literal['lars']
SUPPORTED_OPTIM = Literal[OPTIM_ADAM, OPTIM_LARS]


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

    @property
    def n_views(self):
        return self.model.n_views

    def __init__(self,
                 base_model: AbstractBaseModel,
                 lr: Optional[float] = 1e-3,
                 batch_size: Optional[int] = 64,
                 temperature: Optional[float] = 0.07,
                 max_t: Optional[int] = 90,  # length dl
                 betas=(0.5, 0.99),
                 weight_decay: float = 0,
                 prog_bar: bool = True,
                 next_line: bool = True
                 ):
        super(SimCLRLightning, self).__init__(batch_size, lr, prog_bar, next_line)

        # params
        self.temperature = temperature
        self.max_t = max_t
        self.betas = betas
        self.weight_decay = weight_decay

        # model
        self.model = base_model

        # contrastive loss
        self.info_nce_loss = InfoNCELoss(self.batch_size, self.n_views, self.temperature)

        # cross entropy loss for classification of positive/negative pairs
        self.criterion = torch.nn.CrossEntropyLoss()

        # meters for loss and acc
        num_classes = self.n_views * (self.batch_size - 1) + 1
        top_k = min(5, num_classes)
        self.accuracy = torchmetrics.classification.Accuracy(task="multiclass",
                                                             num_classes=2 * self.batch_size - 1, top_k=top_k)
        # calculate epoch-level mean cross-entropy loss
        self.loss_avg = torchmetrics.MeanMetric()

        # misc

    def forward(self, x):
        return self.model(x)

    def _step(self, batch: ModelInput, phase_name: PHASE_STR):
        """Step function helper shared by training and validation steps which computes the logits and log the loss.

        Args:
            batch: batch data in format of NetInput
            phase_name: the name of current phase, i.e., train or validation, for purpose of loss logging.

        Returns:
            NetOutput containing loss, logits (final-layer output) and true labels.
        """
        # stacked view of original and augmented images
        images = batch['data']

        # obtain the projection
        logits = self(images)
        logits, labels = self.info_nce_loss(logits)

        loss = self.criterion(logits, labels)
        self.accuracy.update(logits, labels)
        self.loss_avg.update(loss)
        filenames = batch['filename']

        out = ModelOutput(loss=loss, logits=logits, ground_truth=labels, filename=filenames, meta=batch['meta'])
        self.log_on_final_batch(phase_name)
        return out

    def training_step(self, batch: ModelInput, batch_idx):
        out = self._step(batch, 'train')
        self.scheduler_step()
        return out

    def validation_step(self, batch: ModelInput, batch_idx):
        out = self._step(batch, 'validation')
        return out

    def predict_step(self, batch: ModelInput, batch_idx: int, dataloader_idx: int = 0):
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

    def log_on_final_batch(self, phase_name: PHASE_STR):
        if not self.trainer.is_last_batch:
            return

        self.log_meter(f"{phase_name}_acc", self.accuracy, logger=True, sync_dist=True)
        self.log_meter(f"{phase_name}_loss", self.loss_avg, logger=True, sync_dist=True)

    def _reset_meters(self):
        self.accuracy.reset()
        self.loss_avg.reset()

    def on_train_epoch_end(self) -> None:
        self._reset_meters()

    def on_validation_epoch_end(self) -> None:
        self._reset_meters()
        self.print_newln()
