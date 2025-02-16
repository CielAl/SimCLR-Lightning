import pytorch_lightning as L
import torchmetrics
from typing import Literal, Callable
from abc import abstractmethod
import torch
import numpy as np

PHASE_TRAIN = Literal['fit']
PHASE_VAL = Literal['validate']
PHASE_TEST = Literal['test']
PHASE_PRED = Literal['predict']
PHASE_STR = Literal[PHASE_TRAIN, PHASE_VAL, PHASE_TEST, PHASE_PRED]


def feature_norm_penalty(feature_map: torch.Tensor,
                         target_norm: torch.Tensor | float = 1000.0,
                         lambda_scale: torch.Tensor | float = 1e-3):
    norm = feature_map.view(feature_map.size(0), -1).norm(p=2, dim=1)
    loss = lambda_scale * ((norm - target_norm) ** 2).mean()  # Penalize deviation
    return loss


class BaseLightningModule(L.LightningModule):
    WARM_UP_EPOCH: int = 10

    lr: float
    batch_size: int
    prog_bar: bool
    # whether print new line after each epoch
    next_line: bool
    max_t: int
    optim_func: Callable

    def param_groups(self, module: torch.nn.Module, weight_decay: float):
        """Set up the optimizer and lr scheduler - adapted from https://theaisummer.com/simclr/


        Returns:
            See LightningModule for more detail. Usually it returns a single optimizer or
            Tuple[List[optimizer], List[scheduler]]/
        """
        def exclude_from_wd_and_adaptation(name):
            if 'bn' in name:
                return True

        param_groups = [
            {
                'params': [p for name, p in module.named_parameters() if not exclude_from_wd_and_adaptation(name)],
                'weight_decay': weight_decay,
                'layer_adaptation': True,
            },
            {
                'params': [p for name, p in module.named_parameters() if exclude_from_wd_and_adaptation(name)],
                'weight_decay': 0.,
                'layer_adaptation': False,
            },
        ]
        return param_groups

    def configure_optimizers_helper(self, optim_func: Callable):
        """Set up the optimizer and lr scheduler - adapted from https://theaisummer.com/simclr/


        Returns:
            See LightningModule for more detail. Usually it returns a single optimizer or
            Tuple[List[optimizer], List[scheduler]]/
        """

        params_groups = self.param_groups(self, self.weight_decay)
        optimizer = optim_func(params_groups, lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_t, eta_min=0,
                                                               last_epoch=-1)

        return [optimizer], [scheduler]

    def configure_optimizers(self):
        return self.configure_optimizers_helper(self.optim_func)

    def log_meter(self, name: str, metric: torchmetrics.Metric | torch.Tensor, on_step: bool = False,
                  on_epoch: bool = True, sync_dist: bool = True,
                  logger: bool = True):
        value = metric.compute() if isinstance(metric, torchmetrics.Metric) else metric
        if isinstance(value, (torch.Tensor, np.ndarray)) and value.nelement() == 1:
            value = value.item()
        self.log(name, value, on_step=on_step, on_epoch=on_epoch, prog_bar=self.prog_bar,
                 logger=logger, batch_size=self.batch_size, sync_dist=sync_dist)
        return value

    def scheduler_step(self) -> None:
        sch = self.lr_schedulers()
        # self.log("LR", sch.get_last_lr(), on_epoch=True, prog_bar=self.prog_bar,
        #          logger=True, batch_size=self.batch_size, sync_dist=True)
        # sch = scheduler
        if self.trainer.training and self.trainer.is_last_batch and self.trainer.current_epoch >= self.WARM_UP_EPOCH:
            sch.step()  # metrics=metric, epoch=self.trainer.current_epoch

    def __init__(self, batch_size: int, lr: float, max_t: int, prog_bar: bool, next_line: bool,
                 optim_func: Callable = torch.optim.Adam,
                 ):
        """

        Args:
            batch_size: batch_size
            lr: learning rate
            max_t: max_t for CosAnnealingScheduler to restart
            prog_bar: whether to log results in progress bars
            next_line: whether to print a new line after each validation epoch. This enables the default tqdm progress
                bar to retain the results of previous epochs in previous lines.
        """
        super().__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.prog_bar = prog_bar
        self.next_line = next_line
        self.max_t = max_t
        self.optim_func = optim_func

    def reset_meter_phase(self, *args, **kwargs):
        """reset all torchmetrics meters

        Args:
            *args:
            **kwargs:

        Returns:

        """
        raise NotImplementedError

    def print_newln(self):
        if self.next_line:
            print("\n")

    def _reset_on_first_batch(self, batch_idx: int, phase_name: PHASE_STR):
        if batch_idx == 0:
            # breakpoint()
            self.reset_meter_phase(phase_name)

    def reset_meter_all(self):
        raise NotImplementedError
