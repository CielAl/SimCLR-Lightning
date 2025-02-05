from torch import nn
from simclr_lightning.models.lightning_modules.base import PHASE_TRAIN, PHASE_VAL, PHASE_TEST, PHASE_PRED, PHASE_STR
from typing import List, Type, Dict, get_args, cast, Callable
import torchmetrics


class MetricDict(nn.ModuleDict):

    def __init__(self):
        super().__init__()

    @classmethod
    def build_from_args(cls, mode_list: List[PHASE_STR], metric: Type[torchmetrics.Metric], **kwargs):
        m_dict = cls()
        for mode in mode_list:
            m_dict[mode] = metric(**kwargs)
        return m_dict

    @classmethod
    def build_all_modes(cls, metric: Type[torchmetrics.Metric] | Callable, **kwargs):
        mode_list = list(get_args(PHASE_STR))
        return cls.build_from_args(mode_list, metric, **kwargs)

    @classmethod
    def build_dict(cls, metric_dict: Dict[PHASE_STR, torchmetrics.Metric]):
        m_dict = cls()
        for mode, metric in metric_dict.items():
            m_dict[mode] = metric
        return m_dict

    def reset_all(self):
        for metric in self.values():
            metric.reset()

    def reset_by_mode(self, mode: PHASE_STR):
        self[mode].reset()

    def get_metric(self, mode: PHASE_STR) -> torchmetrics.Metric:
        return cast(torchmetrics.Metric, self[mode])
