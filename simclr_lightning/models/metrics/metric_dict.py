from torch import nn
from simclr_lightning.models.lightning_modules.base import PHASE_STR
from typing import List, Type, Dict, get_args, cast, Callable
import torchmetrics


class MetricDict(nn.ModuleDict):

    def __init__(self):
        super().__init__()

    @classmethod
    def build_from_args(cls, mode_list: List[str], metric: Type[torchmetrics.Metric], **kwargs):
        m_dict = cls()
        for mode in mode_list:
            m_dict[mode] = metric(**kwargs)
        return m_dict

    @classmethod
    def build_all_modes(cls, metric: Type[torchmetrics.Metric] | Callable, **kwargs):
        mode_list = list(get_args(PHASE_STR))
        return cls.build_from_args(mode_list, metric, **kwargs)

    @classmethod
    def build_dict(cls, metric_dict: Dict[str, torchmetrics.Metric]):
        m_dict = cls()
        for mode, metric in metric_dict.items():
            m_dict[mode] = metric
        return m_dict

    def reset_all(self, *args, **kwargs):
        for metric in self.values():
            metric.reset(*args, **kwargs)

    def reset_by_mode(self, mode: str, *args, **kwargs):
        if mode not in self.keys():
            raise ValueError(f'mode {mode} is not supported')
            # return
        self[mode].reset(*args, **kwargs)

    def get_metric(self, mode: str) -> torchmetrics.Metric:
        return cast(torchmetrics.Metric, self[mode])
