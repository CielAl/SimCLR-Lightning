from torchmetrics import Metric
from torch import nn
from typing import List, Any, Type, Iterator, cast


class MetricList(Metric):

    def update(self, *_: Any, **__: Any) -> None:
        for m in self.metrics:
            m.update(*_, **__)

    def compute(self) -> Any:
        return [x.compute() for x in self.metrics]

    def forward(self, *_: Any, **__: Any) -> Any:
        return [m(*_, **__) for m in self.metrics]

    def __init__(self, metric_list: List[Metric]):
        super().__init__()
        self.metric_list = nn.ModuleList(metric_list)

    def reset(self):
        for metric in self.metric_list:
            metric.reset()

    @classmethod
    def build_from_type(cls, metric: Type[Metric], num_metrics: int, **kwargs):
        metric_list: List[Metric] = [metric(**kwargs) for _ in range(num_metrics)]
        return cls(metric_list)

    def __iter__(self) -> Iterator[Metric]:
        return cast(Iterator[Metric], iter(self.metric_list))
