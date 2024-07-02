import time

from avalanche.evaluation import Metric
from avalanche.evaluation import GenericPluginMetric
from avalanche.training.templates import SupervisedTemplate


class TimeCount(Metric[float]):
    def __init__(self):
        self._count = 0.0

    def update(
        self,
        val,
    ) -> None:
        self._count = val

    def result(self) -> float:
        return self._count

    def reset(self) -> None:
        self._count = 0


class TrainTime(GenericPluginMetric[float, TimeCount]):

    def __init__(self):
        super().__init__(TimeCount(), reset_at="iteration", emit_at="iteration", mode="train")

    def reset(self) -> None:
        self._metric.reset()

    def result(self) -> float:
        return self._metric.result()

    def before_training_iteration(self, strategy: SupervisedTemplate):
        self.time_before = time.time()
        return super().before_training_iteration(strategy)
    
    def after_training_iteration(self, strategy: SupervisedTemplate):
        self.time_after = time.time()
        return super().after_training_iteration(strategy)

    def update(self, strategy):
        assert hasattr(self, "time_after")
        if hasattr(strategy, "train_time"):
            self._metric.update(strategy.train_time)
        else:
            elapsed = self.time_after - self.time_before
            self._metric.update(elapsed)
        pass

    def __str__(self):
        return "Train_time"