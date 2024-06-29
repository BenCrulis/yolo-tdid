from avalanche.evaluation import Metric
from avalanche.evaluation import GenericPluginMetric


class ObjectCount(Metric[float]):
    def __init__(self):
        self._count = 0

    def update(
        self,
        val,
    ) -> None:
        self._count = val

    def result(self) -> float:
        return self._count

    def reset(self) -> None:
        self._count = 0


class SavedObjectCountPluginMetric(GenericPluginMetric[float, ObjectCount]):
    """
    Base class for all accuracies plugin metrics
    """

    def __init__(self, reset_at, emit_at, mode):
        super().__init__(ObjectCount(), reset_at=reset_at, emit_at=emit_at, mode=mode)

    def reset(self) -> None:
        self._metric.reset()

    def result(self) -> float:
        return self._metric.result()

    def update(self, strategy):
        if hasattr(strategy, "feats"):
            self._metric.update(len(strategy.feats))
        pass

    def __str__(self):
        return "Saved_Objects"