from abc import ABC, abstractmethod


class TDID(ABC):
    def save_objects(self, ds):
        pass

    def predict(self, x):
        pass