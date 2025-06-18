from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):
    def __init__(self):
        self.name = self.__class__.__name__
        self.trainable = True

    @abstractmethod
    def forward(self) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def set_params(self):
        pass
