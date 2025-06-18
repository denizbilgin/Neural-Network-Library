from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class Layer(ABC):
    def __init__(self):
        self.name = self.__class__.__name__
        self.trainable = True

    @abstractmethod
    def forward(self, layer_input: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        pass

    @abstractmethod
    def backward(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_params(self):
        pass
