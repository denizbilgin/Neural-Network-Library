from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    @abstractmethod
    def forward(self, linear_output: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, d_activation: np.ndarray, cache: np.ndarray) -> np.ndarray:
        pass
