from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


class Initializer(ABC):
    @abstractmethod
    def initialize(self, shape: Tuple) -> np.ndarray:
        pass
