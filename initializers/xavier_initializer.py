from typing import Tuple
import numpy as np
from initializers.abstracts.initializer import Initializer


class XavierInitializer(Initializer):
    def initialize(self, shape: Tuple) -> np.ndarray:
        # shape = (num neurons, num inputs)
        limit = np.sqrt(6. / (shape[0] + shape[1]))
        return np.random.uniform(-limit, limit, size=shape)
