import numpy as np
from initializers.abstracts.initializer import Initializer


class HeInitializer(Initializer):
    def initialize(self, shape: tuple) -> np.ndarray:
        # shape = (num neurons, num inputs)
        return np.random.randn(shape[0], shape[1]) * np.sqrt(2. / shape[1])
