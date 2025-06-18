import numpy as np
from activations.abstracts.activation import Activation


class ReLU(Activation):
    def forward(self, linear_output: np.ndarray) -> np.ndarray:
        return np.maximum(0, linear_output)

    def backward(self, d_activation: np.ndarray, cache: np.ndarray) -> np.ndarray:
        d_linear_output = np.array(d_activation, copy=True)  # Creating a copy to avoid modifying the original array
        d_linear_output[cache <= 0] = 0
        return d_linear_output
