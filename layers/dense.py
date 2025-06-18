import numpy as np
from layers.abstracts.layer import Layer


class Dense(Layer):
    def __init__(self, num_neurons: int):
        super().__init__()
        self.num_neurons = num_neurons
        self.weights = None     # num neurons, num inputs
        self.biases = None      # num neurons, 1
        self.layer_input = None

    def __linear_forward(self) -> np.ndarray:
        pass

    def __linear_activation_forward(self) -> np.ndarray:
        pass

    def forward(self) -> np.ndarray:
        pass

    def backward(self) -> np.ndarray:
        pass

    def get_params(self):
        pass

    def set_params(self):
        pass


