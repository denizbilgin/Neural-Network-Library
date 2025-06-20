import numpy as np
from typing import Tuple, Optional
from activations.abstracts.activation import Activation
from initializers.abstracts.initializer import Initializer
from initializers.he_initializer import HeInitializer
from layers.abstracts.layer import Layer


class Dense(Layer):
    def __init__(self, num_neurons: int, activation: Activation, initializer: Optional[Initializer] = None):
        super().__init__()
        self.num_neurons = num_neurons
        self.activation = activation
        self.initializer = initializer or HeInitializer()

        self.weights: Optional[np.ndarray] = None     # (num neurons, num inputs)
        self.biases: Optional[np.ndarray] = None      # (num neurons, 1)
        # layer input                                   (num inputs , batch_size)

    def __initialize_parameters(self, layer_input: np.ndarray):
        shape = (self.num_neurons, layer_input.shape[0])
        self.weights = self.initializer.initialize(shape)
        self.biases = np.zeros((self.num_neurons, 1))

    def __linear_forward(self, layer_input: np.ndarray) -> np.ndarray:
        if self.weights is None and self.biases is None:
            self.__initialize_parameters(layer_input)
        linear_output = np.dot(self.weights, layer_input) + self.biases  # Broadcasting for bias
        print(linear_output.shape)
        return linear_output

    def forward(self, layer_input: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        linear_output = self.__linear_forward(layer_input)
        linear_cache = (layer_input, self.weights, self.biases)
        activation_result = self.activation.forward(linear_output)
        cache = (linear_cache, linear_output)
        return activation_result, cache

    def backward(self) -> np.ndarray:
        pass

    def get_params(self):
        pass

    def __str__(self):
        return f'{self.name} with {self.num_neurons} neurons and {self.activation.__class__.__name__} activation.'
