import numpy as np

from layers.dense import Dense
from activations.relu import ReLU

if __name__ == '__main__':
    dense = Dense(4, ReLU())
    dense.forward(np.array([2, 1]))
