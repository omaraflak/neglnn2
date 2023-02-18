import numpy as np
from neglnn.layers.dense import Dense
from neglnn.layers.scalar import Scalar
from neglnn.layers.reshape import Reshape
from neglnn.activations.tanh import Tanh
from neglnn.losses.mse import MSE
from neglnn.initializers.random_uniform import RandomUniform
from neglnn.optimizers.momentum import Momentum
from neglnn.network.network import Network

x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_train = [0, 1, 1, 0]

network = Network.sequential([
    Reshape((2,), (2, 1)),
    Dense(2, 3, initializer=RandomUniform(), optimizer=lambda: Momentum()),
    Tanh(),
    Dense(3, 1, initializer=RandomUniform(), optimizer=lambda: Momentum()),
    Tanh(),
    Scalar()
])

network.fit(x_train, y_train, MSE(), 1000)
print(network.run_all(x_train))