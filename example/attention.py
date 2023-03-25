import numpy as np
from neglnn.layers.input import Input
from neglnn.layers.attention import Attention
from neglnn.losses.mse import MSE
from neglnn.initializers.random_uniform import RandomUniform
from neglnn.optimizers.momentum import Momentum
from neglnn.network.network import Network

x_train = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
y_train = np.reshape([[1, 0], [1, 1], [0.5, 1], [0.1, 0]], (4, 2, 1))

n = 2
d = 1
p = 1

network = Network.sequential([
    Input(),
    Attention((n, 1), (n, p), 1, initializer=RandomUniform(), optimizer=lambda: Momentum())
])

network.fit(x_train, y_train, MSE(), 1000)
print(network.run_all(x_train))