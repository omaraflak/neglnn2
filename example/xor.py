import numpy as np

from neglnn.layers.dense import Dense
from neglnn.activations.tanh import Tanh
from neglnn.losses.mse import MSE
from neglnn.layers.layer import Layer
from neglnn.utils.types import Array, InputKey
from neglnn.initializers.random_uniform import RandomUniform
from neglnn.optimizers.momentum import Momentum
from neglnn.network.network import Network


class MultiplicationLayer(Layer):
    def input_keys(self) -> list[InputKey]:
        return ['a', 'b']

    def forward(self, inputs: dict[InputKey, Array]) -> Array:
        self.a = inputs['a']
        self.b = inputs['b']
        return np.multiply(self.a, self.b)

    def input_gradient(self, output_gradient: Array) -> dict[InputKey, Array]:
        return {
            'a': np.multiply(output_gradient, self.b),
            'b': np.multiply(output_gradient, self.a)
        }


x_train = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
y_train = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

layers: list[Layer] = [
    Dense(2, 3, initializer=RandomUniform(), optimizer=lambda: Momentum()),
    Tanh(),
    Dense(3, 1, initializer=RandomUniform(), optimizer=lambda: Momentum()),
    Tanh(),
    MultiplicationLayer()
]


layers[0].wire(layers[1])
layers[1].wire(layers[2])
layers[2].wire(layers[3])
layers[3].wire(layers[4], 'a')
layers[2].wire(layers[4], 'b')

network = Network(layers)

network.fit(x_train, y_train, MSE(), 1000)
print(network.run_all(x_train))