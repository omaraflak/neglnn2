# NEGLNN 2

**N**ot **E**fficient but **G**reat to **L**earn **N**eural **N**etwork - Graph Computation

# Sequential model

```python
import numpy as np
from neglnn.layers.dense import Dense
from neglnn.activations.tanh import Tanh
from neglnn.losses.mse import MSE
from neglnn.initializers.random_uniform import RandomUniform
from neglnn.optimizers.momentum import Momentum
from neglnn.network.network import Network

x_train = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
y_train = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = Network.sequential([
    Dense(2, 3, initializer=RandomUniform(), optimizer=lambda: Momentum()),
    Tanh(),
    Dense(3, 1, initializer=RandomUniform(), optimizer=lambda: Momentum()),
    Tanh()
])

network.fit(x_train, y_train, MSE(), 1000)
print(network.run_all(x_train))
```

# Graph model

User API still under construction...

```python
import numpy as np
from neglnn.layers.dense import Dense
from neglnn.activations.tanh import Tanh
from neglnn.losses.mse import MSE
from neglnn.layers.layer import Layer
from neglnn.utils.types import Array, InputKey
from neglnn.initializers.random_uniform import RandomUniform
from neglnn.optimizers.momentum import Momentum
from neglnn.network.network import Network
from neglnn.network.graph import Graph


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

dense1 = Dense(2, 3, initializer=RandomUniform(), optimizer=lambda: Momentum())
activation1 = Tanh()
dense2 = Dense(3, 1, initializer=RandomUniform(), optimizer=lambda: Momentum())
activation2 = Tanh()
mult = MultiplicationLayer()

graph = Graph()
graph.connect(dense1, activation1)
graph.connect(activation1, dense2)
graph.connect(dense2, activation2)
# split
graph.connect(activation2, mult, 'a')
graph.connect(dense2, mult, 'b')

network = Network(graph)
network.fit(x_train, y_train, MSE(), 1000)
print(network.run_all(x_train))
```