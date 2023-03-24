import numpy as np
from neglnn.layers.layer import Layer
from neglnn.utils.types import Array, InputKey

class Dot(Layer):
    def forward(self, inputs: dict[InputKey, Array]) -> Array:
        self.a = inputs['a']
        self.b = inputs['b']
        return np.dot(self.a, self.b)

    def input_gradient(self, output_gradient: Array) -> dict[InputKey, Array]:
        return {
            'a': np.dot(output_gradient, self.b.T),
            'b': np.dot(self.a.T, output_gradient)
        }
