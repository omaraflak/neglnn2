import numpy as np
from neglnn.layers.layer import Layer
from neglnn.utils.types import Array, InputKey

class Softmax(Layer):
    def forward(self, inputs: dict[InputKey, Array]) -> Array:
        input = inputs[Layer.SINGLE_INPUT]
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output

    def input_gradient(self, output_gradient: Array) -> dict[InputKey, Array]:
        n = np.size(self.output)
        tmp = np.identity(n) - np.transpose(self.output)
        input_gradient = np.dot(tmp * self.output, output_gradient)
        return {Layer.SINGLE_INPUT: input_gradient}