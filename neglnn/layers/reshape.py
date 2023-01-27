import numpy as np
from neglnn.layers.layer import Layer
from neglnn.utils.types import Array, Shape, InputKey

class Reshape(Layer):
    def __init__(self, input_shape: Shape, output_shape: Shape):
        super().__init__(input_shape, output_shape)

    def forward(self, inputs: dict[InputKey, Array]) -> Array:
        input = inputs[Layer.SINGLE_INPUT]
        return np.reshape(input, self.output_shape)

    def input_gradient(self, output_gradient: Array) -> dict[InputKey, Array]:
        return {Layer.SINGLE_INPUT: np.reshape(output_gradient, self.input_shape)}