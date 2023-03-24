import numpy as np
from neglnn.layers.layer import Layer
from neglnn.network.graph import Graph
from neglnn.utils.types import Array, InputKey

class Transpose(Layer):
    def forward(self, inputs: dict[InputKey, Array]) -> Array:
        input = inputs[Graph.INPUT]
        return np.transpose(input)

    def input_gradient(self, output_gradient: Array) -> dict[InputKey, Array]:
        return {Graph.INPUT: np.transpose(output_gradient)}
