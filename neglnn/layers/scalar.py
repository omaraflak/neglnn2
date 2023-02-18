from neglnn.layers.layer import Layer
from neglnn.network.graph import Graph
from neglnn.utils.types import Array, InputKey

class Scalar(Layer):
    def forward(self, inputs: dict[InputKey, Array]) -> Array:
        input = inputs[Graph.INPUT]
        self.shape = input.shape
        return input.reshape(1)[0]

    def input_gradient(self, output_gradient: Array) -> dict[InputKey, Array]:
        return {Graph.INPUT: output_gradient.reshape(self.shape)}