from neglnn.layers.layer import Layer
from neglnn.network.graph import Graph
from neglnn.utils.types import Array, InputKey

class Input(Layer):
    def forward(self, inputs: dict[InputKey, Array]) -> Array:
        return inputs[Graph.INPUT]

    def input_gradient(self, output_gradient: Array) -> dict[InputKey, Array]:
        return {Graph.INPUT: output_gradient}