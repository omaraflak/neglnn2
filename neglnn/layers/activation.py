from neglnn.layers.layer import Layer
from neglnn.network.graph import Graph
from neglnn.utils.types import Array, InputKey

class Activation(Layer):
    def call(self, x: Array) -> Array:
        raise NotImplementedError()

    def prime(self, x: Array) -> Array:
        raise NotImplementedError()

    def forward(self, inputs: dict[InputKey, Array]) -> Array:
        self.input = inputs[Graph.INPUT]
        return self.call(self.input)

    def input_gradient(self, output_gradient: Array) -> dict[InputKey, Array]:
        return {Graph.INPUT: output_gradient * self.prime(self.input)}