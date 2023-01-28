import numpy as np
from neglnn.layers.layer import Layer
from neglnn.network.graph import Graph
from neglnn.utils.types import Array, InputKey

class Dense(Layer):
    def __init__(self, input_units: int, output_units: int, **kwargs):
        super().__init__((input_units, 1), (output_units, 1), **kwargs)
        self.input_units = input_units
        self.output_units = output_units

    def initialize_parameters(self):
        self.weights = self.initializer.get(self.output_units, self.input_units)
        self.biases = self.initializer.get(self.output_units, 1)

    def parameters(self) -> list[Array]:
        return [self.weights, self.biases]

    def forward(self, inputs: dict[InputKey, Array]) -> Array:
        self.input = inputs[Graph.INPUT]
        return np.dot(self.weights, self.input) + self.biases

    def input_gradient(self, output_gradient: Array) -> dict[InputKey, Array]:
        return {Graph.INPUT: np.dot(self.weights.T, output_gradient)}

    def parameters_gradient(self, output_gradient: Array) -> list[Array]:
        return [np.dot(output_gradient, self.input.T), output_gradient]