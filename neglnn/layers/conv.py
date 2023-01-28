import numpy as np
from neglnn.layers.layer import Layer
from neglnn.network.graph import Graph
from neglnn.layers.conv_unit import ConvUnit
from neglnn.utils.types import Array, Shape3, InputKey

class Conv(Layer):
    def __init__(self, input_shape: Shape3, kernel_size: int, depth: int, **kwargs):
        self.conv_units = [
            ConvUnit(input_shape, kernel_size, **kwargs)
            for _ in range(depth)
        ]
        height, width = self.conv_units[0].output_shape
        super().__init__(input_shape, (depth, height, width), **kwargs)

    def initialize_parameters(self):
        for unit in self.conv_units:
            unit.initialize_parameters()

        self._parameters = [
            x 
            for unit in self.conv_units
            for x in unit.parameters()
        ]

    def parameters(self) -> list[Array]:
        return self._parameters

    def forward(self, inputs: dict[InputKey, Array]) -> Array:
        return np.array([unit.forward(inputs) for unit in self.conv_units])

    def input_gradient(self, output_gradient: Array) -> dict[InputKey, Array]:
        input_gradients = [
            unit.input_gradient(grad)[Graph.INPUT]
            for unit, grad in zip(self.conv_units, output_gradient)
        ]
        return {Graph.INPUT: np.sum(input_gradients, axis=0)}

    def parameters_gradient(self, output_gradient: Array) -> list[Array]:
        return [
            gradient
            for unit, grad in zip(self.conv_units, output_gradient)
            for gradient in unit.parameters_gradient(grad)
        ]
