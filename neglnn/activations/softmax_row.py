import numpy as np
from neglnn.layers.layer import Layer
from neglnn.activations.softmax import Softmax
from neglnn.network.graph import Graph
from neglnn.utils.types import Array, InputKey

class SoftmaxRow(Layer):
    def __init__(self, rows: int):
        super().__init__()
        self.softmax_layers = [
            Softmax()
            for _ in range(rows)
        ]

    def forward(self, inputs: dict[InputKey, Array]) -> Array:
        input = inputs[Graph.INPUT]
        return np.vstack([
            self.softmax_layers[i].forward({
                Graph.INPUT: np.reshape(input[i], (input.shape[1], 1))
            }).flatten()
            for i in range(input.shape[0])
        ])

    def input_gradient(self, output_gradient: Array) -> dict[InputKey, Array]:
        return {
            Graph.INPUT: np.vstack([
                self.softmax_layers[i].input_gradient(np.reshape(output_gradient[i], (output_gradient.shape[1], 1)))[Graph.INPUT].flatten()
                for i in range(output_gradient.shape[0])
            ])
        }
