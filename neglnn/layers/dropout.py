import numpy as np
from neglnn.layers.layer import Layer
from neglnn.network.graph import Graph
from neglnn.utils.types import Array, Float, InputKey

class Dropout(Layer):
    def __init__(self, probability: Float = 0.3, training: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.probability = probability
        self.training = training

    def forward(self, inputs: dict[InputKey, Array]) -> Array:
        input = inputs[Graph.INPUT]
        if not self.training:
            return input
        self.kept = np.random.rand(*input.shape) > self.probability
        return self.kept * input

    def input_gradient(self, output_gradient: Array) -> dict[InputKey, Array]:
        return {Graph.INPUT: self.kept * output_gradient}