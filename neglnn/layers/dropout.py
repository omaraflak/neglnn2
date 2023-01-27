import numpy as np
from neglnn.layers.layer import Layer
from neglnn.utils.types import Array, Float, InputKey

class Dropout(Layer):
    def __init__(self, probability: Float = 0.3, training: bool = True, **kwargs):
        super().__init__(trainable=False, **kwargs)
        self.probability = probability
        self.training = training

    def forward(self, inputs: dict[InputKey, Array]) -> Array:
        input = inputs[Layer.SINGLE_INPUT]
        if self.training:
            return input
        self.kept = np.random.rand(*input.shape) > self.probability
        return self.kept * input

    def input_gradient(self, output_gradient: Array) -> dict[InputKey, Array]:
        return {Layer.SINGLE_INPUT: self.kept * output_gradient}