import numpy as np
from scipy import signal
from neglnn.layers.layer import Layer
from neglnn.utils.types import Array, Shape3, InputKey

class ConvUnit(Layer):
    def __init__(self, input_shape: Shape3, kernel_size: int, **kwargs):
        input_depth, input_height, input_width = input_shape
        output_shape = (
            input_height - kernel_size + 1,
            input_width - kernel_size + 1
        )
        super().__init__(input_shape, output_shape, trainable=True, **kwargs)
        self.kernels_shape = (input_depth, kernel_size, kernel_size)
        self.bias_shape = output_shape

    def initialize(self):
        self.kernels = self.initializer.get(*self.kernels_shape)
        self.bias = self.initializer.get(*self.bias_shape)

    def parameters(self) -> list[Array]:
        return [self.kernels, self.bias]

    def forward(self, inputs: dict[InputKey, Array]) -> Array:
        self.input = inputs[Layer.INPUT]
        return self.bias + np.sum([
            signal.correlate2d(image, kernel, 'valid')
            for image, kernel in zip(self.input, self.kernels)
        ], axis=0)

    def input_gradient(self, output_gradient: Array) -> dict[InputKey, Array]:
        input_gradient = np.array([
            signal.convolve2d(output_gradient, kernel, 'full')
            for kernel in self.kernels
        ])
        return {Layer.INPUT: input_gradient}

    def parameters_gradient(self, output_gradient: Array) -> list[Array]:
        kernels_gradient = np.array([
            signal.correlate2d(image, output_gradient, 'valid')
            for image in self.input
        ])
        return [kernels_gradient, output_gradient]