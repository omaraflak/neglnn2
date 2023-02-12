import numpy as np
from neglnn.utils.types import Array, Shape

class Optimizer:
    def record(self, gradient: Array):
        self.gradient += gradient
        self.counter += 1

    def optimize(self, parameter: Array):
        raise NotImplementedError

    def reset(self):
        self.gradient = np.zeros(self.target_shape)
        self.counter = 0

    def on_target_shape(self, shape: Shape):
        self.target_shape = shape
        self.reset()

    def _get_gradient(self) -> Array:
        return self.gradient / self.counter
