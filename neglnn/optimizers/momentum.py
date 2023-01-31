import numpy as np
from neglnn.optimizers.optimizer import Optimizer
from neglnn.utils.types import Array, Float, Shape

class Momentum(Optimizer):
    def __init__(self, learning_rate: Float = 0.01, mu: Float = 0.95):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu

    def optimize(self, parameter: Array):
        self.v = self.mu * self.v + self.learning_rate * self._avg_gradient()
        parameter -= self.v

    def on_target_shape(self, target_shape: Shape):
        super().on_target_shape(target_shape)
        self.v = np.zeros(target_shape)
