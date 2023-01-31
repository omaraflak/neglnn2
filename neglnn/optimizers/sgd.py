from neglnn.optimizers.optimizer import Optimizer
from neglnn.utils.types import Array, Float

class SGD(Optimizer):
    def __init__(self, learning_rate: Float = 0.01):
        super().__init__()
        self.learning_rate = learning_rate

    def optimize(self, parameter: Array):
        parameter -= self.learning_rate * self._avg_gradient()
