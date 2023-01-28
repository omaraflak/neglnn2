from neglnn.optimizers.optimizer import Optimizer, Update
from neglnn.utils.types import Float

class SGD(Optimizer):
    def __init__(self, learning_rate: Float = 0.01):
        super().__init__()
        self.learning_rate = learning_rate

    def optimize(self):
        self.update.parameter -= self.learning_rate * self.update.gradient