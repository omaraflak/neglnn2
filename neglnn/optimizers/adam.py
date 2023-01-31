import numpy as np
from neglnn.optimizers.optimizer import Optimizer
from neglnn.utils.types import Array, Shape

class Adam(Optimizer):
    def __init__(self, learning_rate=0.0001, beta_1=0.9, beta_2=0.999, eps=1e-8):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.iteration = 1
        
    def optimize(self, parameter: Array):
        gradient = self._avg_gradient()
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * np.power(gradient, 2)
        m_hat = self.m / (1 - np.power(self.beta_1, self.iteration))
        v_hat = self.v / (1 - np.power(self.beta_2, self.iteration))
        self.iteration += 1
        parameter -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

    def on_target_shape(self, target_shape: Shape):
        super().on_target_shape(target_shape)
        self.m = np.zeros(target_shape)
        self.v = np.zeros(target_shape)
