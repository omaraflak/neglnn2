import numpy as np
from neglnn.initializers.initializer import Initializer
from neglnn.utils.types import Array

class XavierNormal(Initializer):
    def get(self, *shape: int) -> Array:
        input_neurons = np.prod(self.target_layer.input_shape)
        output_neurons = np.prod(self.target_layer.output_shape)
        standard_deviation = np.sqrt(2 / (input_neurons + output_neurons))
        return np.random.normal(0, standard_deviation, shape)
