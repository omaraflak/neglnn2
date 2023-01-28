from dataclasses import dataclass
from neglnn.utils.types import Array, Shape

@dataclass
class Update:
    parameter: Array
    gradient: Array

class Optimizer:
    def record(self, update: Update):
        raise NotImplementedError

    def optimize(self):
        raise NotImplementedError

    def should_optimize(self) -> bool:
        raise NotImplementedError

    def on_target_shape(self, shape: Shape):
        self.target_shape = shape
