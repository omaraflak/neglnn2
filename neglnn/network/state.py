from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
from neglnn.utils.types import Float, Shape

if TYPE_CHECKING:
    from neglnn.layers.layer import Layer

@dataclass
class State:
    epochs: int = 0
    training_samples: int = 0
    current_epoch: int = 0
    current_layer: Optional['Layer'] = None
    cost: Float = 0

    @property
    def current_layer_input_shape(self) -> Shape:
        return self.current_layer.input_shape

    @property
    def current_layer_output_shape(self) -> Shape:
        return self.current_layer.output_shape

class Stateful:
    def on_state(self, state: State):
        self.state = state