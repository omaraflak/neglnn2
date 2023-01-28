from typing import TYPE_CHECKING
from neglnn.utils.types import Array

if TYPE_CHECKING:
    from neglnn.layers.layer import Layer

class Initializer:
    def get(self, *shape: int) -> Array:
        raise NotImplementedError

    def on_target_layer(self, layer: 'Layer'):
        self.target_layer = layer
