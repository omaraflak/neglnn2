from neglnn.layers.layer import Layer
from neglnn.network.graph import Graph
from neglnn.network.network import Network
from neglnn.utils.types import Array, InputKey

class Model(Layer):
    def __init__(self, network: Network, **kwargs):
        super().__init__(**kwargs)
        self.network = network

    def initialize_parameters(self):
        pass
    
    def parameters(self) -> list[Array]:
        return []

    def parameters_gradient(self, output_gradient: Array) -> list[Array]:
        return []

    def forward(self, inputs: dict[InputKey, Array]) -> Array:
        return self.network.run(inputs[Graph.INPUT])

    def input_gradient(self, output_gradient: Array) -> dict[InputKey, Array]:
        return {Graph.INPUT: self.network.record_gradient(output_gradient)}

    def optimize(self):
        self.network.optimize()
