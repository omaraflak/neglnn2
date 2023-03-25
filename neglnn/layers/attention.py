import numpy as np
from neglnn.layers.layer import Layer
from neglnn.layers.dense import Dense
from neglnn.layers.transpose import Transpose
from neglnn.layers.dot import Dot
from neglnn.layers.input import Input
from neglnn.activations.linear import Linear
from neglnn.activations.softmax_row import SoftmaxRow
from neglnn.network.network import Network
from neglnn.network.graph import Graph
from neglnn.utils.types import Array, InputKey, Shape2

class Attention(Layer):
    def __init__(self, input_shape: Shape2, output_shape: Shape2, key_dimensions: int, **kwargs):
        super().__init__(input_shape=input_shape, output_shape=output_shape, trainable=True, **kwargs)

        input_size = input_shape[0]
        key_shape = (input_size, key_dimensions)

        input_layer = Input()
        q_layer = Dense(input_shape, key_shape, initializer=kwargs['initializer'], optimizer=kwargs['optimizer'])
        k_layer = Dense(input_shape, key_shape, initializer=kwargs['initializer'], optimizer=kwargs['optimizer'])
        v_layer = Dense(input_shape, output_shape, initializer=kwargs['initializer'], optimizer=kwargs['optimizer'])
        k_t_layer = Transpose()
        dot_qkt_layer = Dot()
        dot_qkt_scaled_layer = Linear(c=1 / np.sqrt(key_dimensions))
        softmax_layer = SoftmaxRow(input_size)
        dot_v_layer = Dot()

        graph = Graph()
        graph.connect(input_layer, q_layer)
        graph.connect(input_layer, k_layer)
        graph.connect(input_layer, v_layer)
        graph.connect(k_layer, k_t_layer)
        graph.connect(q_layer, dot_qkt_layer, 'a')
        graph.connect(k_t_layer, dot_qkt_layer, 'b')
        graph.connect(dot_qkt_layer, dot_qkt_scaled_layer)
        graph.connect(dot_qkt_scaled_layer, softmax_layer)
        graph.connect(softmax_layer, dot_v_layer, 'a')
        graph.connect(v_layer, dot_v_layer, 'b')
        self.network = Network(graph, input_layer, dot_v_layer)

    def initialize_parameters(self):
        pass

    def parameters(self) -> list[Array]:
        return []

    def parameters_gradient(self, output_gradient: Array) -> list[Array]:
        return []

    def forward(self, inputs: dict[InputKey, Array]) -> Array:
        return self.network.run(inputs[Graph.INPUT])

    def input_gradient(self, output_gradient: Array) -> dict[InputKey, Array]:
        return {Graph.INPUT: self.network.record_gradient(output_gradient, optimize=False)}

    def optimize(self):
        self.network.optimize()