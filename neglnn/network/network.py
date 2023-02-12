import pickle
import numpy as np
from typing import Optional
from neglnn.layers.layer import Layer
from neglnn.losses.loss import Loss
from neglnn.network.graph import Graph
from neglnn.utils.types import Array, InputKey

class Network:
    @classmethod
    def sequential(cls, layers: list[Layer], initialize_layers: bool = True) -> 'Network':
        graph = Graph()
        for i in range(len(layers) - 1):
            graph.connect(layers[i], layers[i + 1])
        return Network(graph, initialize_layers=initialize_layers)

    def __init__(
        self,
        graph: Graph,
        source: Optional[Layer] = None,
        sink: Optional[Layer] = None,
        initialize_layers: bool = True
    ):
        self.graph = graph
        self.source = source or graph.get_sources()[0]
        self.sink = sink or graph.get_sinks()[0]
        self.layers = graph.get_ordered_dependencies(self.source, self.sink)
        if initialize_layers:
            self._initialize_layers()

    def fit(
        self,
        x_train: list[Array],
        y_train: list[Array],
        loss: Loss,
        epochs: int,
        batch_size: int = 1,
        verbose: bool = True
    ):
        for i in range(epochs):
            cost = 0
            for j, (x, y) in enumerate(zip(x_train, y_train)):
                output = self.run(x)
                cost += loss.call(y, output)
                output_gradient = loss.prime(y, output)
                self.record_gradient(output_gradient, optimize=(j + 1) % batch_size == 0)
            cost /= len(x_train)
            if verbose:
                print(f'#{i + 1}/{epochs}\t cost={cost:.10f}')

    def run(self, x: Array) -> Array:
        computed: dict[Layer, Array] = dict()
        for layer in self.layers:
            if layer == self.source:
                dependencies = {Graph.INPUT: x}
            else:
                dependencies = {
                    key: computed[parent]
                    for key, parent in self.graph.parents[layer].items()
                }
            computed[layer] = layer.forward(dependencies)
        return computed[self.sink]

    def run_all(self, x_list: list[Array]) -> list[Array]:
        return [self.run(x) for x in x_list]

    def record_gradient(self, output_gradient: Array, optimize: bool = True) -> Array:
        reverse_computed: dict[Layer, dict[InputKey, Array]] = dict()
        for layer in reversed(self.layers):
            if layer == self.sink:
                gradient = output_gradient
            else:
                gradient = np.sum([
                    reverse_computed[child][key]
                    for key, child in self.graph.children[layer].items()
                ], axis=0)

            reverse_computed[layer] = layer.input_gradient(gradient)
            if layer.trainable:
                layer.record_gradients(layer.parameters_gradient(gradient))
                if optimize:
                    layer.optimize()
        return reverse_computed[self.source][Graph.INPUT]

    def optimize(self):
        for layer in self.layers:
            if layer.trainable:
                layer.optimize()

    def subnet(self, source: Layer, sink: Layer) -> 'Network':
        return Network(self.graph.copy(source, sink))

    def save(self, filename: str):
        pickle.dump(self, open(filename, 'wb'))

    @classmethod
    def load(cls, filename: str) -> 'Network':
        return pickle.load(open(filename, 'rb'))

    def __getitem__(self, subscript) -> 'Network':
        return Network.sequential(self.layers[subscript], initialize_layers=False)

    def _initialize_layers(self):
        for layer in self.layers:
            if layer.trainable:
                layer.initialize_parameters()
                layer.setup_optimizers()
