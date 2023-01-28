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
        sources: Optional[list[Layer]] = None,
        sink: Optional[Layer] = None,
        initialize_layers: bool = True
    ):
        self.graph = graph
        self.sources = sources or graph.get_sources()
        self.sink = sink or graph.get_sinks()[0]
        self.execution_order = graph.query(self.sources, self.sink)
        if initialize_layers:
            self._initialize_layers()

    def fit(
        self,
        x_train: list[Array],
        y_train: list[Array],
        loss: Loss,
        epochs: int,
        verbose: bool = True
    ):
        for i in range(epochs):
            cost = 0
            for x, y in zip(x_train, y_train):
                output = self.run(x)
                cost += loss.call(y, output)
                output_gradient = loss.prime(y, output)
                self._train(output_gradient)
            cost /= len(x_train)
            if verbose:
                print(f'#{i + 1}/{epochs}\t cost={cost:2f}')

    def run(self, x: Array) -> Array:
        computed: dict[Layer, Array] = dict()
        for layer in self.execution_order:
            if layer in self.sources:
                keys = layer.input_keys()
                assert len(keys) == 1
                computed[layer] = layer.forward({keys[0]: x})
                continue

            dependencies = {
                key: computed[parent]
                for key, parent in self.graph.parents[layer].items()
            }
            computed[layer] = layer.forward(dependencies)
        return computed[self.sink]

    def run_all(self, x_list: list[Array]) -> list[Array]:
        return [self.run(x) for x in x_list]

    def subnet(self, sources: list[Layer], sink: Layer) -> 'Network':
        return Network(self.graph.copy(sources, sink))

    def __getitem__(self, subscript) -> 'Network':
        return Network.sequential(self.execution_order[subscript], initialize_layers=False)

    def _train(self, output_gradient: Array):
        reverse_computed: dict[Layer, dict[InputKey, Array]] = dict()
        for layer in reversed(self.execution_order):
            if layer == self.sink:
                reverse_computed[layer] = layer.input_gradient(output_gradient)
                if layer.trainable:
                    layer.optimize(layer.parameters_gradient(output_gradient))
                continue

            children_gradient = np.sum([
                reverse_computed[child][key]
                for key, child in self.graph.children[layer].items()
            ], axis=0)
            reverse_computed[layer] = layer.input_gradient(children_gradient)
            if layer.trainable:
                layer.optimize(layer.parameters_gradient(children_gradient))

    def _initialize_layers(self):
        for layer in self.graph.layers:
            if layer.trainable:
                layer.initialize_parameters()
                layer.setup_optimizers()
