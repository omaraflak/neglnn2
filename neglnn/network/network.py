import numpy as np
from typing import Optional, Callable
from neglnn.layers.layer import Layer
from neglnn.losses.loss import Loss
from neglnn.network.state import State
from neglnn.utils.types import Array, InputKey

class Network:
    def __init__(
        self,
        layers: list[Layer],
        sources: Optional[list[Layer]] = None,
        sink: Optional[Layer] = None
    ):
        self.layers = layers
        self.sources = sources or Network.detect_sources(layers)
        self.sink = sink or Network.detect_sink(layers)
        self.execution_order = Layer.query_graph(self.sources, self.sink)

    def fit(
        self,
        x_train: list[Array],
        y_train: list[Array],
        loss: Loss,
        epochs: int,
        verbose: bool = True,
        callback: Optional[Callable[[State], None]] = None 
    ):
        state = self._initialize()
        state.epochs = epochs
        state.training_samples = len(x_train)

        # training loop
        for i in range(epochs):
            state.current_epoch = i
            cost = 0

            # go through all training samples
            for x, y in zip(x_train, y_train):
                # forward propagation                
                output = self.run(x)

                # error for display purpose
                cost += loss.call(y, output)

                # backward propagation
                output_gradient = loss.prime(y, output)
                self._train(output_gradient)

            cost /= len(x_train)
            state.cost = cost

            if verbose:
                print(f'#{i + 1}/{epochs}\t cost={cost:2f}')
            
            if callback:
                callback(state)

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
                for key, parent in Layer.PARENT_GRAPH[layer].items()
            }
            computed[layer] = layer.forward(dependencies)
        return computed[self.sink]

    def run_all(self, x_list: list[Array]) -> list[Array]:
        return [self.run(x) for x in x_list]

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
                for key, child in Layer.CHILD_GRAPH[layer].items()
            ], axis=0)
            reverse_computed[layer] = layer.input_gradient(children_gradient)
            if layer.trainable:
                layer.optimize(layer.parameters_gradient(children_gradient))

    def _initialize(self) -> State:
        state = State(layers=self.execution_order)
        
        # provide state to layers
        for layer in self.layers:
            layer.on_state(state)

        # provide state to initializers
        for layer in self.layers:
            if layer.trainable:
                layer.initializer.on_state(state)

        # initialize layers parameters
        for layer in self.layers:
            if layer.trainable:
                layer.initialize()

        # initialize layers optimizers
        for layer in self.layers:
            if layer.trainable:
                layer.setup_optimizers()

        return state

    def subnet(self, sources: list[Layer], sink: Layer) -> 'Network':
        return Network(Layer.query_graph(sources, sink))

    def __getitem__(self, subscript) -> 'Network':
        layers = self.layers[subscript]
        return Network(layers, [layers[0]], layers[-1])

    @classmethod
    def sequential(cls, layers: list[Layer]) -> 'Network':
        for i in range(len(layers) - 1):
            layers[i].wire(layers[i + 1])
        return Network(layers, [layers[0]], layers[-1])

    @staticmethod
    def detect_sources(layers: list[Layer]) -> list[Layer]:
        sources = [
            layer
            for layer, parents in Layer.PARENT_GRAPH.items()
            if not parents and layer in layers
        ]
        return sources

    @staticmethod
    def detect_sink(layers: list[Layer]) -> Layer:
        sinks = [
            layer
            for layer, children in Layer.CHILD_GRAPH.items()
            if not children and layer in layers
        ]
        assert len(sinks) == 1
        return sinks[0]