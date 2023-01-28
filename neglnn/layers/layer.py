from collections import deque
from typing import Callable, Optional
from neglnn.network.state import Stateful
from neglnn.initializers.initializer import Initializer
from neglnn.optimizers.optimizer import Optimizer, Update
from neglnn.utils.types import Array, Shape, InputKey
from neglnn.utils.identifiable import Identifiable

"""
Base Laye class. Initialization order is the following:
- `on_state`
- `on_initializer`
- `on_optimizer`
"""
class Layer(Stateful, Identifiable):
    PARENT_GRAPH: dict['Layer', dict[InputKey, 'Layer']] = dict()
    CHILD_GRAPH: dict['Layer', dict[InputKey, 'Layer']] = dict()
    INPUT = 'input'

    def __init__(
        self,
        input_shape: Optional[Shape] = None,
        output_shape: Optional[Shape] = None,
        trainable: bool = False,
        initializer: Optional[Initializer] = None,
        optimizer: Optional[Callable[[], Optimizer]] = None,
        visible_to_graph: bool = True
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.trainable = trainable
        self.initializer = initializer
        self.optimizer = optimizer
        self.visible_to_graph = visible_to_graph
        if self.visible_to_graph:
            Layer.PARENT_GRAPH[self] = dict()
            Layer.CHILD_GRAPH[self] = dict()

    def initialize(self):
        """
        Initialize the trainable parameters here using the `initializer`
        """
        raise NotImplementedError

    def setup_optimizers(self):
        self.optimizers: list[Optimizer] = []
        for parameter in self.parameters():
            optimizer = self.optimizer()
            optimizer.on_state(self.state)
            optimizer.on_target_shape(parameter.shape)
            self.optimizers.append(optimizer)

    def input_keys(self) -> list[InputKey]:
        """
        Returns input keys that are later passed to the forward method.
        """
        return ['input']

    def parameters(self) -> list[Array]:
        """
        Return a list of all the trainable parameters
        """
        raise NotImplementedError()

    def forward(self, inputs: dict[InputKey, Array]) -> Array:
        """
        Returns the output of the layer.
        """
        raise NotImplementedError()
    
    def input_gradient(self, output_gradient: Array) -> dict[InputKey, Array]:
        """
        Returns the derivative of the output of the layer with respect to
        each input.
        """
        raise NotImplementedError()

    def parameters_gradient(self, output_gradient: Array) -> list[Array]:
        """
        Returns the derivative of the output of the layer with respect to
        each parameter, in the order of which `parameters()` provided them.
        """
        raise NotImplementedError()

    def optimize(self, parameter_gradients: list[Array]):
        for optimizer, parameter, gradient in zip(
            self.optimizers,
            self.parameters(),
            parameter_gradients
        ):
            optimizer.record(Update(parameter, gradient))
            if optimizer.should_optimize():
                optimizer.optimize()

    def wire(self, layer: 'Layer', key: InputKey = INPUT):
        Layer.CHILD_GRAPH[self][key] = layer
        Layer.PARENT_GRAPH[layer][key] = self

    def __hash__(self) -> int:
        return hash(self.uid)

    @staticmethod
    def query_graph(sources: list['Layer'], sink: 'Layer') -> list['Layer']:
        q = deque(sources)
        seen: set[Layer] = set()
        path: list[Layer] = []
        while q:
            node = q.popleft()
            if node in seen:
                continue
            
            if any(
                p not in seen and node not in sources
                for p in Layer.PARENT_GRAPH[node].values()
            ):
                q.insert(1, node)
                continue

            seen.add(node)
            path.append(node)
            
            if node == sink:
                break

            for n in Layer.CHILD_GRAPH[node].values():
                q.append(n)

        return path