from typing import Callable, Optional
from neglnn.initializers.initializer import Initializer
from neglnn.optimizers.optimizer import Optimizer, Update
from neglnn.utils.types import Array, Shape, InputKey
from neglnn.utils.identifiable import Identifiable

"""
Base Layer class.
"""
class Layer(Identifiable):
    def __init__(
        self,
        input_shape: Optional[Shape] = None,
        output_shape: Optional[Shape] = None,
        initializer: Optional[Initializer] = None,
        optimizer: Optional[Callable[[], Optimizer]] = None
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.initializer = initializer
        self.optimizer = optimizer
        if self.initializer:
            self.initializer.on_target_layer(self)

    @property
    def trainable(self) -> bool:
        return self.initializer and self.optimizer

    def initialize_parameters(self):
        """
        Initialize the trainable parameters here using the `initializer`
        """
        raise NotImplementedError

    def setup_optimizers(self):
        self.optimizers: list[Optimizer] = []
        for parameter in self.parameters():
            optimizer = self.optimizer()
            optimizer.on_target_shape(parameter.shape)
            self.optimizers.append(optimizer)

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

    def __hash__(self) -> int:
        return hash(self.uid)