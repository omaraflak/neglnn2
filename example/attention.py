import numpy as np
from neglnn.layers.dense import Dense
from neglnn.layers.transpose import Transpose
from neglnn.layers.dot import Dot
from neglnn.layers.input import Input
from neglnn.activations.softmax_row import SoftmaxRow
from neglnn.losses.mse import MSE
from neglnn.initializers.random_uniform import RandomUniform
from neglnn.optimizers.momentum import Momentum
from neglnn.network.network import Network
from neglnn.network.graph import Graph

x_train = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
y_train = np.reshape([[1, 0], [1, 1], [0.5, 1], [0.1, 0]], (4, 2, 1))

n = 2
d = 1
p = 1

input_layer = Input()
q_layer = Dense((n, 1), (n, d), initializer=RandomUniform(), optimizer=lambda: Momentum())
k_layer = Dense((n, 1), (n, d), initializer=RandomUniform(), optimizer=lambda: Momentum())
v_layer = Dense((n, 1), (n, p), initializer=RandomUniform(), optimizer=lambda: Momentum())
k_t_layer = Transpose()
dot_qkt_layer = Dot()
soft_layer = SoftmaxRow(n)
dot_sv_layer = Dot()

graph = Graph()
graph.connect(input_layer, q_layer)
graph.connect(input_layer, k_layer)
graph.connect(input_layer, v_layer)
graph.connect(k_layer, k_t_layer)
graph.connect(q_layer, dot_qkt_layer, 'a')
graph.connect(k_t_layer, dot_qkt_layer, 'b')
graph.connect(dot_qkt_layer, soft_layer)
graph.connect(soft_layer, dot_sv_layer, 'a')
graph.connect(v_layer, dot_sv_layer, 'b')

network = Network(graph)
network.fit(x_train, y_train, MSE(), 1000)
print(network.run_all(x_train))