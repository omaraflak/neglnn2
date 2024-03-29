import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

from neglnn.layers.dense import Dense
from neglnn.layers.reshape import Reshape
from neglnn.layers.model import Model
from neglnn.activations.tanh import Tanh
from neglnn.losses.mse import MSE
from neglnn.initializers.xavier_normal import XavierNormal
from neglnn.optimizers.momentum import Momentum
from neglnn.network.network import Network

def load_data(limit: int):
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    return x_train[:limit], x_test

encoder = Network.sequential([
    Reshape((28, 28), (28 * 28, 1)),
    Dense((28 * 28, 1), (30, 1), initializer=XavierNormal(), optimizer=lambda: Momentum(0.1)),
    Tanh(),
    Dense((30, 1), (16, 1), initializer=XavierNormal(), optimizer=lambda: Momentum(0.1)),
    Tanh()
])

decoder = Network.sequential([
    Dense((16, 1), (30, 1), initializer=XavierNormal(), optimizer=lambda: Momentum(0.1)),
    Tanh(),
    Dense((30, 1), (28 * 28, 1), initializer=XavierNormal(), optimizer=lambda: Momentum(0.1)),
    Tanh(),
    Reshape((28 * 28, 1), (28, 28))
])

network = Network.sequential([Model(encoder), Model(decoder)])

x_train, x_test = load_data(1000)
network.fit(x_train, x_train, MSE(), 50)

f, ax = plt.subplots(5, 3)
for i in range(5):
    code = encoder.run(x_test[i])
    reconstructed = decoder.run(code)
    ax[i][0].imshow(x_test[i], cmap='gray')
    ax[i][1].imshow(np.reshape(code, (4, 4)), cmap='gray')
    ax[i][2].imshow(reconstructed, cmap='gray')
plt.show()