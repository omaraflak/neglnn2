import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

from neglnn.layers.dense import Dense
from neglnn.layers.reshape import Reshape
from neglnn.activations.tanh import Tanh
from neglnn.losses.mse import MSE
from neglnn.initializers.xavier import Xavier
from neglnn.optimizers.momentum import Momentum
from neglnn.network.network import Network, BlockBuilder

def load_data(limit: int):
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32')
    x_train /= 255

    x_test = x_test.astype('float32')
    x_test /= 255

    return x_train[:limit], x_test

network = Network.create([
    BlockBuilder(Reshape((28, 28), (1, 784))),
    BlockBuilder(Dense(784, 30), Xavier(), lambda: Momentum(0.1)),
    BlockBuilder(Tanh()),
    BlockBuilder(Dense(30, 16), Xavier(), lambda: Momentum(0.1)),
    BlockBuilder(Tanh()),
    BlockBuilder(Dense(16, 30), Xavier(), lambda: Momentum(0.1)),
    BlockBuilder(Tanh()),
    BlockBuilder(Dense(30, 784), Xavier(), lambda: Momentum(0.1)),
    BlockBuilder(Reshape((1, 784), (28, 28)))
])

x_train, x_test = load_data(1000)

network.fit(x_train, x_train, MSE(), 50)

encoder = network[:5]
decoder = network[5:]

_, ax = plt.subplots(5, 3)
for i in range(5):
    code = encoder.run(x_test[i])
    reconstructed = decoder.run(code)
    ax[i][0].imshow(x_test[i], cmap='gray')
    ax[i][1].imshow(np.reshape(code, (4, 4)), cmap='gray')
    ax[i][2].imshow(reconstructed, cmap='gray')
plt.show()