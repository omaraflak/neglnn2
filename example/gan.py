import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

from neglnn.layers.dense import Dense
from neglnn.activations.leaky_relu import LeakyRelu
from neglnn.activations.sigmoid import Sigmoid
from neglnn.losses.binary_cross_entropy import BinaryCrossEntropy
from neglnn.optimizers.adam import Adam
from neglnn.initializers.he_normal import HeNormal
from neglnn.network.network import Network

# load mnist dataset
samples_per_class = 100
(x_train, y_train), _ = mnist.load_data()
x_train = x_train.astype('float32') / 255
x_train = x_train[:samples_per_class * 10]
x_train = np.reshape(x_train, (samples_per_class * 10, -1, 1))

# generator model
noise_size = 100
G = Network.sequential([
    Dense(noise_size, 200, initializer=HeNormal(), optimizer=lambda: Adam(beta_1=0.5)),
    LeakyRelu(0.2),
    Dense(200, 784, initializer=HeNormal(), optimizer=lambda: Adam(beta_1=0.5)),
    Sigmoid()
])

# discriminator
D = Network.sequential([
    Dense(784, 200, initializer=HeNormal(), optimizer=lambda: Adam(beta_1=0.5)),
    LeakyRelu(0.2),
    Dense(200, 1, initializer=HeNormal(), optimizer=lambda: Adam(beta_1=0.5)),
    Sigmoid()
])

# params
loss = BinaryCrossEntropy()
epochs = 150
batch_size = 8

# intermediate generation to create GIF
gen_count = 10
seeds = np.random.randn(gen_count, noise_size, 1)
print_fq = 2

# labels
REAL = np.array([[1]])
FAKE = np.array([[0]])

# training
G_errors = []
D_errors = []
for epoch in range(epochs):
    G_error, D_error = 0, 0
    for index, real_image in enumerate(x_train):
        # generate image
        noise = np.random.randn(noise_size, 1)
        fake_image = G.run(noise)

        # discriminate real image + backward
        real_predict = D.run(real_image)
        soft_real = np.random.uniform(0.9, 1)
        D.record_gradient(loss.prime(soft_real, real_predict), optimize=False)

        # discriminate fake image + backward
        fake_predict = D.run(fake_image)
        dEDdDG = loss.prime(FAKE, fake_predict)
        dEDdG = D.record_gradient(dEDdDG, optimize=False)

        # backward generator
        dDGdG = dEDdG / dEDdDG
        dEGdDG = loss.prime(REAL, fake_predict)
        G.record_gradient(dEGdDG * dDGdG, optimize=False)

        G_error += loss.call(REAL, fake_predict)
        D_error += loss.call(soft_real, real_predict) + loss.call(FAKE, fake_predict)

        if index % batch_size == 0:
            G.optimize()
            D.optimize()

    G_error /= len(x_train)
    D_error /= len(x_train)
    G_errors.append(G_error)
    D_errors.append(D_error)
    print('%d/%d, g_error=%f, d_error=%f' % (epoch + 1, epochs, G_error, D_error))

    if (epoch + 1) % print_fq == 0:
        plt.figure(figsize=(15, 5))
        for i, seed in enumerate(seeds):
            gen = G.run(seed) * 255
            image = np.reshape(gen, (28, 28))
            plt.subplot(1, gen_count, i + 1)
            plt.imshow(image, cmap='binary')
        plt.savefig('images/epoch_%d.png' % (epoch + 1))
        plt.close()

G.save('generator.bin')
D.save('discriminator.bin')