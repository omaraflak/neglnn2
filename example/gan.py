import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

from neglnn.layers.dense import Dense
from neglnn.layers.model import Model
from neglnn.layers.dropout import Dropout
from neglnn.layers.scalar import Scalar
from neglnn.activations.leaky_relu import LeakyRelu
from neglnn.activations.sigmoid import Sigmoid
from neglnn.losses.binary_cross_entropy import BinaryCrossEntropy
from neglnn.optimizers.adam import Adam
from neglnn.initializers.he_normal import HeNormal
from neglnn.network.network import Network

# load mnist dataset
(x_train, y_train), _ = mnist.load_data()
x_train = x_train.astype('float32') / 255
x_train = np.reshape(x_train, (x_train.shape[0], -1, 1))

# generator
noise_size = 100
G = Network.sequential([
    Dense((noise_size, 1), (200, 1), initializer=HeNormal(), optimizer=lambda: Adam(learning_rate=0.0002, beta_1=0.5)),
    LeakyRelu(0.2),
    Dense((200, 1), (784, 1), initializer=HeNormal(), optimizer=lambda: Adam(learning_rate=0.0002, beta_1=0.5)),
    Sigmoid()
])

# discriminator
D = Network.sequential([
    Dense((784, 1), (200, 1), initializer=HeNormal(), optimizer=lambda: Adam(learning_rate=0.0002, beta_1=0.5)),
    LeakyRelu(0.2),
    Dropout(0.4),
    Dense((200, 1), (1, 1), initializer=HeNormal(), optimizer=lambda: Adam(learning_rate=0.0002, beta_1=0.5)),
    Sigmoid(),
    Scalar()
])

# GAN
G_model = Model(G)
D_model = Model(D)
GAN = Network.sequential([G_model, D_model])

# params
loss = BinaryCrossEntropy()
epochs = 50
batch_size = 256
half_batch = batch_size // 2
batch_per_epoch = 50

# intermediate image generation
gen_count = 10
seeds = np.random.randn(gen_count, noise_size, 1)
print_fq = 2

# labels
REAL = 1
FAKE = 0
D_TRAINING_Y = half_batch * [FAKE] + half_batch * [REAL]
G_TRAINING_Y = batch_size * [REAL]

# training
for epoch in range(epochs):
    G_error, D_error = 0, 0
    for j in range(batch_per_epoch):
        # train discriminator
        fake_images = G.run_all(np.random.randn(half_batch, noise_size, 1))
        real_images = x_train[np.random.randint(x_train.shape[0], size=half_batch)]
        d_training_x = np.vstack((fake_images, real_images))
        D_error += D.fit_once(d_training_x, D_TRAINING_Y, loss, batch_size=8)

        # train generator
        g_training_x = np.random.randn(batch_size, noise_size, 1)
        D_model.trainable = False
        G_error += GAN.fit_once(g_training_x, G_TRAINING_Y, loss, batch_size=8)
        D_model.trainable = True

    G_error /= batch_per_epoch
    D_error /= batch_per_epoch
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