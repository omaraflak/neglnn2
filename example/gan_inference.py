import numpy as np
import matplotlib.pyplot as plt
from neglnn.network.network import Network

generator = Network.load('generator.bin')
discriminator = Network.load('discriminator.bin')
noise_size = 100

plt.figure(figsize=(25, 30))
for i in range(10):
    for j in range(10):
        seed = np.random.randn(noise_size, 1)
        generated = generator.run(seed)
        prediction = discriminator.run(generated) > 0.5
        image = np.reshape(generated, (28, 28))
        plt.subplot(10, 10, i * 10 + j + 1, title=str(prediction))
        plt.imshow(image, cmap='binary')

plt.savefig('inference.jpeg')