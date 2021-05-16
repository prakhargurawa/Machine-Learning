# Some parts from vae.py, some from
# https://gist.github.com/s33a11ev1l/7917feed3dd59bb3bc2efcdaa02117a3
#
# Notice: relative to vae.py, we increase the latent dim from 2 to
# 100, greatly increase the capacity of the model, add LeakyReLU and
# BatchNorm, use Adam with low learning rate and momentum decay, and
# increase training time.

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, BatchNormalization,LeakyReLU
#from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

img_shape = (28 * 28,)
z_dim = 100



# Define generator model (like AE decoder)
# z -> hidden layer -> image
G = Sequential()
G.add(Input(shape=(z_dim,)))
G.add(Dense(256))
G.add(LeakyReLU(alpha=0.2))
G.add(BatchNormalization(momentum=0.8))
G.add(Dense(512))
G.add(LeakyReLU(alpha=0.2))
G.add(BatchNormalization(momentum=0.8))
G.add(Dense(1024))
G.add(LeakyReLU(alpha=0.2))
G.add(BatchNormalization(momentum=0.8))
G.add(Dense(np.prod(img_shape), activation="tanh"))
G.summary()

# Define discriminator model
# image -> hidden layer -> class 0 or 1 (real or fake; note, not digits)
D = Sequential()
D.add(Input(shape=(img_shape)))
D.add(Dense(512))
D.add(LeakyReLU(alpha=0.2))
D.add(Dense(256))
D.add(LeakyReLU(alpha=0.2))
D.add(Dense(1, activation='sigmoid')) # binary classification
D.summary()
# Adam with relatively low learning rate and beta_1 (momentum decay)
D.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5),
          metrics=['accuracy'])

# Set discriminator not trainable, create GAN and optimizer for GAN
# which will train generator only, and compile
D.trainable = False
z = Input(shape=(z_dim,))
GAN = Model(z, D(G(z)))
GAN.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5),
          metrics=['accuracy'])
GAN.summary()

# Prepare a dataset. Discard the labels and test data. We don't create
# a tf Dataset as it's convenient for us to access the training data
# directly.
(x_train, _), (_, _) = mnist.load_data()
x_train = x_train / 127.5 - 1. # map to range [-1, 1]: suitable for tanh
x_train = x_train.reshape(x_train.shape[0], 28 * 28)
batch_size=100
epochs = 10
steps = epochs * x_train.shape[0] // batch_size
sample_interval=100
# fake -> 1, real -> 0
y_fake = np.ones((batch_size, 1))
y_real = np.zeros((batch_size, 1))

def sample_images(epoch):
    # save an image to see some of our fake images
    r, c = 5, 5
    z = np.random.normal(0, 1, (r * c, z_dim))
    x_fake = G.predict(z)

    fig, axs = plt.subplots(r, c)
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(x_fake[i*r+j].reshape((28, 28)), cmap='gray')
            axs[i, j].axis('off')
    print("saving")
    plt.tight_layout()
    plt.savefig("gan_mnist.png")
    plt.close()
    

# after about 500 steps we can see it is training "in the right
# direction". Results are ok after 3000 steps. The full run takes ~20m
for step in range(steps):

    # choose real images, x_real, and make fake images, x_fake
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    x_real = x_train[idx]
    z = np.random.normal(0, 1, (batch_size, z_dim))
    x_fake = G.predict(z)

    # train D
    loss_d_real = D.train_on_batch(x_real, y_real)
    loss_d_fake = D.train_on_batch(x_fake, y_fake)
    loss_d = 0.5 * np.add(loss_d_real, loss_d_fake)

    # train G, by training GAN
    # pass y_real, ie pretend these images are real
    loss_g = GAN.train_on_batch(z, y_real)

    # notice loss_d and loss_g are each really (loss, accuracy),
    # because we requested accuracy as a metric.
    print(f"{step:4d}/{steps} [D {loss_d[0]:.2f}, acc. (on x_real and x_fake) {100 * loss_d[1]:3.0f}] [G {loss_g[0]:.2f}, acc. (on x_fake) {100 * loss_g[1]:3.0f}]")
    if step % sample_interval == 0 or step == steps - 1:
        sample_images(step)


