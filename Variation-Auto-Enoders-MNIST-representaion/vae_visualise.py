# Plot two visualisations of the latent space of an auto-encoder.
#
# It uses the plotting code from here:
# https://keras.io/examples/generative/vae/

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


"""
## Display a grid of sampled digits
"""

def plot_latent_space(n=30, figsize=15):
    # display a n*n 2D manifold of digits
    digit_size = 28
    # look at scatterplot: the training data is distributed in [-3,
    # 3]^2, or so.
    scale = 3.0 
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys")
    plt.show()



"""
## Display how the latent space clusters different digit classes
"""


def plot_label_clusters(data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean = encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels,
                cmap="tab10") # categorical 10-value colourmap
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train[0:2560]
y_train = y_train[0:2560] # this is enough for scatterplot

x_train = x_train.reshape(-1, 784)

x_train = np.expand_dims(x_train, -1).astype("float32") / 255

vae = keras.models.load_model("vae.saved_model")
encoder = keras.models.load_model("vae_encoder.saved_model")
decoder = keras.models.load_model("vae_decoder.saved_model")

#plot_latent_space()
plot_label_clusters(x_train, y_train)

