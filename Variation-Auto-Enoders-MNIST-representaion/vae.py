# Variational Auto-Encoder in Keras
#
# This is a simple VAE for MNIST digits. It does not use convolutions,
# so we can focus on the VAE aspect.
#
# This code is mostly from Chollet:
# https://keras.io/getting_started/intro_to_keras_for_researchers/#endtoend-experiment-example-1-variational-autoencoders
#
# However, it uses the approach to creating the dual loss from:
# https://gist.github.com/tik0/6aa42cabb9cf9e21567c3deb309107b7

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K

original_dim = 784
intermediate_dim = 64
latent_dim = 2

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        # override the inherited .call(self, inputs) method
        z_mean, z_log_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# Define encoder model:
# input -> hidden layer -> (z_mean, z_log_var) -> (sampling) z
original_inputs = tf.keras.Input(shape=(original_dim,), name="encoder_input")
x = layers.Dense(intermediate_dim, activation="relu")(original_inputs)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()((z_mean, z_log_var))
encoder = tf.keras.Model(inputs=original_inputs, outputs=z, name="encoder")

# Define decoder model:
# z -> hidden layer -> output
latent_inputs = tf.keras.Input(shape=(latent_dim,), name="z_sampling")
x = layers.Dense(intermediate_dim, activation="relu")(latent_inputs)
outputs = layers.Dense(original_dim, activation="sigmoid")(x)
decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name="decoder")

# Define VAE model.
outputs = decoder(z)
vae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name="vae")

# Loss approach from
# https://gist.github.com/tik0/6aa42cabb9cf9e21567c3deb309107b7

reconstruction_loss = mse(original_inputs, outputs)
reconstruction_loss = original_dim * K.mean(reconstruction_loss)
kl_loss = -0.5 * tf.reduce_mean(
    z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)

# Set up our losses on the model, and create them as metrics too.
# The Model's loss is the sum of the two losses.
vae.add_loss(kl_loss)
vae.add_metric(kl_loss, name='kl_loss', aggregation='mean')
vae.add_loss(reconstruction_loss)
vae.add_metric(reconstruction_loss, name='mse_loss', aggregation='mean')

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Prepare a dataset. We'll ignore the test data.
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    x_train.reshape(60000, 784).astype("float32") / 255)
# Use x_train as both inputs & targets
dataset = dataset.map(lambda x: (x, x))  
dataset = dataset.shuffle(buffer_size=1024).batch(32)

# Compile. Don't need to specify loss here, as we already ran add_loss
# (in fact twice).
vae.compile(optimizer)

# Train.
vae.fit(dataset, epochs=15)

# Save. We will save the VAE (which includes both parts) and each part
# separately (so we can call them separately when visualising later).
vae.save("vae.saved_model")
encoder.save("vae_encoder.saved_model")
decoder.save("vae_decoder.saved_model")


vae.save("vae.saved_model")
print("Saved model to disk")