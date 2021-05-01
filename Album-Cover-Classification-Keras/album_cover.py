# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 05:29:19 2021

@author: prakh
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


image_size = (220, 220) # any other sizes are resized to this
batch_size = 4
dirname = "Album_Covers"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dirname,
    image_size=image_size,
    batch_size=batch_size,
)


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
for images, labels in train_ds.take(1):
    for i in range(2):
        ax = plt.subplot(1, 2, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
"""
input_shape = (*image_size, 3) # specify 3 colour channels
model = keras.models.Sequential([
    keras.Input(shape=image_size),
    layers.experimental.preprocessing.Rescaling(1.0 / 255),   
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.GlobalAveragePooling2D(), # convert each neuron's output to a scalar
    layers.Dense(1, activation="sigmoid"),
])"""
    
    
input_shape = (*image_size, 3) # specify 3 colour channels
model = keras.models.Sequential([
    keras.Input(shape=input_shape),
    layers.experimental.preprocessing.Rescaling(1.0 / 255),
    layers.Conv2D(5, (3, 3), strides=2, padding="same"), # 5 neurons, 3x3 kernel
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(5, (3, 3), strides=2, padding="same"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.5),
    layers.GlobalAveragePooling2D(), # convert each neuron's output to a scalar
    layers.Dense(1, activation="sigmoid") # classification
])
    
model.summary()

keras.utils.plot_model(model, show_shapes=True)


epochs = 5

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds, epochs=epochs
)


for x, y in train_ds.take(1): # take 1 batch (4 images with their labels)
    predictions = model.predict(x)
    score = predictions
    print("score", score)
    print("y", y)