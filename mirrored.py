import argparse
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# load dataset
(trainX, trainY), (testX, testY) = mnist.load_data()
# convert labels to categorical variables
trainY = to_categorical(trainY)
testY = to_categorical(testY)

# reshape dataset
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))
# Normalize the data
trainX = trainX.astype('float32') / 255.0
testX = testX.astype('float32') / 255.0

batch_size = 128
epochs = 5
parser = argparse.ArgumentParser()
parser.add_argument(' -- n_gpus', type=int, default=1)
args = parser.parse_args()
n_gpus = args.n_gpus

device_type = 'GPU'
devices = tf.config.experimental.list_physical_devices(
          device_type)
devices_names = [d.name.split("e:")[1] for d in devices]
strategy = tf.distribute.MirroredStrategy(
           devices=devices_names[:n_gpus])

with strategy.scope():
  model = models.Sequential(
        [
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(256, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(1000, activation='relu'),
            layers.Dense(100, activation='relu'),
            layers.Dense(10, activation="softmax"),
        ]
    )
  model.compile(loss="categorical_crossentropy", 
                   optimizer="adam", metrics=["accuracy"])

model.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, validation_split=0.1)

