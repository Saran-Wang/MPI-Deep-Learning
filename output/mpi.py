import time
import copy
from tqdm import tqdm
import numpy as np
from mpi4py import MPI
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

very_start = time.time()

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

start_time = time.time()
# load dataset
(trainX, trainY), (testX, testY) = mnist.load_data()
# reshape dataset
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))
# convert labels to categorical variables
trainY = to_categorical(trainY)
testY = to_categorical(testY)
# Normalize the data
trainX = trainX.astype('float32') / 255.0
testX = testX.astype('float32') / 255.0

# split the dataset to different subsets to each process
trainx = np.array_split(trainX,size)[rank]
trainy = np.array_split(trainY,size)[rank]

batch_size = 128
train_dataset = tf.data.Dataset.from_tensor_slices((trainx, trainy))
train_dataset = train_dataset.batch(batch_size)

# establish model
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

# Instantiate an optimizer.
optimizer = keras.optimizers.Adam(learning_rate=1e-3*size)
# Instantiate a loss function.
loss_fn = keras.losses.CategoricalCrossentropy()
# Prepare the metrics.
train_acc_metric = keras.metrics.CategoricalAccuracy()

print("here")
comm.Barrier()
if rank == 0:
  print("Time taken to read file: %.2fs" % (time.time() - start_time))

#train
epochs = 5
for epoch in range(epochs):
    if rank == 0:
      print("\nStart of epoch %d" % (epoch,))
      start_time = time.time()
    loss_epoch = 0
    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in tqdm(enumerate(train_dataset)):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)
        loss_epoch = loss_epoch + loss_value
        if rank == 0 and step >= 10 and step % 10 == 0:
          grad_time = time.time()
        grads = tape.gradient(loss_value, model.trainable_weights)
        if rank == 0 and step >= 10 and step % 10 == 0:
          print("At step %d, Update gradient for each batch taken: %.2fs" % (step, time.time() - grad_time))

        grads_np = np.array(grads, dtype=object)


        # sum up all gradients across all ranks
        comm.Barrier()
        grads_sum = comm.reduce(grads_np, op=MPI.SUM, root=0)

        if rank == 0 :
          grads_sum = grads_sum/size
          
        # distribute the average gradients to all ranks
        comm.Barrier()
        grads = comm.bcast(grads_sum, root=0)
        
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Update training metric.
        train_acc_metric.update_state(y_batch_train, logits)
    
    # Display metrics at the end of each epoch.
    loss_epoch = loss_epoch/len(train_dataset)
    train_acc = train_acc_metric.result()

    comm.Barrier()
    if rank == 0:
      print("Time taken: %.2fs" % (time.time() - start_time)) # print out the time of each epoch (# of size processes)

    loss_epoch_sum = comm.reduce(loss_epoch, op=MPI.SUM, root=0)
    train_acc_sum = comm.reduce(train_acc, op=MPI.SUM, root=0)
    

    if rank == 0:
      loss_epoch_avg = loss_epoch_sum/size
      train_acc_avg = train_acc_sum/size
      print("Training acc over epoch: %.4f, Training loss (for one epoch) at step: %.4f" % (float(train_acc_avg), float(loss_epoch_avg)))

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()
    comm.Barrier()

# Test result
if rank == 0:
  logits = model(testX, training=False)
  loss_value = loss_fn(testY, logits)
  train_acc_metric.update_state(testY, logits)
  print("Test loss:", float(loss_value))
  print("Test accuracy:", float(train_acc_metric.result()))
  print("Total run time: %.2fs" % (time.time() - very_start))


MPI.Finalize()
