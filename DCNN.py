# For Deep Convolution Neural Networks (DCNN)

import tensorflow as tf
from tensorflow.keras import layers, models, initializers
from tensorflow.keras.callbacks import LearningRateScheduler

# hyperparameter #######################################
N_tw = 30  # Time sequence length
N_ft = 14  # Number of features
FN = 10    # Number of filters (each convolution layer)
FL = 10    # Filter size (FL * 1)
batch_size = 512
neurons_fc = 100  # Neurons in fully-connected layer
dropout_rate = 0.5
epochs = 250

# define model
model = models.Sequential()

# Xavier normal initializer
initializer = initializers.GlorotUniform()

# 1D Convolution layers
model.add(layers.Conv1D(filters=FN, kernel_size=FL, padding='same', activation='tanh', kernel_initializer=initializer, input_shape=(N_tw, N_ft)))
model.add(layers.Conv1D(filters=FN, kernel_size=FL, padding='same', activation='tanh', kernel_initializer=initializer))
model.add(layers.Conv1D(filters=FN, kernel_size=FL, padding='same', activation='tanh', kernel_initializer=initializer))
model.add(layers.Conv1D(filters=FN, kernel_size=FL, padding='same', activation='tanh', kernel_initializer=initializer))

# High-level representation using another Conv1D layer
model.add(layers.Conv1D(filters=1, kernel_size=3, padding='same', activation='tanh'))
# Flatten the 2D feature map
model.add(layers.Flatten())
# Dropout layer
model.add(layers.Dropout(dropout_rate))
# Fully-connected layer
model.add(layers.Dense(neurons_fc, activation='tanh', kernel_initializer=initializer))
# Output layer for RUL estimation
model.add(layers.Dense(1, kernel_initializer=initializer))

# Learning rate scheduler function
def lr_schedule(epoch, lr):
    if epoch < 200:   # The learning rate is 0.001 for fast optimization
        return 0.001
    else:             # The learning rate of 0.0001 is used afterwards for stable convergence.
        return 0.0001

# Learning rate scheduler callback
lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)

# compile
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error')

model.summary()

# x_train은 N × 30 × 14 형태, y_train은 N 형태로 제공됩니다.
# x_train.shape == (N, 30, 14)
# y_train.shape == (N, )
# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[lr_scheduler])

