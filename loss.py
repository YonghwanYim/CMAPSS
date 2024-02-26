# Directed Mean Squared Error를 고려한 Loss function
import tensorflow as tf
def directed_mse_loss(y_true, y_pred):
    penalty = tf.where(y_pred > y_true, 13.0, 10.0)
    squared_error = tf.square(y_pred - y_true)
    return tf.reduce_mean(penalty * squared_error, axis=-1)