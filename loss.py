import tensorflow as tf

previous_prediction = None
previous_true_label = None

def directed_mse_loss(y_true, y_pred):
    """
    Apply asymmetric penalties for over- and under-predictions
    by incorporating prediction direction into the loss function.
    """
    penalty = tf.where(y_pred > y_true, 13.0, 10.0)
    squared_error = tf.square(y_pred - y_true)
    return tf.reduce_mean(penalty * squared_error, axis=-1)




