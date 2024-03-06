# Directed Mean Squared Error를 고려한 Loss function
import tensorflow as tf

previous_prediction = None
previous_true_label = None

def directed_mse_loss(y_true, y_pred):
    penalty = tf.where(y_pred > y_true, 13.0, 10.0)
    squared_error = tf.square(y_pred - y_true)
    return tf.reduce_mean(penalty * squared_error, axis=-1)


# 이 함수를 수정하던, run_cpu.py를 수정하던 해야함. 우선 gradient를 구할 수 없어서 에러가 발생함.
# different_td_loss를 수정하는 것보다 RL 환경 구축이 우선이니 RL부터 완성하고 시간 남으면 수정하자.

def different_td_loss(y_true, y_pred, td_alpha):
    global previous_prediction
    global previous_true_label

    if previous_prediction is None or previous_true_label is None:
        # 이전 예측값 또는 이전 실제 값이 없으면 loss를 계산하지 않음
        return 0.0
    # 매 step이 아니라, 애초에 previous_prediction, TD 두 인자에 대한 column을 한번에 구해두고 반영?
    # 그렇게하면 학습이 되려나? 좀 더 고민해보자. 급한건 아님.

    squared_error_current = tf.square(y_pred - y_true)
    td = previous_true_label - y_true

    squared_error_td = tf.square(y_pred - (previous_prediction - td))
    diff_td_loss = tf.reduce_mean(squared_error_current + td_alpha * squared_error_td)

    return diff_td_loss



