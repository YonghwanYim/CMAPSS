import pickle
import numpy as np

# use same environment.
# 알고리즘과는 분리해야 함. value-based, policy-based 모두 적용할 수 있도록.

class Environment:
    def __init__(self, data):
        self.environment = data
        self.max_unit_number = self.environment['unit_number'].max()
        self.states = self.environment[['s_{}'.format(i) for i in range(1, 22)]] # if dummy column (s_0)? -> "for i in range(0, 22)"
        self.lr_states = self.environment[['s_{}'.format(i) for i in range(0, 22)]]

    def nextStateIndex(self, action, current_index):
        # action에 따라 next state가 달라짐.
        if action == 'continue':
            return current_index + 1
        else:
            # 이 부분을 수정해서 마지막 engine에서 replacement 시 out of index 문제를 해결할 수 있음 (continue인 경우도 마찬가지)
            # replace인 경우 next_state는 다음 엔진(unit_number)의 첫번째 행.
            current_unit_number = self.environment['unit_number'].iloc[current_index]
            next_unit_initial_index = self.environment[self.environment['unit_number'] == current_unit_number].index[-1] + 1
            return next_unit_initial_index

    def stateMinIndex(self, current_index):
        # 사전에 정의된 stopping time까지 continue할 때 기준점으로 잡기 위해 current_index에 해당되는 unit의 처음 index 반환.
        # 이 min index에 해당 unit의 t_replace 값만큼을 continue.
        current_unit_number = self.environment['unit_number'].iloc[current_index]
        return self.environment[self.environment['unit_number'] == current_unit_number].index[0]

    def get_states(self):
        return self.states


class Agent:
    def __init__(self, actions=["continue", "replace"]):
        self.actions = actions
        # Linear Function Approximation (value-based)
        self.weights = {action: np.random.normal(loc=0, scale=0.5, size=21) for action in actions}
        self.best_weights = {action: np.random.normal(loc=0, scale=0.5, size=21) for action in actions} # for save best weight
        self.lr_weights_by_td = (np.random.normal(loc=0, scale=0.5, size=22))
        self.lr_best_weights = {action: np.random.normal(loc=0, scale=0.5, size=22) for action in actions} # 상수항까지 포함해서 22.

    def get_weights(self):
        return self.weights

    def get_best_weights(self):
        return self.best_weights

    def save_weights(self, action, weights):
        #self.weights[action] = weights[action]
        if action in self.weights:
            self.weights[action] = weights
        else:
            print(f"Error: '{action}' is not a vaild action.")

    def save_best_weights(self, best_weights):
        self.best_weights = best_weights

    def update_lr_weights_by_gradient(self, gradient, learning_rate):
        self.lr_weights_by_td = self.lr_weights_by_td - learning_rate * gradient



class Rewards:
    def __init__(self, r_continue, r_continue_but_failure, r_replace, r_actual_continue, r_actual_failure, r_actual_replace):
        self.r_continue = -r_continue                             # cost (+) -> reward (-)
        self.r_continue_but_failure = -(r_continue_but_failure)   # cost (+) -> reward (-)
        self.r_replace = -(r_replace)                             # cost (+) -> reward (-)
        self.r_actual_continue = r_actual_continue                # reward -> reward (부호 변경하지 않아도 됨)
        self.r_actual_failure = r_actual_failure
        self.r_actual_replace = r_actual_replace

    def get_reward(self, current_index, next_index, action, environment):
        current_unit_number = environment['unit_number'].iloc[current_index]
        next_unit_number = environment['unit_number'].iloc[next_index]

        if action == 'continue':
            if current_unit_number == next_unit_number:
                time_difference = self.calculate_time_difference(current_index, next_index, environment)
                return self.r_continue * time_difference
            else:
                return self.r_continue_but_failure
        elif action == 'replace':
            return self.r_replace

    def get_actual_reward(self, current_index, next_index, action, environment):
        current_unit_number = environment['unit_number'].iloc[current_index]
        next_unit_number = environment['unit_number'].iloc[next_index]

        if action == 'continue':
            if current_unit_number == next_unit_number:
                time_difference = self.calculate_time_difference(current_index, next_index, environment)
                return self.r_actual_continue * time_difference
            else:
                return self.r_actual_failure
        elif action == 'replace':
            return self.r_actual_replace

    def calculate_time_difference(self, current_index, next_index, environment):
        current_time_cycles = environment['time_cycles'].iloc[current_index]
        next_time_cycles = environment['time_cycles'].iloc[next_index]
        return next_time_cycles - current_time_cycles

    """
    def get_reward(self, current_index, next_index, action, environment):
        current_unit_number = environment['unit_number'].iloc[current_index]
        next_unit_number = environment['unit_number'].iloc[next_index]

        if action == 'continue':
            return self.r_continue if current_unit_number == next_unit_number else self.r_continue_but_failure
        elif action == 'replace':
            return self.r_replace

    def get_actual_reward(self, current_index, next_index, action, environment):
        current_unit_number = environment['unit_number'].iloc[current_index]
        next_unit_number = environment['unit_number'].iloc[next_index]

        if action == 'continue':
            return self.r_actual_continue if current_unit_number == next_unit_number else self.r_actual_failure
        elif action == 'replace':
            return self.r_actual_replace
    """