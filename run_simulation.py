# General lib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import configparser
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import pickle        # it is suitable for saving Python objects.
import warnings
from copy import copy
import random

# Custom .py
from linear_regression_TD import Linear_Regression_TD
from simulation_env import SimulationEnvironment
from loss import directed_mse_loss
from loss import different_td_loss
from loss import previous_prediction, previous_true_label  # use global variable
from loss_td import DecisionAwareTD
from RL_component import Environment
from RL_component import Rewards
from RL_component import Agent

# Filter out the warning
warnings.filterwarnings("ignore", message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*")

# generate configuration instance
config = configparser.ConfigParser()

# a list to store dataframes containing simulation results for each sample
full_by_loss_dfs_list = []
average_by_loss_dfs = []

# For reinforcement learning
average_rewards = []
average_actual_rewards = []
training_loss = []
lr_td_training_loss = []

test_average_rewards = []
test_average_actual_rewards = []      # for average performance.
test_average_usage_times = []
test_replace_failures = []


class RunSimulation():
    def __init__(self, config_path):
        config.read(config_path)
        # number of dataset (1, 2, 3, 4)
        self.num_dataset = int(config['SimulationSettings']['num_dataset'])
        self.loss_labels = eval(config['SimulationSettings']['loss_labels'])  # 문자열 리스트를 리스트로 변환
        # The boundary value of unit numbers for dividing the train and valid datasets
        self.split_unit_number = int(config['SimulationSettings']['split_unit_number'])
        # Number of sample datasets
        self.num_sample_datasets = int(config['SimulationSettings']['num_sample_datasets'])
        # Randomly extract only a subset of observational probability data from the entire dataset
        self.observation_probability = float(config['SimulationSettings']['observation_probability'])

        # Constant for crucial_moment loss
        self.crucial_moment = int(config['SimulationSettings']['crucial_moment'])
        self.td_crucial_moment = int(config['SimulationSettings']['td_crucial_moment'])
        self.directed_crucial_moment = int(config['SimulationSettings']['directed_crucial_moment'])

        # Learning rate for different td loss() (loss.py)
        self.td_alpha = float(config['SimulationSettings']['td_alpha'])
        self.td_beta = float(config['SimulationSettings']['td_beta'])
        self.td_learning_rate = float(config['SimulationSettings']['td_learning_rate'])
        self.max_epoch = int(config['SimulationSettings']['max_epoch'])
        self.td_weight = np.random.normal(loc=0, scale=0.5, size=22)
        self.td_weight_21 = np.random.normal(loc=0, scale=0.5, size=21) # RL과 weight 크기 맞춰서 비교하기 위함.

        self.td_simulation_threshold = float(config['SimulationSettings']['td_simulation_threshold'])

        # constant of simulation
        self.threshold_start = int(config['SimulationSettings']['threshold_start'])
        self.threshold_end = int(config['SimulationSettings']['threshold_end'])
        self.threshold_values = list(range(self.threshold_start, self.threshold_end + 1))

        # Define cost
        self.REPLACE_COST = int(config['SimulationSettings']['REPLACE_COST'])
        self.FAILURE_COST = int(config['SimulationSettings']['FAILURE_COST'])
        self.CONTINUE_COST = float(config['SimulationSettings']['CONTINUE_COST'])
        self.REWARD_ACTUAL_REPLACE = int(config['SimulationSettings']['REWARD_ACTUAL_REPLACE'])
        self.REWARD_ACTUAL_FAILURE = int(config['SimulationSettings']['REWARD_ACTUAL_FAILURE'])
        self.REWARD_ACTUAL_CONTINUE = int(config['SimulationSettings']['REWARD_ACTUAL_CONTINUE'])

        # Hyperparameter for reinforcement learning
        self.gamma = float(config['RL_Settings']['discount_factor'])
        self.alpha = float(config['RL_Settings']['learning_rate'])
        self.initial_epsilon = float(config['RL_Settings']['initial_epsilon'])
        self.epsilon_delta = float(config['RL_Settings']['epsilon_delta'])
        self.min_epsilon = float(config['RL_Settings']['min_epsilon'])
        self.max_episodes = int(config['RL_Settings']['max_episodes'])

        # RL
        self.columns_to_scale = ['s_1', 's_2', 's_3', 's_4', 's_5', 's_6', 's_7', 's_8', 's_9', 's_10', 's_11', 's_12',
                            's_13', 's_14', 's_15', 's_16', 's_17', 's_18', 's_19', 's_20', 's_21']
        self.best_average_reward = -10000   # 처음에만 이렇게 초기화하고, 그 다음에는 알아서 반영.

        # 사전에 정의된 stopping time에 따른 exploration을 위한 parameter
        self.min_t_replace = int(config['StoppingTime']['min_t_replace'])
        self.max_t_replace = int(config['StoppingTime']['max_t_replace']) # 10%의 데이터만 있다는 것을 감안해서 원래 값보다 1/10 수준으로 유지

        # class instance 생성
        self.env = SimulationEnvironment()
        self.agent = Agent()     # RL
        self.reward = Rewards(self.CONTINUE_COST, self.FAILURE_COST, self.REPLACE_COST, self.REWARD_ACTUAL_CONTINUE,
                              self.REWARD_ACTUAL_FAILURE, self.REWARD_ACTUAL_REPLACE)

        # dataset 분할
        self.train_data, self.valid_data, self.full_data = self.env.load_data(self.num_dataset, self.split_unit_number)
        # sampling
        self.sampled_datasets = self.env.sampling_datasets(self.num_sample_datasets, self.observation_probability,
                                                      self.train_data, self.valid_data, self.full_data)
        # sampled_datasets에 RUL column 추가.
        self.sampled_datasets_with_RUL = self.env.add_RUL_column_to_sampled_datasets(self.sampled_datasets)


    def train_lr_by_DA_TD_loss(self, data_sample_index, alpha, beta, learning_rate):
        # 전체 데이터에 대해 gradient를 한번에 구하는 method (DecisionAwareTD 클래스에서 함수 정의에 따라 달라질 수 있음).
        train_data = self.sampled_datasets_with_RUL[data_sample_index][0].copy()
        train_data[self.columns_to_scale] = train_data[self.columns_to_scale].apply(self.env.min_max_scaling, axis=0)
        train_data.reset_index(drop=True, inplace=True)  # index reset

        DA_TD = DecisionAwareTD(train_data, beta, self.td_weight, alpha)

        DA_TD.preprocessing() # 여기에 scaling 후 s_0에 1을 초기화 (LR의 상수항) 하는 코드 포함. 따라서 test시에는 s_0 추가.
        #print(self.td_weight)

        gradient = DA_TD.calculate_gradient() # prediction loss, decision loss가, alpha로 같이 들어가있음.
        #gradient = DA_TD.calculate_gradient_only_TD() # prediction accuracy term 무시하고 decision loss만 사용
        self.td_weight = self.td_weight - learning_rate * gradient
        #print(self.td_weight)

        #loss = ((self.td_weight.T.dot(DA_TD.X_t.T) - DA_TD.Y.T) ** 2).values.sum()
        #lr_td_training_loss.append(loss)

        lr_td_training_loss.append((gradient ** 2).values.sum())

    def calaulate_solution_of_lr_DA_TD_loss(self, lambd):
        # gradient를 구하지 않고 closed-form solution을 이용해 weight 계산 (24.06.26 시점에선 무의미. max가 있어 closed-form을 구할 수 없음)
        # 모든 샘플 데이터를 하나의 데이터로 합쳐서 수행
        combined_train_data = pd.DataFrame()

        for i in range(self.num_sample_datasets):
            sample_data = self.sampled_datasets_with_RUL[i][0].copy()
            combined_train_data = pd.concat([combined_train_data, sample_data], ignore_index=True)

        combined_train_data[self.columns_to_scale] = combined_train_data[self.columns_to_scale].apply(self.env.min_max_scaling, axis=0)
        combined_train_data.reset_index(drop=True, inplace=True)  # index reset

        DA_TD = DecisionAwareTD(combined_train_data, self.td_beta, self.td_weight, self.td_alpha)
        DA_TD.preprocessing()

        self.td_weight = DA_TD.calculate_closed_form_solution(lambd)
        #self.td_weight = DA_TD.calculate_closed_form_solution_ratio(lambd)

        # test
        combined_train_data.insert(loc=combined_train_data.columns.get_loc('s_1'), column='s_0', value=1)
        selected_data = combined_train_data.iloc[:, 5:27]
        print(selected_data)
        print(selected_data.dot(self.td_weight))


        # Save weights to a file using pickle
        with open('LR_TD_weight_test.pkl', 'wb') as f:
            pickle.dump(self.td_weight, f)

    def train_many_lr_by_DA_TD_loss(self):
        # TD loss의 gradient를 이용해 반복 학습. train_lr_by_DA_TD_loss 사용.
        for epoch in range(self.max_epoch):
            print(epoch + 1)
            for i in range(self.num_sample_datasets):
                # self.train_lr_by_td_loss(i, epoch, self.td_alpha, 0.001287, 0.005)
                self.train_lr_by_DA_TD_loss(i, self.td_alpha, self.td_beta, self.td_learning_rate)

        self.env.plot_training_loss(self.max_epoch, self.num_sample_datasets, lr_td_training_loss)
        print(self.td_weight)

        # Save weights to a file using pickle
        with open('LR_TD_weight_test.pkl', 'wb') as f:
            pickle.dump(self.td_weight, f)

    def train_continue_many_lr_by_DA_td_loss(self):
        # 학습된 weight을 이어서 TD loss로 학습
        with open('LR_TD_weight_test.pkl', 'rb') as f:
            self.td_weight = pickle.load(f)

        # Iterate over the number of sample datasets
        for epoch in range(self.max_epoch):
            print(epoch + 1)
            # Iterate over the number of sample datasets
            for i in range(self.num_sample_datasets):
                self.train_lr_by_DA_TD_loss(i, self.td_alpha, self.td_beta, self.td_learning_rate)

        self.env.plot_training_loss(self.max_epoch, self.num_sample_datasets, lr_td_training_loss)
        print(self.td_weight)

        # Save weights to a file using pickle
        with open('LR_TD_weight_test.pkl', 'wb') as f:
            pickle.dump(self.td_weight, f)


    # 모든 데이터가 관측 가능할 때, td_loss를 이용해 학습시킨 lr weight
    def train_lr_by_td_loss(self, data_sample_index, epoch, alpha, beta, learning_rate):
        # TD loss를 학습하는 방식을, RL에서 state를 바꿔가며 하던 방식 그대로 수행 (즉 RL 코드 이용해서 학습).
        # 이 TD loss는 random observation이 반영이 안되어있어서, 실제 time step의 차인 td가 아니라 1로 들어감.
        # 이 코드를 쓰려면 나중에 수정해야함.
        state_index = 0
        num_of_step = 0
        sum_of_gradient = 0
        loss_epoch = 0

        train_data = self.sampled_datasets_with_RUL[data_sample_index][0].copy()
        train_data[self.columns_to_scale] = train_data[self.columns_to_scale].apply(self.env.min_max_scaling, axis=0)

        # dummy row 추가 (마지막 row를 2번 복사해서 뒤에 추가)
        dummy_row = train_data.iloc[-1].copy()
        dummy_row['unit_number'] = int(dummy_row['unit_number']) + 1
        dummy_row_2 = dummy_row.copy()
        dummy_row_2['unit_number'] = int(dummy_row_2['unit_number']) + 1

        train_data = train_data._append(dummy_row)
        train_data = train_data._append(dummy_row_2)
        # float으로 변환된 자료형을 다시 int로 변환
        train_data['unit_number'] = train_data['unit_number'].astype(int)
        train_data['time_cycles'] = train_data['time_cycles'].astype(int)
        train_data['RUL'] = train_data['RUL'].astype(int)

        train_data.reset_index(drop=True, inplace=True)  # index reset (reset 하지 않으면 state 전이가 되지 않음)
        RL_env = Environment(train_data)

        for unit_num in range(RL_env.max_unit_number):  # unit num : 0, ... , (max_unit_number - 1)
            state = RL_env.states.iloc[state_index].values

            # TD loss를 위해, 엔진이 바뀌고 처음 반복은 건너뜀.
            first_iteration = True
            # 항상 continue action만 수행.
            while (state_index < RL_env.environment[
                RL_env.environment['unit_number'] == (RL_env.environment['unit_number'].max() - 2)].index[-1] + 1) \
                    and RL_env.environment['unit_number'].iloc[state_index] == (unit_num + 1):

                gradient = 0

                # TD loss 계산을 위해 엔진이 바뀌고 처음 반복은 건너뜀. (t-1이 존재하지 않으므로)
                if first_iteration:
                    first_iteration = False
                    state_index += 1  # state_index를 증가시키고 continue로 건너뜀.
                    continue

                current_state = state
                chosen_action = 'continue'

                next_state_index = RL_env.nextStateIndex(chosen_action, state_index)
                next_state = RL_env.states.iloc[next_state_index].values

                # 리워드를 통해 엔진이 바뀌는 것을 알 수 있음.
                current_reward = self.reward.get_reward(state_index, next_state_index, chosen_action,
                                                        RL_env.environment)

                WX_t = np.dot(self.agent.lr_weights_by_td, state)  # W * X_t
                #print(WX_t)
                X_diff = state - RL_env.states.iloc[state_index - 1].values  # X_t - X_{t-1}
                #print(X_diff)

                # 엔진 내의 마지막 time_step이면 td를 (1/beta)로 gradient를 업데이트.
                if current_reward == (self.reward.r_continue_but_failure):
                    gradient = 2 * (WX_t - RL_env.environment['RUL'].iloc[state_index]) * current_state + \
                               2 * alpha * (np.dot(self.agent.lr_weights_by_td, X_diff) - (1 / beta)) * X_diff
                # 마지막 time_step이 아닌 경우에는 td를 -1로 gradient update
                else :
                    gradient = 2 * (WX_t - RL_env.environment['RUL'].iloc[state_index]) * current_state + \
                               2 * alpha * (np.dot(self.agent.lr_weights_by_td, X_diff) + 1) * X_diff

                sum_of_gradient += gradient
                #print(gradient)
                #print(sum_of_gradient)

                # 다음 상태로 이동
                state_index = next_state_index
                state = RL_env.states.iloc[state_index].values
                num_of_step += 1

        mean_sum_of_gradient = sum_of_gradient / num_of_step
        #print(mean_sum_of_gradient)
        self.agent.update_lr_weights_by_gradient(mean_sum_of_gradient, learning_rate)

        lr_td_training_loss.append(np.dot(mean_sum_of_gradient, mean_sum_of_gradient))

    def train_lr_by_td_loss_random_observation(self, data_sample_index, epoch, alpha, beta, learning_rate):
        # random observation임을 고려해서 td가 실제 흘러간 타임스탭만큼 들어가는 코드.
        # Q-learning (off-policy TD(0)와 equivalent한 update)
        # 이 코드는 RL처럼 매 스탭마다 weight을 업데이트 함. (action은 계속 continue 하는 버전)
        # 2024.07.01 새롭게 정의한 td loss로 수정.
        state_index = 0
        num_of_step = 0
        sum_of_gradient = 0
        loss_epoch = 0

        train_data = self.sampled_datasets_with_RUL[data_sample_index][0].copy()
        train_data[self.columns_to_scale] = train_data[self.columns_to_scale].apply(self.env.min_max_scaling, axis=0)

        # dummy row 추가 (마지막 row를 2번 복사해서 뒤에 추가)
        dummy_row = train_data.iloc[-1].copy()
        dummy_row['unit_number'] = int(dummy_row['unit_number']) + 1
        dummy_row_2 = dummy_row.copy()
        dummy_row_2['unit_number'] = int(dummy_row_2['unit_number']) + 1

        train_data = train_data._append(dummy_row)
        train_data = train_data._append(dummy_row_2)
        # float으로 변환된 자료형을 다시 int로 변환
        train_data['unit_number'] = train_data['unit_number'].astype(int)
        train_data['time_cycles'] = train_data['time_cycles'].astype(int)
        train_data['RUL'] = train_data['RUL'].astype(int)

        # s_1 열 왼쪽에 s_0 열을 새롭게 추가하고 모든 값을 1로 초기화 (상수항에 대한 weight w_0를 학습시키기 위함)
        train_data.insert(loc=train_data.columns.get_loc('s_1'), column='s_0', value=1)

        train_data.reset_index(drop=True, inplace=True)  # index reset (reset 하지 않으면 state 전이가 되지 않음)
        RL_env = Environment(train_data)

        for unit_num in range(RL_env.max_unit_number):  # unit num : 0, ... , (max_unit_number - 1)
            # state는 22차원.
            state = RL_env.lr_states.iloc[state_index].values

            # 항상 continue action만 수행.
            while (state_index < RL_env.environment[
                RL_env.environment['unit_number'] == (RL_env.environment['unit_number'].max() - 2)].index[-1] + 1) \
                    and RL_env.environment['unit_number'].iloc[state_index] == (unit_num + 1):

                gradient = 0

                current_state = state
                chosen_action = 'continue'

                next_state_index = RL_env.nextStateIndex(chosen_action, state_index)

                # 22차원 벡터용 코드
                next_state = RL_env.lr_states.iloc[next_state_index].values # max operator 안에 들어감

                # 리워드를 통해 엔진이 바뀌는 것을 알 수 있음.
                current_reward = self.reward.get_reward(state_index, next_state_index, chosen_action,
                                                        RL_env.environment)

                WX_t = np.dot(self.agent.lr_weights_by_td, current_state)  # w * x_t
                WX_t_1 = np.dot(self.agent.lr_weights_by_td, next_state)   # w * x_{t+1}

                # t = tau_i 일 때는 다음과 같이 gradient를 업데이트 (time cycle이 엔진 내의 마지막 time cycle일 때. 즉 continue 하면 failure 하는 상태)
                if current_reward == (self.reward.r_continue_but_failure):
                    gradient = -2 * (1 - self.td_alpha) * (RL_env.environment['RUL'].iloc[
                                                               state_index] - WX_t) * current_state - 2 * self.td_alpha * (
                                           -(1 / self.td_beta) - WX_t) * current_state
                    self.agent.update_lr_weights_by_gradient(gradient, learning_rate) # gradient descent


                # t가 '1 <= t < tau_i' 인 경우에는 아래와 같이 gradient를 업데이트 (엔진 내에서 마지막 time cycle이 아닐 때)
                else:
                    time_difference = RL_env.environment['time_cycles'].iloc[next_state_index] - \
                                      RL_env.environment['time_cycles'].iloc[state_index]

                    gradient = -2 * (1 - self.td_alpha) * (RL_env.environment['RUL'].iloc[
                                                               state_index] - WX_t) * current_state - 2 * self.td_alpha * (
                                           time_difference + max(WX_t_1, 0) - WX_t) * current_state
                    self.agent.update_lr_weights_by_gradient(gradient, learning_rate) # gradient descent

                # 다음 상태로 이동
                state_index = next_state_index

                #22차원용 코드
                #state = RL_env.lr_states.iloc[state_index].values

                num_of_step += 1

                # 그냥 gradient의 크기가 얼마나 줄어드나 확인하기 위한 용도 (학습과는 상관 없음)
                sum_of_gradient += gradient

        # 학습이 잘 되고 있는지 loss를 확인하기 위한 코드 (TD loss)
        mean_sum_of_gradient = sum_of_gradient / num_of_step
        lr_td_training_loss.append(np.dot(mean_sum_of_gradient, mean_sum_of_gradient))

    def train_lr_by_td_loss_random_observation_21(self, data_sample_index, epoch, alpha, beta, learning_rate):
        # random observation임을 고려해서 td가 실제 흘러간 타임스탭만큼 들어가는 코드.
        # Q-learning (off-policy TD(0)와 equivalent한 update)
        # 이 코드는 RL처럼 매 스탭마다 weight을 업데이트 함. (action은 계속 continue 하는 버전)
        # 2024.07.01 새롭게 정의한 td loss로 수정.
        state_index = 0
        num_of_step = 0
        sum_of_gradient = 0
        loss_epoch = 0

        train_data = self.sampled_datasets_with_RUL[data_sample_index][0].copy()
        train_data[self.columns_to_scale] = train_data[self.columns_to_scale].apply(self.env.min_max_scaling, axis=0)

        # dummy row 추가 (마지막 row를 2번 복사해서 뒤에 추가)
        dummy_row = train_data.iloc[-1].copy()
        dummy_row['unit_number'] = int(dummy_row['unit_number']) + 1
        dummy_row_2 = dummy_row.copy()
        dummy_row_2['unit_number'] = int(dummy_row_2['unit_number']) + 1

        train_data = train_data._append(dummy_row)
        train_data = train_data._append(dummy_row_2)
        # float으로 변환된 자료형을 다시 int로 변환
        train_data['unit_number'] = train_data['unit_number'].astype(int)
        train_data['time_cycles'] = train_data['time_cycles'].astype(int)
        train_data['RUL'] = train_data['RUL'].astype(int)

        # s_1 열 왼쪽에 s_0 열을 새롭게 추가하고 모든 값을 1로 초기화 (상수항에 대한 weight w_0를 학습시키기 위함)
        train_data.insert(loc=train_data.columns.get_loc('s_1'), column='s_0', value=1)

        train_data.reset_index(drop=True, inplace=True)  # index reset (reset 하지 않으면 state 전이가 되지 않음)
        RL_env = Environment(train_data)

        for unit_num in range(RL_env.max_unit_number):  # unit num : 0, ... , (max_unit_number - 1)
            # state는 22차원.
            # state = RL_env.lr_states.iloc[state_index].values
            # state는 21차원 (RL과 비교를 위해 임시로).
            state = RL_env.states.iloc[state_index].values

            # 항상 continue action만 수행.
            while (state_index < RL_env.environment[
                RL_env.environment['unit_number'] == (RL_env.environment['unit_number'].max() - 2)].index[-1] + 1) \
                    and RL_env.environment['unit_number'].iloc[state_index] == (unit_num + 1):

                gradient = 0

                current_state = state
                chosen_action = 'continue'

                next_state_index = RL_env.nextStateIndex(chosen_action, state_index)

                # 22차원 벡터용 코드
                # next_state = RL_env.lr_states.iloc[next_state_index].values # max operator 안에 들어감
                # 21차원 벡터용 코드 (RL과 퍼포먼스 일치 확인용)
                next_state = RL_env.states.iloc[next_state_index].values  # max operator 안에 들어감

                # 리워드를 통해 엔진이 바뀌는 것을 알 수 있음.
                current_reward = self.reward.get_reward(state_index, next_state_index, chosen_action,
                                                        RL_env.environment)

                WX_t = np.dot(self.agent.lr_weights_by_td_21, current_state)  # w * x_t
                WX_t_1 = np.dot(self.agent.lr_weights_by_td_21, next_state)  # w * x_{t+1}

                # t = tau_i 일 때는 다음과 같이 gradient를 업데이트 (time cycle이 엔진 내의 마지막 time cycle일 때. 즉 continue 하면 failure 하는 상태)
                if current_reward == (self.reward.r_continue_but_failure):
                    gradient = -2 * (1 - self.td_alpha) * (RL_env.environment['RUL'].iloc[
                                                               state_index] - WX_t) * current_state - 2 * self.td_alpha * (
                                       -(1 / self.td_beta) - WX_t) * current_state
                    self.agent.update_lr_weights_by_gradient_21(gradient, learning_rate)  # gradient descent


                # t가 '1 <= t < tau_i' 인 경우에는 아래와 같이 gradient를 업데이트 (엔진 내에서 마지막 time cycle이 아닐 때)
                else:
                    time_difference = RL_env.environment['time_cycles'].iloc[next_state_index] - \
                                      RL_env.environment['time_cycles'].iloc[state_index]

                    gradient = -2 * (1 - self.td_alpha) * (RL_env.environment['RUL'].iloc[
                                                               state_index] - WX_t) * current_state - 2 * self.td_alpha * (
                                       time_difference + max(WX_t_1, 0) - WX_t) * current_state
                    self.agent.update_lr_weights_by_gradient_21(gradient, learning_rate)  # gradient descent

                # 다음 상태로 이동
                state_index = next_state_index

                # 22차원용 코드
                # state = RL_env.lr_states.iloc[state_index].values
                # 21차원용 코드 (RL과 성능 일치하는지 확인용)
                state = RL_env.states.iloc[state_index].values

                num_of_step += 1

                # 그냥 gradient의 크기가 얼마나 줄어드나 확인하기 위한 용도 (학습과는 상관 없음)
                sum_of_gradient += gradient

        # 학습이 잘 되고 있는지 loss를 확인하기 위한 코드 (TD loss)
        mean_sum_of_gradient = sum_of_gradient / num_of_step
        lr_td_training_loss.append(np.dot(mean_sum_of_gradient, mean_sum_of_gradient))

    def train_many_lr_by_td_loss(self):  # 샘플 데이터셋 전체를 하나의 epoch로 취급.
        # RL학습시키는데 사용한 코드로 td loss를 반복 학습하는 코드 (처음 학습시킬 때 사용)

        # Iterate over the number of sample datasets
        for epoch in range(self.max_epoch):
            print(epoch + 1)

            # Iterate over the number of sample datasets
            for i in range(self.num_sample_datasets):
                # random observation으로 학습시킴.
                self.train_lr_by_td_loss_random_observation(i, epoch, self.td_alpha, self.td_beta,
                                                            self.td_learning_rate)

        self.env.plot_training_loss(self.max_epoch, self.num_sample_datasets, lr_td_training_loss)

        # Save RL_best_weights to a file using pickle
        with open('LR_TD_weight_by_RL_code.pkl', 'wb') as f:
            pickle.dump(self.agent.lr_weights_by_td, f)

    def train_many_lr_by_td_loss_21(self):  # 샘플 데이터셋 전체를 하나의 epoch로 취급.
        # RL학습시키는데 사용한 코드로 td loss를 반복 학습하는 코드 (처음 학습시킬 때 사용)

        # Iterate over the number of sample datasets
        for epoch in range(self.max_epoch):
            print(epoch + 1)

            # Iterate over the number of sample datasets
            for i in range(self.num_sample_datasets):
                # random observation으로 학습시킴.
                self.train_lr_by_td_loss_random_observation_21(i, epoch, self.td_alpha, self.td_beta,
                                                            self.td_learning_rate)

        self.env.plot_training_loss(self.max_epoch, self.num_sample_datasets, lr_td_training_loss)

        # Save RL_best_weights to a file using pickle
        with open('LR_TD_weight_by_RL_code.pkl', 'wb') as f:
            pickle.dump(self.agent.lr_weights_by_td_21, f)

    def train_continue_many_lr_by_td_loss(self):
        # RL학습시키는데 사용한 코드로 td loss를 이어서 학습시키는 코드.
        with open('LR_TD_weight_by_RL_code.pkl', 'rb') as f:
            self.agent.lr_weights_by_td = pickle.load(f)

        # Iterate over the number of sample datasets
        for epoch in range(self.max_epoch):
            # decay epsilon (linear)
            # print test
            print(epoch + 1)

            # Iterate over the number of sample datasets
            for i in range(self.num_sample_datasets):
                self.train_lr_by_td_loss_random_observation(i, epoch, self.td_alpha, self.td_beta, self.td_learning_rate)

        self.env.plot_training_loss(self.max_epoch, self.num_sample_datasets, lr_td_training_loss)

        # Save RL_best_weights to a file using pickle
        with open('LR_TD_weight_by_RL_code.pkl', 'wb') as f:
            pickle.dump(self.agent.lr_weights_by_td, f)


    def train_RL_off_policy(self, data_sample_index, epsilon, episode):
        # off-policy TD(0) [Q-learning] 코드. 이 코드를 이용해서 베타로 변형한 문제를 학습시켰음)
        # off-policy로 학습. 학습된 weight을 train에는 사용하지 않음.
        # behavior policy : continue;  target policy = greedy (maxQ)
        # 계속 continue만 하더라도 target policy는 maxQ이기에 q-value를 approximation 할 수 있음.
        replace_failure = 0
        state_index = 0  # index를 가리키는 pointer로 사용 (episode 마다 초기화)
        total_reward = 0
        total_actual_reward = 0
        num_of_step = 0
        loss_episode = 0

        # reward의 정의상 replace의 reward는 0이므로, 0으로 weight을 초기화.
        # 뒤에선 chosen action (continue)에 대해서만 weight이 업데이트가 되므로, 학습이 끝나도 replace의 q-value는 항상 0암.
        self.agent.save_weights('replace', np.zeros(21))  # set 0

        train_data = self.sampled_datasets_with_RUL[data_sample_index][0].copy()
        train_data[self.columns_to_scale] = train_data[self.columns_to_scale].apply(self.env.min_max_scaling, axis=0)

        # out of index 에러 방지용으로 넣어두는 dummy row들임. 학습하는 것과는 관계 없음.
        # dummy row 추가 (마지막 row를 2번 복사해서 뒤에 추가)
        dummy_row = train_data.iloc[-1].copy()
        dummy_row['unit_number'] = int(dummy_row['unit_number']) + 1
        dummy_row_2 = dummy_row.copy()
        dummy_row_2['unit_number'] = int(dummy_row_2['unit_number']) + 1

        train_data = train_data._append(dummy_row)
        train_data = train_data._append(dummy_row_2)
        # float으로 변환된 자료형을 다시 int로 변환 (에러 방지용)
        train_data['unit_number'] = train_data['unit_number'].astype(int)
        train_data['time_cycles'] = train_data['time_cycles'].astype(int)
        train_data['RUL'] = train_data['RUL'].astype(int)

        train_data.reset_index(drop=True, inplace=True)  # index reset (reset 하지 않으면 state 전이가 되지 않음)

        RL_env = Environment(train_data) # 인스턴스 생성.

        for unit_num in range(RL_env.max_unit_number):  # unit num : 0, ... , (max_unit_number - 1)
            state = RL_env.states.iloc[state_index].values

            # 항상 continue action만 수행. (behavior policy : 무조건 continue)
            while (state_index < RL_env.environment[
                RL_env.environment['unit_number'] == (RL_env.environment['unit_number'].max() - 2)].index[-1] + 1) \
                    and RL_env.environment['unit_number'].iloc[state_index] == (unit_num + 1):
                current_state = state
                chosen_action = 'continue'

                next_state_index = RL_env.nextStateIndex(chosen_action, state_index)
                next_state = RL_env.states.iloc[next_state_index].values

                current_reward = self.reward.get_reward(state_index, next_state_index, chosen_action,
                                                        RL_env.environment)

                # count 'replace failure'
                if current_reward == (self.reward.r_continue_but_failure):
                    replace_failure += 1

                # update q-value (Linear Function Approximation)
                # 기존에 잘못하고 있었던 코드.
                """
                # next state q는 max_q로 구하고, current action은 continue.
                next_state_q = max([np.dot(self.agent.weights[a], next_state) for a in self.agent.actions]) """

                # 새롭게 변경한 코드. 기존과 달리 현재 state가 엔진 내에서 마지막 state라면 next_q는 0으로 되도록 함.
                # 이렇게 해야 RUL prediction과 연관지을 때 괴리가 없음. 안그러면 다음 엔진의 첫번째 state의 q-value를 가져오게됨.
                if current_reward == self.reward.r_continue_but_failure:
                    next_state_q = 0
                else:
                    next_state_q = max([np.dot(self.agent.weights[a], next_state) for a in self.agent.actions])

                current_state_q = np.dot(self.agent.weights[chosen_action],
                                         current_state)  # A  ~ random generated episode

                # TD target, weight
                TD_target = current_reward + self.gamma * next_state_q
                delta_w = self.alpha * (TD_target - current_state_q) * current_state  # current state -> gradient

                # update weights
                self.agent.save_weights(chosen_action, self.agent.weights[chosen_action] + delta_w)

                # 총 리워드 업데이트
                total_reward += current_reward
                loss_episode += TD_target - current_state_q

                # 원래 문제의 reward 저장 (출력용; 학습에 사용 x)
                total_actual_reward += self.reward.get_actual_reward(state_index, next_state_index, chosen_action,
                                                                     RL_env.environment)

                # 다음 상태로 이동
                state_index = next_state_index
                state = RL_env.states.iloc[state_index].values
                num_of_step += 1

        # episode 학습 결과 출력
        average_reward = total_reward / num_of_step
        average_actual_reward = total_actual_reward / num_of_step

        # best weight 저장
        if average_reward > self.best_average_reward:
            self.best_average_reward = copy(average_reward)
            self.agent.save_best_weights(self.agent.get_weights())

        average_rewards.append(average_reward)
        average_actual_rewards.append(average_actual_reward)
        training_loss.append(np.abs(loss_episode))


    def train_RL_new(self, data_sample_index, epsilon, episode): # method 호출 당, 전체 엔진에 대해 학습이 진행됨.
        # 이 코드도 특정 t시점을 미리 정해두고 그떄까지는 쭉 exploration 하는 코드임. off-policy 설정이 안들어가있음.
        # behavior policy와, target policy 모두 exploration을 하는 코드고 시간이 지날수록 greedy하게 학습되도록 해둠.
        replace_failure = 0
        state_index = 0     # index를 가리키는 pointer로 사용 (episode 마다 초기화)
        total_reward = 0
        total_actual_reward = 0
        num_of_step = 0
        loss_episode = 0

        train_data = self.sampled_datasets_with_RUL[data_sample_index][0].copy()
        train_data[self.columns_to_scale] = train_data[self.columns_to_scale].apply(self.env.min_max_scaling, axis=0)
        train_data.reset_index(drop=True, inplace=True) # index reset (reset 하지 않으면 state 전이가 되지 않음)

        RL_env = Environment(train_data)

        for unit_num in range(RL_env.max_unit_number):      # unit num : 0, ... , (max_unit_number - 1)
            state = RL_env.states.iloc[state_index].values

            # random stopping time을 위한 조건 분기 (exploration)
            if np.random.rand() < epsilon:
                t_replace = random.randint(self.min_t_replace, self.max_t_replace)  # 하나의 unit에 대해서만 t_replace를 뽑음.
                while (state_index < RL_env.environment[
                    RL_env.environment['unit_number'] == (RL_env.environment['unit_number'].max() - 2)].index[-1] + 1) \
                        and RL_env.environment['unit_number'].iloc[state_index] == (unit_num + 1):
                    current_state = state
                    chosen_action = 'continue' if state_index < RL_env.stateMinIndex(
                        state_index) + t_replace else 'replace'  # 미리 언제 replace를 할 지 정해둠 (epsilon greedy와 유사한 효과).

                    next_state_index = RL_env.nextStateIndex(chosen_action, state_index)
                    next_state = RL_env.states.iloc[next_state_index].values

                    current_reward = self.reward.get_reward(state_index, next_state_index, chosen_action, RL_env.environment)

                    # count 'replace failure'
                    if current_reward == (self.reward.r_continue_but_failure):
                        replace_failure += 1

                    # next action
                    next_chosen_action = 'continue' if next_state_index < RL_env.stateMinIndex(
                        next_state_index) + t_replace else 'replace'

                    # update q-value (Linear Function Approximation)
                    next_state_q = np.dot(self.agent.weights[next_chosen_action],
                                          next_state)        # A' ~ random generated episode
                    current_state_q = np.dot(self.agent.weights[chosen_action],
                                             current_state)  # A  ~ random generated episode

                    # TD target, weight
                    TD_target = current_reward + self.gamma * next_state_q
                    delta_w = self.alpha * (TD_target - current_state_q) * current_state  # current state -> gradient

                    # update weights
                    #self.agent.weights[chosen_action] = self.agent.weights[chosen_action] + delta_w

                    # update weights
                    self.agent.save_weights(chosen_action, self.agent.weights[chosen_action] + delta_w)

                    # 총 리워드 업데이트
                    total_reward += current_reward
                    loss_episode += TD_target - current_state_q

                    # 원래 문제의 reward 저장 (출력용; 학습에 사용 x)
                    total_actual_reward += self.reward.get_actual_reward(state_index, next_state_index, chosen_action, RL_env.environment)


                    # 다음 상태로 이동
                    state_index = next_state_index
                    state = RL_env.states.iloc[state_index].values
                    num_of_step += 1

                # random episode가 아닌 경우, greedy action 수행.
            else:
                while (state_index <
                       RL_env.environment[
                           RL_env.environment['unit_number'] == (RL_env.environment['unit_number'].max() - 2)].index[
                           -1] + 1) and RL_env.environment['unit_number'].iloc[state_index] == (unit_num + 1):
                    current_state = state
                    chosen_action = max(self.agent.actions,
                                        key=lambda a: np.dot(self.agent.weights[a], current_state))  # greedy action
                    next_state_index = RL_env.nextStateIndex(chosen_action, state_index)
                    next_state = RL_env.states.iloc[next_state_index].values
                    current_reward = self.reward.get_reward(state_index, next_state_index, chosen_action, RL_env.environment)

                    # count 'replace failure'
                    if current_reward == (self.reward.r_continue_but_failure):
                        replace_failure += 1

                    # update q-value (Linear Function Approximation)
                    next_state_q = max(
                        [np.dot(self.agent.weights[a], next_state) for a in self.agent.actions])  # A' ~ greedy action
                    current_state_q = np.dot(self.agent.weights[chosen_action], current_state)    # A  ~ greedy action

                    TD_target = current_reward + self.gamma * next_state_q
                    delta_w = self.alpha * (TD_target - current_state_q) * current_state

                    #self.agent.weights[chosen_action] = self.agent.weights[chosen_action] + delta_w

                    # update weights
                    self.agent.save_weights(chosen_action, self.agent.weights[chosen_action] + delta_w)

                    total_reward += current_reward
                    loss_episode += TD_target - current_state_q

                    # 원래 문제의 reward 저장 (출력용; 학습에 사용 x)
                    total_actual_reward += self.reward.get_actual_reward(state_index, next_state_index,
                                                                             chosen_action, RL_env.environment)

                    # 다음 상태로 이동
                    state_index = next_state_index
                    state = RL_env.states.iloc[state_index].values
                    num_of_step += 1
        # episode 학습 결과 출력
        average_reward = total_reward / num_of_step
        average_actual_reward = total_actual_reward / num_of_step

        # best weight 저장
        if average_reward > self.best_average_reward:
            self.best_average_reward = copy(average_reward)
            self.agent.save_best_weights(self.agent.get_weights())

        average_rewards.append(average_reward)
        average_actual_rewards.append(average_actual_reward)
        training_loss.append(np.abs(loss_episode))

        print(
            f"episode : {episode + 1}, replace failure : {replace_failure}, average reward : {average_reward}, "
            f"loss : {np.abs(loss_episode)}, actual average reward : {average_actual_reward}")

    def train_RL(self, data_sample_index, epsilon, episode):  # 한번의 episode에 해당됨.
        # 초기에 썼던 코드. Exploration을 특정 t를 정해두고 그떄까지는 쭉 continue 하도록 학습시킴.
        replace_failure = 0  # each episode 마다 초기화. 누적시킬 필요는 없음.
        state_index = 0  # state index -> index pointer로 취급하자. (episode 마다 초기화)
        total_reward = 0
        num_of_step = 0
        loss_episode = 0

        train_data = self.sampled_datasets_with_RUL[data_sample_index][0].copy()

        train_data[self.columns_to_scale] = train_data[self.columns_to_scale].apply(self.env.min_max_scaling, axis=0)

        train_data.reset_index(drop=True, inplace=True) # index reset. (리셋하지 않으면 state 전이가 되지 않음)

        RL_env = Environment(train_data)

        for unit_num in range(RL_env.max_unit_number):  # unit num : 0, .... , (max_unit_number - 1)
            state = RL_env.states.iloc[state_index].values

            # random stopping time을 위한 조건 분기 (exploration)
            if np.random.rand() < epsilon:
                t_replace = random.randint(self.min_t_replace, self.max_t_replace)  # 하나의 unit에 대해서만 t_replace를 뽑음.
                while (state_index < RL_env.environment[RL_env.environment['unit_number'] == (RL_env.environment['unit_number'].max() - 2)].index[-1] + 1) \
                       and RL_env.environment['unit_number'].iloc[state_index] == (unit_num + 1):
                    current_state = state
                    chosen_action = 'continue' if state_index < RL_env.stateMinIndex(state_index) + t_replace else 'replace'


                    next_state_index = RL_env.nextStateIndex(chosen_action, state_index)
                    next_state = RL_env.states.iloc[next_state_index].values

                    current_reward = self.reward.get_reward(state_index, next_state_index, chosen_action, RL_env.environment)

                    # count 'replace failure'
                    if current_reward == (self.reward.r_continue_but_failure):
                        replace_failure += 1

                    # next action
                    next_chosen_action = 'continue' if next_state_index < RL_env.stateMinIndex(next_state_index) + t_replace else 'replace'

                    # update q-value (Linear Function Approximation)
                    next_state_q = np.dot(self.agent.weights[next_chosen_action], next_state)   # A' ~ random generated episode
                    current_state_q = np.dot(self.agent.weights[chosen_action], current_state)  # A  ~ random generated episode

                    # TD target, weight
                    TD_target = current_reward + self.gamma * next_state_q
                    delta_w = self.alpha * (TD_target - current_state_q) * current_state

                    self.agent.weights[chosen_action] = self.agent.weights[chosen_action] + delta_w

                    # 총 리워드 업데이트
                    total_reward += current_reward
                    loss_episode += TD_target - current_state_q

                    # 다음 상태로 이동
                    state_index = next_state_index
                    state = RL_env.states.iloc[state_index].values
                    num_of_step += 1

            # random episode가 아닌 경우, greedy action 수행.
            else:
                while (state_index <
                       RL_env.environment[RL_env.environment['unit_number'] == (RL_env.environment['unit_number'].max() - 2)].index[
                           -1] + 1) and RL_env.environment['unit_number'].iloc[state_index] == (unit_num + 1):
                    current_state = state
                    chosen_action = max(self.agent.actions, key=lambda a: np.dot(self.agent.weights[a], current_state))  # greedy action
                    next_state_index = RL_env.nextStateIndex(chosen_action, state_index)
                    next_state = RL_env.states.iloc[next_state_index].values
                    current_reward = self.reward.get_reward(state_index, next_state_index, chosen_action, RL_env.environment)

                    # count 'replace failure'
                    if current_reward == (self.reward.r_continue_but_failure):
                        replace_failure += 1

                    # update q-value (Linear Function Approximation)
                    next_state_q = max([np.dot(self.agent.weights[a], next_state) for a in self.agent.actions])  # A' ~ greedy action
                    current_state_q = np.dot(self.agent.weights[chosen_action], current_state)  # A  ~ greedy action

                    TD_target = current_reward + self.gamma * next_state_q
                    delta_w = self.alpha * (TD_target - current_state_q) * current_state

                    self.agent.weights[chosen_action] = self.agent.weights[chosen_action] + delta_w

                    total_reward += current_reward
                    loss_episode += TD_target - current_state_q

                    # 다음 상태로 이동
                    state_index = next_state_index
                    state = RL_env.states.iloc[state_index].values
                    num_of_step += 1


        # episode 학습 결과 출력
        average_reward = total_reward / num_of_step
        #average_number_of_observation = number_of_observation / RL_env.max_unit_number

        # best weight 저장
        if average_reward > self.best_average_reward:
            self.best_average_reward = copy(average_reward)
            self.agent.save_best_weights(self.agent.get_weights())  # 이 method를 그대로 활용하려면 위에서 weight을 agent에 save 해뒀어야함.

        average_rewards.append(average_reward)
        training_loss.append(np.abs(loss_episode))
        #average_number_of_observations.append(average_number_of_observation)
        print(
            f"episode : {episode + 1}, replace failure : {replace_failure}, Average Reward : {average_reward}, "
            f"loss : {np.abs(loss_episode)}")

    def train_many_off_policy_RL(self): # 샘플 데이터셋 전체를 하나의 episode로 취급.
        # off policy Q-learning [ TD(0) control ] 로 반복 학습하는 method (최종 결과 내는데 사용함)
        # Set Q_repalce = 0 (by definition)
        self.agent.save_weights('replace', np.zeros(21))

        # Iterate over the number of sample datasets
        for episode in range(self.max_episodes):
            # decay epsilon (linear)
            # print test
            print(episode + 1)
            epsilon = max(self.min_epsilon, self.initial_epsilon - episode * self.epsilon_delta)

            # Iterate over the number of sample datasets
            for i in range(self.num_sample_datasets):
                self.train_RL_off_policy(i, epsilon, episode)

        # self.agent.get_best_weights() 이걸 이용해서 best weights을 저장하자.

        self.env.plot_average_reward(self.max_episodes, self.num_sample_datasets, average_rewards)
        self.env.plot_training_loss(self.max_episodes, self.num_sample_datasets, training_loss)
        self.env.plot_actual_average_reward(self.max_episodes, self.num_sample_datasets, average_actual_rewards)

        # Save RL_best_weights to a file using pickle
        with open('RL_best_weights_continue_240425.pkl', 'wb') as f:
            pickle.dump(self.agent.get_best_weights(), f)

    def train_many_by_saved_weights_off_policy_RL(self): # 샘플 데이터셋 전체를 하나의 episode로 취급.
        # 사전에 학습된 weights으로 이어서 학습하기 위한 method.
        with open('RL_best_weights_continue_240425.pkl', 'rb') as f:
            self.agent.best_weights = pickle.load(f)

        self.agent.weights = self.agent.best_weights
        self.agent.save_weights('replace', np.zeros(21))

        # Iterate over the number of sample datasets
        for episode in range(self.max_episodes):
            # decay epsilon (linear)
            # print test
            print(episode + 1)
            epsilon = max(self.min_epsilon, self.initial_epsilon - episode * self.epsilon_delta)

            # Iterate over the number of sample datasets
            for i in range(self.num_sample_datasets):
                self.train_RL_off_policy(i, epsilon, episode)

        # self.agent.get_best_weights() 이걸 이용해서 best weights을 저장하자.

        self.env.plot_average_reward(self.max_episodes, self.num_sample_datasets, average_rewards)
        self.env.plot_training_loss(self.max_episodes, self.num_sample_datasets, training_loss)
        self.env.plot_actual_average_reward(self.max_episodes, self.num_sample_datasets, average_actual_rewards)

        # Save RL_best_weights to a file using pickle
        with open('RL_best_weights_continue_240425.pkl', 'wb') as f:
            pickle.dump(self.agent.get_best_weights(), f)
    def train_many_by_saved_weights_RL(self): # 샘플 데이터셋 전체를 하나의 episode로 취급.
        # 사전에 학습된 weights으로 이어서 학습하기 위한 method.
        with open('RL_best_weights_continue_240425.pkl', 'rb') as f:
            self.agent.best_weights = pickle.load(f)

        # Iterate over the number of sample datasets
        for episode in range(self.max_episodes):
            # decay epsilon (linear)
            # print test
            print(episode + 1)
            epsilon = max(self.min_epsilon, self.initial_epsilon - episode * self.epsilon_delta)

            # Iterate over the number of sample datasets
            for i in range(self.num_sample_datasets):
                self.train_RL_new(i, epsilon, episode)

        # self.agent.get_best_weights() 이걸 이용해서 best weights을 저장하자.

        self.env.plot_average_reward(self.max_episodes, self.num_sample_datasets, average_rewards)
        self.env.plot_training_loss(self.max_episodes, self.num_sample_datasets, training_loss)
        self.env.plot_actual_average_reward(self.max_episodes, self.num_sample_datasets, average_actual_rewards)

        # Save RL_best_weights to a file using pickle
        with open('RL_best_weights_continue_240425.pkl', 'wb') as f:
            pickle.dump(self.agent.get_best_weights(), f)

    def train_many_RL(self):
        # Iterate over the number of sample datasets
        escape_const = 0
        for episode in range(self.max_episodes):
            # Iterate over the number of sample datasets
            for i in range(self.num_sample_datasets):
                # decay epsilon (linear)
                epsilon = max(self.min_epsilon, self.initial_epsilon - escape_const * self.epsilon_delta)
                self.train_RL(i, epsilon, escape_const)
                escape_const += 1
                if escape_const == (self.max_episodes):
                    break
            if escape_const == (self.max_episodes):
                break
        # 지금 상황은 (max_episodes - num_sample_datasets) * num_sample_datasets만큼 추가로 학습이 진행됨. (episode가 하나씩 밀리며)
        # self.agent.get_best_weights() 이걸 이용해서 best weights을 저장하자.

        self.env.plot_average_reward(self.max_episodes, average_rewards)
        self.env.plot_training_loss(self.max_episodes, training_loss)
        #self.env.plot_number_of_observation(self.max_episodes, average_number_of_observations)

        # Save RL_best_weights to a file using pickle
        with open('RL_best_weights_continue_0.pkl', 'wb') as f:
            pickle.dump(self.agent.get_best_weights(), f)

    def test_RL_random_observation(self, data_sample_index):
        # 매 순간 관측하지 못할 때, 학습된 RL weight으로 test
        global test_average_rewards, test_average_usage_times, test_replace_failures
        replace_failure = 0  # each episode 마다 초기화. 누적시킬 필요는 없음.
        state_index = 0      # state index -> index pointer로 취급하자. (episode 마다 초기화)
        total_reward = 0
        total_actual_reward = 0
        num_of_step = 0
        total_operation_time = 0

        full_data = self.sampled_datasets_with_RUL[data_sample_index][2].copy()
        full_data[self.columns_to_scale] = full_data[self.columns_to_scale].apply(self.env.min_max_scaling, axis=0)

        #### dummy row 추가 (마지막 row를 2번 복사해서 뒤에 추가) ##################
        dummy_row = full_data.iloc[-1].copy()
        dummy_row['unit_number'] = int(dummy_row['unit_number']) + 1
        dummy_row_2 = dummy_row.copy()
        dummy_row_2['unit_number'] = int(dummy_row_2['unit_number']) + 1

        full_data = full_data._append(dummy_row)
        full_data = full_data._append(dummy_row_2)
        # float으로 변환된 자료형을 다시 int로 변환
        full_data['unit_number'] = full_data['unit_number'].astype(int)
        full_data['time_cycles'] = full_data['time_cycles'].astype(int)
        full_data['RUL'] = full_data['RUL'].astype(int)
        ####################################################################

        full_data.reset_index(drop=True, inplace=True) # index reset. (리셋하지 않으면 state 전이가 되지 않음)

        RL_env = Environment(full_data)

        for unit_num in range(RL_env.max_unit_number):  # unit num : 0, .... , (max_unit_number - 1)
            state = RL_env.states.iloc[state_index].values

            while (state_index <
                   RL_env.environment[
                       RL_env.environment['unit_number'] == (RL_env.environment['unit_number'].max() - 2)].index[
                       -1] + 1) and RL_env.environment['unit_number'].iloc[state_index] == (unit_num + 1):
                current_state = state
                chosen_action = max(self.agent.actions,
                                    key=lambda a: np.dot(self.agent.best_weights[a], current_state))  # greedy action
                next_state_index = RL_env.nextStateIndex(chosen_action, state_index)
                next_state = RL_env.states.iloc[next_state_index].values
                current_reward = self.reward.get_reward(state_index, next_state_index, chosen_action, RL_env.environment)

                if chosen_action == 'replace':
                    print('replace')
                    print(RL_env.environment.iloc[state_index]['time_cycles'])
                    total_operation_time += RL_env.environment.iloc[state_index]['time_cycles']

                # count 'replace failure'
                if current_reward == (self.reward.r_continue_but_failure):
                    print('continue but failure')
                    print(RL_env.environment.iloc[state_index]['time_cycles'])
                    total_operation_time += RL_env.environment.iloc[state_index]['time_cycles']
                    replace_failure += 1

                # update total reward.
                total_reward += current_reward

                # 원래 문제의 reward 저장 (출력용; 학습에 사용 x)
                total_actual_reward += self.reward.get_actual_reward(state_index, next_state_index,
                                                                     chosen_action, RL_env.environment)

                # move next state.
                state_index = next_state_index
                state = RL_env.states.iloc[state_index].values
                num_of_step += 1

        # average_reward = total_reward / num_of_step
        # average_actual_reward = total_actual_reward / num_of_step
        average_reward = total_reward / total_operation_time
        average_actual_reward = total_actual_reward / total_operation_time
        average_usage_time = total_operation_time / (RL_env.environment['unit_number'].max() - 2)
        print(
            f"number of engine : {RL_env.environment['unit_number'].max() - 2}, average reward : {average_reward},"
            f" actual average reward : {average_actual_reward},"
            f" replace failure : {replace_failure}, average usage time : {average_usage_time}")

        test_average_rewards.append(average_reward)
        test_average_actual_rewards.append(average_actual_reward)
        test_average_usage_times.append(average_usage_time)
        test_replace_failures.append(replace_failure)

    def test_TD_loss_random_observation(self, data_sample_index):
        global test_average_rewards, test_average_usage_times, test_replace_failures
        replace_failure = 0  # each episode 마다 초기화. 누적시킬 필요는 없음.
        state_index = 0      # state index -> index pointer로 취급하자. (episode 마다 초기화)
        total_reward = 0
        total_actual_reward = 0
        num_of_step = 0
        total_operation_time = 0

        full_data = self.sampled_datasets_with_RUL[data_sample_index][2].copy()
        full_data[self.columns_to_scale] = full_data[self.columns_to_scale].apply(self.env.min_max_scaling, axis=0)

        #### dummy row 추가 (마지막 row를 2번 복사해서 뒤에 추가) ##################
        dummy_row = full_data.iloc[-1].copy()
        dummy_row['unit_number'] = int(dummy_row['unit_number']) + 1
        dummy_row_2 = dummy_row.copy()
        dummy_row_2['unit_number'] = int(dummy_row_2['unit_number']) + 1

        full_data = full_data._append(dummy_row)
        full_data = full_data._append(dummy_row_2)
        # float으로 변환된 자료형을 다시 int로 변환
        full_data['unit_number'] = full_data['unit_number'].astype(int)
        full_data['time_cycles'] = full_data['time_cycles'].astype(int)
        full_data['RUL'] = full_data['RUL'].astype(int)
        ####################################################################

        # s_1 열 왼쪽에 s_0 열을 새롭게 추가하고 모든 값을 1로 초기화
        full_data.insert(loc=full_data.columns.get_loc('s_1'), column='s_0', value=1)

        full_data.reset_index(drop=True, inplace=True) # index reset. (리셋하지 않으면 state 전이가 되지 않음)
        RL_env = Environment(full_data)

        for unit_num in range(RL_env.max_unit_number):  # unit num : 0, .... , (max_unit_number - 1)
            state = RL_env.lr_states.iloc[state_index].values

            while (state_index <
                   RL_env.environment[
                       RL_env.environment['unit_number'] == (RL_env.environment['unit_number'].max() - 2)].index[
                       -1] + 1) and RL_env.environment['unit_number'].iloc[state_index] == (unit_num + 1):
                current_state = state

                if np.dot(self.agent.lr_best_weights['continue'], current_state) > self.td_simulation_threshold:
                    chosen_action = 'continue'
                else:
                    chosen_action = 'replace'
                next_state_index = RL_env.nextStateIndex(chosen_action, state_index)
                #next_state = RL_env.lr_states.iloc[next_state_index].values
                current_reward = self.reward.get_reward(state_index, next_state_index, chosen_action, RL_env.environment)

                if chosen_action == 'replace':
                    print('replace')
                    print(RL_env.environment.iloc[state_index]['time_cycles'])
                    total_operation_time += RL_env.environment.iloc[state_index]['time_cycles']

                # count 'replace failure'
                if current_reward == (self.reward.r_continue_but_failure):
                    print('continue but failure')
                    print(RL_env.environment.iloc[state_index]['time_cycles'])
                    total_operation_time += RL_env.environment.iloc[state_index]['time_cycles'] # 혹시 성능에 차이가 있다면 max time_cycle 쪽 점검
                    replace_failure += 1

                # update total reward.
                total_reward += current_reward

                # 원래 문제의 reward 저장 (출력용; 학습에 사용 x)
                total_actual_reward += self.reward.get_actual_reward(state_index, next_state_index,
                                                                     chosen_action, RL_env.environment)

                # move next state.
                state_index = next_state_index
                state = RL_env.lr_states.iloc[state_index].values
                num_of_step += 1

        average_reward = total_reward / total_operation_time
        average_actual_reward = total_actual_reward / total_operation_time
        average_usage_time = total_operation_time / (RL_env.environment['unit_number'].max() - 2)
        print(
            f"number of engine : {RL_env.environment['unit_number'].max() - 2}, average reward : {average_reward},"
            f" actual average reward : {average_actual_reward},"
            f" replace failure : {replace_failure}, average usage time : {average_usage_time}")

        test_average_rewards.append(average_reward)
        test_average_actual_rewards.append(average_actual_reward)
        test_average_usage_times.append(average_usage_time)
        test_replace_failures.append(replace_failure)

    def test_TD_loss_random_observation_21(self, data_sample_index):
        global test_average_rewards, test_average_usage_times, test_replace_failures
        replace_failure = 0  # each episode 마다 초기화. 누적시킬 필요는 없음.
        state_index = 0      # state index -> index pointer로 취급하자. (episode 마다 초기화)
        total_reward = 0
        total_actual_reward = 0
        num_of_step = 0
        total_operation_time = 0

        full_data = self.sampled_datasets_with_RUL[data_sample_index][2].copy()
        full_data[self.columns_to_scale] = full_data[self.columns_to_scale].apply(self.env.min_max_scaling, axis=0)

        #### dummy row 추가 (마지막 row를 2번 복사해서 뒤에 추가) ##################
        dummy_row = full_data.iloc[-1].copy()
        dummy_row['unit_number'] = int(dummy_row['unit_number']) + 1
        dummy_row_2 = dummy_row.copy()
        dummy_row_2['unit_number'] = int(dummy_row_2['unit_number']) + 1

        full_data = full_data._append(dummy_row)
        full_data = full_data._append(dummy_row_2)
        # float으로 변환된 자료형을 다시 int로 변환
        full_data['unit_number'] = full_data['unit_number'].astype(int)
        full_data['time_cycles'] = full_data['time_cycles'].astype(int)
        full_data['RUL'] = full_data['RUL'].astype(int)
        ####################################################################

        # s_1 열 왼쪽에 s_0 열을 새롭게 추가하고 모든 값을 1로 초기화
        full_data.insert(loc=full_data.columns.get_loc('s_1'), column='s_0', value=1)

        full_data.reset_index(drop=True, inplace=True) # index reset. (리셋하지 않으면 state 전이가 되지 않음)
        RL_env = Environment(full_data)

        for unit_num in range(RL_env.max_unit_number):  # unit num : 0, .... , (max_unit_number - 1)
            state = RL_env.states.iloc[state_index].values

            while (state_index <
                   RL_env.environment[
                       RL_env.environment['unit_number'] == (RL_env.environment['unit_number'].max() - 2)].index[
                       -1] + 1) and RL_env.environment['unit_number'].iloc[state_index] == (unit_num + 1):
                current_state = state

                if np.dot(self.agent.lr_best_weights_21['continue'], current_state) > self.td_simulation_threshold:
                    chosen_action = 'continue'
                else:
                    chosen_action = 'replace'
                next_state_index = RL_env.nextStateIndex(chosen_action, state_index)
                #next_state = RL_env.lr_states.iloc[next_state_index].values
                current_reward = self.reward.get_reward(state_index, next_state_index, chosen_action, RL_env.environment)

                if chosen_action == 'replace':
                    print('replace')
                    print(RL_env.environment.iloc[state_index]['time_cycles'])
                    total_operation_time += RL_env.environment.iloc[state_index]['time_cycles']

                # count 'replace failure'
                if current_reward == (self.reward.r_continue_but_failure):
                    print('continue but failure')
                    print(RL_env.environment.iloc[state_index]['time_cycles'])
                    total_operation_time += RL_env.environment.iloc[state_index]['time_cycles'] # 혹시 성능에 차이가 있다면 max time_cycle 쪽 점검
                    replace_failure += 1

                # update total reward.
                total_reward += current_reward

                # 원래 문제의 reward 저장 (출력용; 학습에 사용 x)
                total_actual_reward += self.reward.get_actual_reward(state_index, next_state_index,
                                                                     chosen_action, RL_env.environment)

                # move next state.
                state_index = next_state_index
                state = RL_env.states.iloc[state_index].values
                num_of_step += 1

        average_reward = total_reward / total_operation_time
        average_actual_reward = total_actual_reward / total_operation_time
        average_usage_time = total_operation_time / (RL_env.environment['unit_number'].max() - 2)
        print(
            f"number of engine : {RL_env.environment['unit_number'].max() - 2}, average reward : {average_reward},"
            f" actual average reward : {average_actual_reward},"
            f" replace failure : {replace_failure}, average usage time : {average_usage_time}")

        test_average_rewards.append(average_reward)
        test_average_actual_rewards.append(average_actual_reward)
        test_average_usage_times.append(average_usage_time)
        test_replace_failures.append(replace_failure)

    def test_RL(self, data_sample_index):
        global test_average_rewards, test_average_usage_times, test_replace_failures
        replace_failure = 0  # each episode 마다 초기화. 누적시킬 필요는 없음.
        state_index = 0      # state index -> index pointer로 취급하자. (episode 마다 초기화)
        total_reward = 0
        num_of_step = 0
        total_operation_time = 0

        full_data = self.sampled_datasets_with_RUL[data_sample_index][2].copy()
        full_data[self.columns_to_scale] = full_data[self.columns_to_scale].apply(self.env.min_max_scaling, axis=0)
        full_data.reset_index(drop=True, inplace=True) # index reset. (리셋하지 않으면 state 전이가 되지 않음)

        RL_env = Environment(full_data)

        for unit_num in range(RL_env.max_unit_number):  # unit num : 0, .... , (max_unit_number - 1)
            state = RL_env.states.iloc[state_index].values

            while (state_index <
                   RL_env.environment[
                       RL_env.environment['unit_number'] == (RL_env.environment['unit_number'].max() - 2)].index[
                       -1] + 1) and RL_env.environment['unit_number'].iloc[state_index] == (unit_num + 1):
                current_state = state
                chosen_action = max(self.agent.actions,
                                    key=lambda a: np.dot(self.agent.best_weights[a], current_state))  # greedy action
                next_state_index = RL_env.nextStateIndex(chosen_action, state_index)
                next_state = RL_env.states.iloc[next_state_index].values
                current_reward = self.reward.get_reward(state_index, next_state_index, chosen_action, RL_env.environment)

                if chosen_action == 'replace':
                    print('replace')
                    print(RL_env.environment.iloc[state_index]['time_cycles'])
                    total_operation_time += RL_env.environment.iloc[state_index]['time_cycles']

                # count 'replace failure'
                if current_reward == (self.reward.r_continue_but_failure):
                    print('continue but failure')
                    print(RL_env.environment.iloc[state_index]['time_cycles'])
                    total_operation_time += RL_env.environment.iloc[state_index]['time_cycles']
                    replace_failure += 1

                # update total reward.
                total_reward += current_reward

                # move next state.
                state_index = next_state_index
                state = RL_env.states.iloc[state_index].values
                num_of_step += 1

        average_reward = total_reward / num_of_step
        average_usage_time = total_operation_time / (RL_env.environment['unit_number'].max() - 2)
        print(
            f"Number of Engine : {RL_env.environment['unit_number'].max() - 2}, Average Reward : {average_reward},"
            f" replace failure : {replace_failure}, average usage time : {average_usage_time}")

        test_average_rewards.append(average_reward)
        test_average_usage_times.append(average_usage_time)
        test_replace_failures.append(replace_failure)

    def run_RL_simulation(self):
        global test_average_rewards, test_average_usage_times, test_replace_failures
        # 이 method는 학습이 완료된 bestweight을 불러와서 테스트 환경에서 실행.
        # 아마도 continue의 reward에 따른 다양한 RL 학습 결과를 테스트 환경에서 실행 후 하나의 plot에 점을 찍어내야 함.
        # 점을 찍을 때, continue의 reward가 무엇이었는지 같이 표시해주면 좋음. reward에 따른 성능을 볼 수 있도록.
        # Load average_by_loss_dfs from the file
        with open('average_by_loss_dfs.pkl', 'rb') as f:
            average_by_loss_dfs = pickle.load(f)
        self.env.plot_simulation_results_scale_up(average_by_loss_dfs, self.num_dataset, self.loss_labels)
        with open('RL_best_weights_continue_240425.pkl', 'rb') as f:
            self.agent.best_weights = pickle.load(f)

        for i in range(self.num_sample_datasets):
            self.test_RL_random_observation(i)

        print(test_average_usage_times)
        print(self.num_sample_datasets)

        test_average_reward = sum(test_average_rewards) / self.num_sample_datasets
        test_average_actual_reward = sum(test_average_actual_rewards) / self.num_sample_datasets
        test_average_usage_time = sum(test_average_usage_times) / self.num_sample_datasets
        test_replace_failure = sum(test_replace_failures) / self.num_sample_datasets
        # 데이터 셋이 바뀌면 probability 뒤의 100은 총 엔진수에 맞게 바꿔줘야 함. 나중에 수정.
        failure_probability = test_replace_failure / 100
        lambda_policy = (-self.REWARD_ACTUAL_REPLACE + failure_probability * (-self.REWARD_ACTUAL_FAILURE + self.REWARD_ACTUAL_REPLACE)) / test_average_usage_time
        beta_policy = lambda_policy / (-self.REWARD_ACTUAL_FAILURE + self.REWARD_ACTUAL_REPLACE)


        print(
            f" Test average reward : {test_average_reward}, Test actual average reward : {test_average_actual_reward},"
            f" Test replace failure : {test_replace_failure}, Test average usage time : {test_average_usage_time}, Failure probability : {failure_probability}, lambda : {lambda_policy}, beta : {beta_policy}")

        #self.env.plot_RL_results_scale_up(average_by_loss_dfs, self.num_dataset, self.loss_labels,
        #                                  test_replace_failure, test_average_usage_time, self.CONTINUE_COST)
    def Q_value_to_RUL(self, data_sample_index, scale):
        # 학습된 continue action에 대한 weights을 이용해 RUL을 예측.
        # replace action에 대한 q-value는 0으로 고정이므로, 이를 threshold로 보면 됨.
        # Q_continue <= Q_replace (0)일 때 replace를 하므로.

        with open('RL_best_weights_beta_00077362686.pkl', 'rb') as f:
            self.agent.best_weights = pickle.load(f)

        full_data = self.sampled_datasets_with_RUL[data_sample_index][2].copy()
        full_data[self.columns_to_scale] = full_data[self.columns_to_scale].apply(self.env.min_max_scaling, axis=0)

        full_data.reset_index(drop=True, inplace=True)  # index reset. (리셋하지 않으면 state 전이가 되지 않음)

        self.env.plot_RUL_prediction_by_q_value(full_data, self.agent.best_weights['continue'], scale)

    def plot_Q_value_to_RUL_all_samples(self, scale):
        for i in range(self.num_sample_datasets):
            self.Q_value_to_RUL(i, scale)

    def lr_td_loss_to_RUL(self, data_sample_index, scale):
        # 학습된 continue action에 대한 weights을 이용해 RUL을 예측.
        # replace action에 대한 q-value는 0으로 고정이므로, 이를 threshold로 보면 됨.
        # Q_continue <= Q_replace (0)일 때 replace를 하므로.

        with open('LR_TD_weight_by_RL_code.pkl', 'rb') as f:
            self.td_weight = pickle.load(f)

        full_data = self.sampled_datasets_with_RUL[data_sample_index][2].copy()
        full_data[self.columns_to_scale] = full_data[self.columns_to_scale].apply(self.env.min_max_scaling, axis=0)

        full_data.insert(loc=full_data.columns.get_loc('s_1'), column='s_0', value=1)

        full_data.reset_index(drop=True, inplace=True)

        self.env.plot_RUL_prediction_by_lr_td_loss(full_data, self.td_weight, scale)

    def plot_lr_td_loss_to_RUL_all_samples(self, scale):
        for i in range(self.num_sample_datasets):
            self.lr_td_loss_to_RUL(i, scale)

    def lr_td_loss_to_RUL_21(self, data_sample_index, scale):
        # 학습된 continue action에 대한 weights을 이용해 RUL을 예측.
        # replace action에 대한 q-value는 0으로 고정이므로, 이를 threshold로 보면 됨.
        # Q_continue <= Q_replace (0)일 때 replace를 하므로.

        with open('LR_TD_weight_by_RL_code_alpha_09.pkl', 'rb') as f:
            self.td_weight = pickle.load(f)

        full_data = self.sampled_datasets_with_RUL[data_sample_index][2].copy()
        full_data[self.columns_to_scale] = full_data[self.columns_to_scale].apply(self.env.min_max_scaling, axis=0)

        #full_data.insert(loc=full_data.columns.get_loc('s_1'), column='s_0', value=1)

        full_data.reset_index(drop=True, inplace=True)

        self.env.plot_RUL_prediction_by_lr_td_loss_21(full_data, self.td_weight, scale)

    def plot_lr_td_loss_to_RUL_all_samples_21(self, scale):
        for i in range(self.num_sample_datasets):
            self.lr_td_loss_to_RUL_21(i, scale)

    def run_lr_simulation(self, data_sample_index):
        # Data preprocessing 1 : separate train, valid, full datasets.
        train_data = self.sampled_datasets_with_RUL[data_sample_index][0].copy()
        valid_data = self.sampled_datasets_with_RUL[data_sample_index][1].copy()
        full_data = self.sampled_datasets_with_RUL[data_sample_index][2].copy()

        # Data preprocessing 2 : separate index, label, data
        train_index_names, x_train, x_test, y_train, y_test = self.env.drop_labels_from_train_data(train_data)
        valid_index_names, x_valid, y_valid = self.env.drop_labels_from_data(valid_data)
        full_index_names, x_full, y_full = self.env.drop_labels_from_data(full_data)

        # Data preprocessing 3 : scaling
        x_train = self.env.data_scaler(x_train)
        x_valid = self.env.data_scaler(x_valid)
        x_full = self.env.data_scaler(x_full)

        # Data preprocessing 4 : Clipping y_train to have 195 as the maximum value (it means y = 197 -> y = 195)
        #y_train = y_train.clip(upper = 195)

        # 1. Original Linear Regression #################################################
        lr1 = LinearRegression()
        lr1.fit(x_train, y_train)  # Fitting
        y_lr1_train, y_lr1_valid, y_lr1_full = self.env.predict_and_save(lr1, x_train, x_valid, x_full)

        # 2. Crucial moments loss function - Linear Regression ##########################
        # Filter and save only data that is less than crucial moment constant.
        filtered_data = x_train[y_train <= self.crucial_moment]
        filtered_labels = y_train[y_train <= self.crucial_moment]

        lr2 = LinearRegression()
        lr2.fit(filtered_data, filtered_labels)  # Fitting
        y_lr2_train, y_lr2_valid, y_lr2_full = self.env.predict_and_save(lr2, x_train, x_valid, x_full)

        # 3. TD Loss function - Linear Regression (ridge) ################################
        lr3 = Linear_Regression_TD()
        lr3.fit(x_train, y_train, 0.5, 10)  # Fitting; fit(X, Y, alpha, lambda)
        y_lr3_train, y_lr3_valid, y_lr3_full = self.env.predict_and_save(lr3, x_train, x_valid, x_full)

        """
        y_train_float = y_train.astype(np.float32)  # penalty를 활용할 때 data type을 맞춰주기 위함.
        lr3 = models.Sequential([
            layers.Input(shape=(x_train.shape[1],)),
            layers.Dense(1)   # fully connected layer -> 1 (number of output node)
        ])

        lr3.compile(optimizer='adam', loss=lambda y_true, y_pred: different_td_loss(y_true, y_pred, self.td_alpha))
        # fit
        lr3.fit(x_train, y_train_float, epochs=2500, verbose=0)  # fitting (verbose=2 -> print loss / epoch)
        
        for i in range(len(x_train)):
            x = x_train[i:i + 1]
            y = y_train_float[i:i + 1]
            lr3.train_on_batch(x, y)

            # 이전 예측값과 이전 실제 값 업데이트
            previous_prediction = lr3.predict(x)
            previous_true_label = y

        y_lr3_train, y_lr3_valid, y_lr3_full = self.env.predict_and_save(lr3, x_train, x_valid, x_full)
        """

        # 4. TD + Crucial moments loss function - Linear Regression (ridge) ##############
        filtered_data = x_train[y_train <= self.td_crucial_moment]
        filtered_labels = y_train[y_train <= self.td_crucial_moment]

        lr4 = Linear_Regression_TD()
        lr4.fit(filtered_data, filtered_labels, 0.5, 10)  # Fitting; fit(X, Y, alpha, lambda)
        y_lr4_train, y_lr4_valid, y_lr4_full = self.env.predict_and_save(lr4, x_train, x_valid, x_full)

        # 5. Directed MSE ################################################################
        y_train_float = y_train.astype(np.float32) # penalty를 활용할 때 data type을 맞춰주기 위함.
        lr5 = models.Sequential([
            layers.Input(shape=(x_train.shape[1],)),
            layers.Dense(1)   # fully connected layer -> 1 (number of output node)
        ])
        lr5.compile(optimizer='adam', loss=directed_mse_loss)   # model compile
        lr5.fit(x_train, y_train_float, epochs=2500, verbose=0) # fitting (verbose=2 -> print loss / epoch)
        y_lr5_train, y_lr5_valid, y_lr5_full = self.env.predict_and_save(lr5, x_train, x_valid, x_full)

        # 6. Directed Crucial moments MSE #################################################
        filtered_data = x_train[y_train <= self.directed_crucial_moment]
        filtered_labels = y_train_float[y_train_float <= self.directed_crucial_moment]

        lr6 = models.Sequential([
            layers.Input(shape=(x_train.shape[1],)),
            layers.Dense(1)   # fully connected layer -> 1 (number of output node)
        ])
        lr6.compile(optimizer='adam', loss=directed_mse_loss)            # model compile
        lr6.fit(filtered_data, filtered_labels, epochs=2500, verbose=0)  # fitting (verbose=2 -> print loss / epoch)
        y_lr6_train, y_lr6_valid, y_lr6_full = self.env.predict_and_save(lr6, x_train, x_valid, x_full)

        # Remove index for concat. If not removed, NaN values will be included in the dataframe.
        valid_index_names = valid_index_names.reset_index(drop=True)
        y_valid = y_valid.reset_index(drop=True)
        full_index_names = full_index_names.reset_index(drop=True)
        y_full = y_full.reset_index(drop=True)

        # merge dataframe (valid)
        merged_valid_dfs = self.env.merge_dataframe(valid_index_names, y_valid,
                                                    y_lr1_valid, y_lr2_valid, y_lr3_valid,
                                                    y_lr4_valid, y_lr5_valid, y_lr6_valid)

        # merge dataframe (full)
        merged_full_dfs = self.env.merge_dataframe(full_index_names, y_full,
                                                    y_lr1_full, y_lr2_full, y_lr3_full,
                                                    y_lr4_full, y_lr5_full, y_lr6_full)

        # plot online prediction
        #self.env.plot_online_RUL_prediction(merged_full_dfs)

        # simulation by threshold
        full_by_threshold_dfs_list = self.env.random_obs_simulation_by_threshold(merged_full_dfs, self.threshold_values,
                                                                      -self.REWARD_ACTUAL_REPLACE, -self.REWARD_ACTUAL_FAILURE)
        full_by_loss_dfs = self.env.calculate_NoF_AUT_by_threshold(full_by_threshold_dfs_list, self.threshold_values)

        # save the learning results to a global variable
        full_by_loss_dfs_list.append(full_by_loss_dfs)

        #self.env.plot_NoF_AUT_by_threshold(full_by_loss_dfs, self.num_dataset)
        #self.env.plot_AC_AUT_by_threshold(full_by_loss_dfs, self.num_dataset)

        #self.env.plot_simulation_results(full_by_loss_dfs, self.num_dataset)
        #self.env.plot_simulation_results_scale_up(full_by_loss_dfs, self.num_dataset, self.loss_labels)

    def run_lr_simulation_only_MSE(self, data_sample_index):
        # Data preprocessing 1 : separate train, valid, full datasets.
        train_data = self.sampled_datasets_with_RUL[data_sample_index][0].copy()
        valid_data = self.sampled_datasets_with_RUL[data_sample_index][1].copy()
        full_data = self.sampled_datasets_with_RUL[data_sample_index][2].copy()

        # Data preprocessing 2 : separate index, label, data
        train_index_names, x_train, x_test, y_train, y_test = self.env.drop_labels_from_train_data(train_data)
        valid_index_names, x_valid, y_valid = self.env.drop_labels_from_data(valid_data)
        full_index_names, x_full, y_full = self.env.drop_labels_from_data(full_data)

        # Data preprocessing 3 : scaling
        x_train = self.env.data_scaler(x_train)
        x_valid = self.env.data_scaler(x_valid)
        x_full = self.env.data_scaler(x_full)

        # 1. Original Linear Regression #################################################
        lr1 = LinearRegression()
        lr1.fit(x_train, y_train)  # Fitting
        y_lr1_train, y_lr1_valid, y_lr1_full = self.env.predict_and_save(lr1, x_train, x_valid, x_full)

        # Remove index for concat. If not removed, NaN values will be included in the dataframe.
        valid_index_names = valid_index_names.reset_index(drop=True)
        y_valid = y_valid.reset_index(drop=True)
        full_index_names = full_index_names.reset_index(drop=True)
        y_full = y_full.reset_index(drop=True)

        # merge dataframe (valid)
        merged_valid_dfs = self.env.merge_dataframe_only_MSE(valid_index_names, y_valid, y_lr1_valid)

        # merge dataframe (full)
        merged_full_dfs = self.env.merge_dataframe_only_MSE(full_index_names, y_full, y_lr1_full)

        # simulation by threshold
        full_by_threshold_dfs_list = self.env.random_obs_simulation_by_threshold(merged_full_dfs, self.threshold_values,
                                                                                 -self.REWARD_ACTUAL_REPLACE,
                                                                                 -self.REWARD_ACTUAL_FAILURE)
        full_by_loss_dfs = self.env.calculate_NoF_AUT_by_threshold(full_by_threshold_dfs_list, self.threshold_values)

        # save the learning results to a global variable
        full_by_loss_dfs_list.append(full_by_loss_dfs)


    def run_TD_loss_simulation(self):
        global test_average_rewards, test_average_usage_times, test_replace_failures
        # 이 method는 학습이 완료된 bestweight을 불러와서 테스트 환경에서 실행.
        # 아마도 continue의 reward에 따른 다양한 RL 학습 결과를 테스트 환경에서 실행 후 하나의 plot에 점을 찍어내야 함.
        # 점을 찍을 때, continue의 reward가 무엇이었는지 같이 표시해주면 좋음. reward에 따른 성능을 볼 수 있도록.
        # Load average_by_loss_dfs from the file
        with open('average_by_loss_dfs.pkl', 'rb') as f:
            average_by_loss_dfs = pickle.load(f)
        self.env.plot_simulation_results_scale_up(average_by_loss_dfs, self.num_dataset, self.loss_labels, False)
        with open('LR_TD_weight_by_RL_code.pkl', 'rb') as f:
            self.agent.lr_best_weights['continue'] = pickle.load(f)
        self.agent.lr_best_weights['replace'] = np.zeros(22) # constant 항까지 포함됨. 그래서 21차원이 아니라 22차원.
        print(self.agent.lr_best_weights)

        for i in range(self.num_sample_datasets):
            self.test_TD_loss_random_observation(i)

        print(test_average_usage_times)
        print(self.num_sample_datasets)

        test_average_reward = sum(test_average_rewards) / self.num_sample_datasets
        test_average_actual_reward = sum(test_average_actual_rewards) / self.num_sample_datasets
        test_average_usage_time = sum(test_average_usage_times) / self.num_sample_datasets
        test_replace_failure = sum(test_replace_failures) / self.num_sample_datasets
        failure_probability = test_replace_failure / 100
        lambda_policy = (-self.REWARD_ACTUAL_REPLACE + failure_probability * (-self.REWARD_ACTUAL_FAILURE + self.REWARD_ACTUAL_REPLACE)) / test_average_usage_time
        beta_policy = lambda_policy / (-self.REWARD_ACTUAL_FAILURE + self.REWARD_ACTUAL_REPLACE)

        print(
            f" Test average reward : {test_average_reward}, Test actual average reward : {test_average_actual_reward},"
            f" Test replace failure : {test_replace_failure}, Test average usage time : {test_average_usage_time}, Failure probability : {failure_probability}, lambda : {lambda_policy}, beta : {beta_policy}")

        self.env.plot_RL_results_scale_up(average_by_loss_dfs, self.num_dataset, self.loss_labels,
                                          test_replace_failure, test_average_usage_time, self.CONTINUE_COST)

    def run_TD_loss_simulation_21(self):
        global test_average_rewards, test_average_usage_times, test_replace_failures
        # 이 method는 학습이 완료된 bestweight을 불러와서 테스트 환경에서 실행.
        # 아마도 continue의 reward에 따른 다양한 RL 학습 결과를 테스트 환경에서 실행 후 하나의 plot에 점을 찍어내야 함.
        # 점을 찍을 때, continue의 reward가 무엇이었는지 같이 표시해주면 좋음. reward에 따른 성능을 볼 수 있도록.
        # Load average_by_loss_dfs from the file
        with open('average_by_loss_dfs.pkl', 'rb') as f:
            average_by_loss_dfs = pickle.load(f)
        self.env.plot_simulation_results_scale_up(average_by_loss_dfs, self.num_dataset, self.loss_labels, False)
        with open('LR_TD_weight_by_RL_code_alpha_09.pkl', 'rb') as f:
            self.agent.lr_best_weights_21['continue'] = pickle.load(f)
        self.agent.lr_best_weights_21['replace'] = np.zeros(21) # constant 없이 21차원.
        print(self.agent.lr_best_weights)

        for i in range(self.num_sample_datasets):
            self.test_TD_loss_random_observation_21(i)

        print(test_average_usage_times)
        print(self.num_sample_datasets)

        test_average_reward = sum(test_average_rewards) / self.num_sample_datasets
        test_average_actual_reward = sum(test_average_actual_rewards) / self.num_sample_datasets
        test_average_usage_time = sum(test_average_usage_times) / self.num_sample_datasets
        test_replace_failure = sum(test_replace_failures) / self.num_sample_datasets
        failure_probability = test_replace_failure / 100
        lambda_policy = (-self.REWARD_ACTUAL_REPLACE + failure_probability * (-self.REWARD_ACTUAL_FAILURE + self.REWARD_ACTUAL_REPLACE)) / test_average_usage_time
        beta_policy = lambda_policy / (-self.REWARD_ACTUAL_FAILURE + self.REWARD_ACTUAL_REPLACE)

        print(
            f" Test average reward : {test_average_reward}, Test actual average reward : {test_average_actual_reward},"
            f" Test replace failure : {test_replace_failure}, Test average usage time : {test_average_usage_time}, Failure probability : {failure_probability}, lambda : {lambda_policy}, beta : {beta_policy}")

        self.env.plot_RL_results_scale_up(average_by_loss_dfs, self.num_dataset, self.loss_labels,
                                          test_replace_failure, test_average_usage_time, self.CONTINUE_COST)

    def run_many(self):
        # Iterate over the number of sample datasets
        for i in range(self.num_sample_datasets):
            # Perform simulation for the current sample dataset index
            self.run_lr_simulation(i)
            print(f"Completed sample index: {i + 1} out of {self.num_sample_datasets}")

        # calculate average performance
        average_by_loss_dfs = self.env.calculate_average_performance(full_by_loss_dfs_list, self.num_sample_datasets)

        # Save average_by_loss_dfs to a file using pickle
        with open('average_by_loss_dfs.pkl', 'wb') as f:
            pickle.dump(average_by_loss_dfs, f)

        self.env.plot_simulation_results_scale_up(average_by_loss_dfs, self.num_dataset, self.loss_labels)

    def run_many_only_MSE(self):
        # Iterate over the number of sample datasets
        for i in range(self.num_sample_datasets):
            # Perform simulation for the current sample dataset index
            self.run_lr_simulation_only_MSE(i)
            print(f"Completed sample index: {i + 1} out of {self.num_sample_datasets}")

        # calculate average performance
        average_by_loss_dfs = self.env.calculate_average_performance_only_MSE(full_by_loss_dfs_list, self.num_sample_datasets)

        # Save average_by_loss_dfs to a file using pickle
        with open('average_by_loss_dfs_only_MSE.pkl', 'wb') as f:
            pickle.dump(average_by_loss_dfs, f)

        self.env.plot_simulation_results_scale_up(average_by_loss_dfs, self.num_dataset, self.loss_labels, True) # True : only MSE

    def plot_results(self):
        # 여러가지 plot들 한번에 출력하는 코드.
        global test_average_rewards, test_average_usage_times, test_replace_failures
        with open('average_by_loss_dfs_only_MSE.pkl', 'rb') as f:
            average_by_loss_dfs = pickle.load(f)
        self.env.plot_simulation_results(average_by_loss_dfs , self.num_dataset)
        #self.env.plot_simulation_results_scale_up(average_by_loss_dfs, self.num_dataset, self.loss_labels)
        self.env.plot_simulation_results_x_y_swap(average_by_loss_dfs, self.num_dataset, self.loss_labels, 100)
        self.env.plot_simulation_results_x_y_swap_cost(average_by_loss_dfs, self.num_dataset, self.loss_labels, 100, -self.REWARD_ACTUAL_REPLACE, -self.REWARD_ACTUAL_FAILURE)
        self.env.plot_simulation_results_x_y_swap_cost_scale_up(average_by_loss_dfs, self.num_dataset, self.loss_labels, 100, -self.REWARD_ACTUAL_REPLACE, -self.REWARD_ACTUAL_FAILURE)
        # AUT_Pi, P_failure는 위의 method에서 return하고 저장 후, 아래 method로 전달하자. 지금은 임시로 값을 직접 넣어둠.
        self.env.plot_simulation_results_x_y_swap_point_lambda_2(average_by_loss_dfs, self.num_dataset, self.loss_labels, 100)
        self.env.plot_simulation_results_x_y_swap_point_td_loss(average_by_loss_dfs, self.num_dataset, self.loss_labels, 100)



"""generate instance"""
run_sim = RunSimulation('config_009.ini')
#run_sim_1 = RunSimulation('config_004.ini')
""" ################################
Linear Regression Simulation
"""
#run_sim.run_many()
#run_sim.plot_results()

# MSE만 사용해서 LR 시뮬레이션 하는 코드. 다른 loss function들은 이제 필요 없음.
#run_sim.run_many_only_MSE()
run_sim.plot_lr_td_loss_to_RUL_all_samples_21(1) # RUl prediction plot
#run_sim.plot_results() # plot 그리는 코드 (퍼포먼스 비교용)

""" #################################
Reinforcement Learning (value-based)
"""
# Weights를 처음 학습시킬때만 실행.
#run_sim.train_many_off_policy_RL()

# 학습된 weights를 바탕으로 이어서 학습하기 위한 method.
#run_sim_1.train_many_by_saved_weights_off_policy_RL()
#run_sim.train_many_by_saved_weights_off_policy_RL()

# 저장된 weights으로 전체 엔진에 대한 test 수행.
#run_sim.run_RL_simulation()
#run_sim_1.run_RL_simulation()


""" RL 코드를 기반으로 한, TD loss로 Linear regression 학습. """
#run_sim.train_many_lr_by_td_loss()
#run_sim.train_continue_many_lr_by_td_loss()   # 이미 학습된 weight을 이어서 학습시킬 때 사용.
#run_sim.run_TD_loss_simulation()              # 학습 결과 시뮬레이션.

# 21차원으로 학습시키는 코드, 테스트
#run_sim.train_many_lr_by_td_loss_21()
#run_sim.run_TD_loss_simulation_21()


""" RUL prediction by Q-value"""
"""
beta_C_scale = 1 / 12.87
average_reward_scale = 1 / 11.2902
print(beta_C_scale)
#print(average_reward_scale)
run_sim.plot_Q_value_to_RUL_all_samples(beta_C_scale)
#run_sim.plot_Q_value_to_RUL_all_samples(average_reward_scale)
"""

# TD Loss 학습 코드.
"""
run_sim.train_many_lr_by_DA_TD_loss()
#run_sim.train_continue_many_lr_by_DA_td_loss()

#run_sim.calaulate_solution_of_lr_DA_TD_loss(80)  # lambda (ridge)
run_sim.run_TD_loss_simulation()
run_sim.plot_lr_td_loss_to_RUL_all_samples(1)
"""

