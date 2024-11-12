# General lib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import configparser

# for deep learning
import tensorflow as tf
from tensorflow.keras import layers, models, initializers
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
import pandas as pd
import pickle  # it is suitable for saving Python objects.
import warnings
from copy import copy
import random
import matplotlib.pyplot as plt
import torch
import gc

# Custom .py
#from linear_regression_TD import Linear_Regression_TD
from simulation_env import SimulationEnvironment
from loss import directed_mse_loss
from loss_td import DecisionAwareTD
from RL_component import Environment
from RL_component import Rewards
from RL_component import Agent

# DCNN
from DeepCNN import DCNN, DCNN_Model

# Filter out the warning
warnings.filterwarnings("ignore",
                        message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*")

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
test_average_actual_rewards = []  # for average performance.
test_average_usage_times = []
test_replace_failures = []

# For test (2024.10.02; Fixed theta)
prediction_loss = 0
decision_loss = 0


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
        self.td_weight_21 = np.random.normal(loc=0, scale=0.5, size=21)  # RL과 weight 크기 맞춰서 비교하기 위함.

        self.td_simulation_threshold = float(config['SimulationSettings']['td_simulation_threshold'])
        self.is_crucial_moment = config.getboolean('SimulationSettings', 'is_crucial_moment')
        self.is_td_front = config.getboolean('SimulationSettings', 'is_td_front')

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

        # Hyperparameter for reinforcement learning ###########################################################
        self.gamma = float(config['RL_Settings']['discount_factor'])
        self.alpha = float(config['RL_Settings']['learning_rate'])
        self.initial_epsilon = float(config['RL_Settings']['initial_epsilon'])
        self.epsilon_delta = float(config['RL_Settings']['epsilon_delta'])
        self.min_epsilon = float(config['RL_Settings']['min_epsilon'])
        self.max_episodes = int(config['RL_Settings']['max_episodes'])
        # RL
        self.columns_to_scale = ['s_1', 's_2', 's_3', 's_4', 's_5', 's_6', 's_7', 's_8', 's_9', 's_10', 's_11', 's_12',
                                 's_13', 's_14', 's_15', 's_16', 's_17', 's_18', 's_19', 's_20', 's_21']
        self.best_average_reward = -10000  # 처음에만 이렇게 초기화하고, 그 다음에는 알아서 반영.
        ########################################################################################################

        # Hyperparameter for DCNN ################################################################################
        self.DCNN_N_tw = int(config['DCNN_Settings']['N_tw']) # Time sequence length (Time window)
        self.DCNN_N_ft = int(config['DCNN_Settings']['N_ft']) # Number of features
        self.DCNN_F_N = int(config['DCNN_Settings']['F_N'])   # Number of filters (each convolution layer)
        self.DCNN_F_L = int(config['DCNN_Settings']['F_L'])   # Filter size (1D filter)
        self.DCNN_batch_size = int(config['DCNN_Settings']['batch_size'])
        self.DCNN_neurons_fc = int(config['DCNN_Settings']['neurons_fc']) # Neurons in fully-connected layer
        self.DCNN_dropout_rate = float(config['DCNN_Settings']['dropout_rate'])
        self.DCNN_epochs = int(config['DCNN_Settings']['epochs'])

        self.DCNN_is_fully_observe = config.getboolean('DCNN_Settings', 'is_fully_observe')
        self.DCNN_is_td_loss = config.getboolean('DCNN_Settings', 'is_td_loss')
        ########################################################################################################


        # 사전에 정의된 stopping time에 따른 exploration을 위한 parameter
        self.min_t_replace = int(config['StoppingTime']['min_t_replace'])
        self.max_t_replace = int(config['StoppingTime']['max_t_replace'])  # 10%의 데이터만 있다는 것을 감안해서 원래 값보다 1/10 수준으로 유지

        # class instance 생성
        self.env = SimulationEnvironment()
        self.agent = Agent()  # RL
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

        DA_TD.preprocessing()  # 여기에 scaling 후 s_0에 1을 초기화 (LR의 상수항) 하는 코드 포함. 따라서 test시에는 s_0 추가.
        # print(self.td_weight)

        gradient = DA_TD.calculate_gradient()  # prediction loss, decision loss가, alpha로 같이 들어가있음.
        # gradient = DA_TD.calculate_gradient_only_TD() # prediction accuracy term 무시하고 decision loss만 사용
        self.td_weight = self.td_weight - learning_rate * gradient
        # print(self.td_weight)

        # loss = ((self.td_weight.T.dot(DA_TD.X_t.T) - DA_TD.Y.T) ** 2).values.sum()
        # lr_td_training_loss.append(loss)

        lr_td_training_loss.append((gradient ** 2).values.sum())

    def calaulate_solution_of_lr_DA_TD_loss(self, lambd):
        # gradient를 구하지 않고 closed-form solution을 이용해 weight 계산 (24.06.26 시점에선 무의미. max가 있어 closed-form을 구할 수 없음)
        # 모든 샘플 데이터를 하나의 데이터로 합쳐서 수행
        combined_train_data = pd.DataFrame()

        for i in range(self.num_sample_datasets):
            sample_data = self.sampled_datasets_with_RUL[i][0].copy()
            combined_train_data = pd.concat([combined_train_data, sample_data], ignore_index=True)

        combined_train_data[self.columns_to_scale] = combined_train_data[self.columns_to_scale].apply(
            self.env.min_max_scaling, axis=0)
        combined_train_data.reset_index(drop=True, inplace=True)  # index reset

        DA_TD = DecisionAwareTD(combined_train_data, self.td_beta, self.td_weight, self.td_alpha)
        DA_TD.preprocessing()

        self.td_weight = DA_TD.calculate_closed_form_solution(lambd)
        # self.td_weight = DA_TD.calculate_closed_form_solution_ratio(lambd)

        # test
        combined_train_data.insert(loc=combined_train_data.columns.get_loc('s_1'), column='s_0', value=1)
        selected_data = combined_train_data.iloc[:, 5:27]
        print(selected_data)
        print(selected_data.dot(self.td_weight))

        # Save weights to a file using pickle
        with open('pkl_file/LR_TD_weight_test.pkl', 'wb') as f:
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
        with open('pkl_file/LR_TD_weight_test.pkl', 'wb') as f:
            pickle.dump(self.td_weight, f)

    def train_continue_many_lr_by_DA_td_loss(self):
        # 학습된 weight을 이어서 TD loss로 학습
        with open('pkl_file/LR_TD_weight_test.pkl', 'rb') as f:
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
        with open('pkl_file/LR_TD_weight_test.pkl', 'wb') as f:
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
                # print(WX_t)
                X_diff = state - RL_env.states.iloc[state_index - 1].values  # X_t - X_{t-1}
                # print(X_diff)

                # 엔진 내의 마지막 time_step이면 td를 (1/beta)로 gradient를 업데이트.
                if current_reward == (self.reward.r_continue_but_failure):
                    gradient = 2 * (WX_t - RL_env.environment['RUL'].iloc[state_index]) * current_state + \
                               2 * alpha * (np.dot(self.agent.lr_weights_by_td, X_diff) - (1 / beta)) * X_diff
                # 마지막 time_step이 아닌 경우에는 td를 -1로 gradient update
                else:
                    gradient = 2 * (WX_t - RL_env.environment['RUL'].iloc[state_index]) * current_state + \
                               2 * alpha * (np.dot(self.agent.lr_weights_by_td, X_diff) + 1) * X_diff

                sum_of_gradient += gradient
                # print(gradient)
                # print(sum_of_gradient)

                # 다음 상태로 이동
                state_index = next_state_index
                state = RL_env.states.iloc[state_index].values
                num_of_step += 1

        mean_sum_of_gradient = sum_of_gradient / num_of_step
        # print(mean_sum_of_gradient)
        self.agent.update_lr_weights_by_gradient(mean_sum_of_gradient, learning_rate)

        lr_td_training_loss.append(np.dot(mean_sum_of_gradient, mean_sum_of_gradient))

    def train_lr_by_td_loss_w_theta(self, data_sample_index, epoch, alpha, beta, learning_rate):
        # random observation임을 고려해서 td가 실제 흘러간 타임스탭만큼 들어가는 코드.
        # Q-learning (off-policy TD(0)와 equivalent한 update)
        # 이 코드는 RL처럼 매 스탭마다 weight을 업데이트( 함. (action은 계속 continue 하는 버전)
        # threshold를 constant가 아닌 학습 대상으로 변경.
        # 2024.07.29 threshold도 gradient로 함께 학습되도록 함.
        state_index = 0
        num_of_step = 0
        sum_of_gradient = 0
        loss_epoch = 0

        train_data = self.sampled_datasets_with_RUL[data_sample_index][0].copy()
        train_data[self.columns_to_scale] = train_data[self.columns_to_scale].apply(self.env.min_max_scaling, axis=0)

        # crucial moment 적용을 위해 RUL이 crucial moment보다 적게 남은 데이터만 필터링함.
        # default = False 임. 학습된 theta가 어떻게 바뀌는지 실험하기 위해 넣은 기능.
        if self.is_crucial_moment == True :
            train_data = train_data[train_data['RUL'] <= self.td_crucial_moment]

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
                next_state = RL_env.lr_states.iloc[next_state_index].values  # max operator 안에 들어감

                # 리워드를 통해 엔진이 바뀌는 것을 알 수 있음.
                current_reward = self.reward.get_reward(state_index, next_state_index, chosen_action,
                                                        RL_env.environment)

                WX_t = np.dot(self.agent.lr_weights_by_td, current_state)  # w * x_t
                WX_t_1 = np.dot(self.agent.lr_weights_by_td, next_state)  # w * x_{t+1}

                # t = tau_i 일 때는 다음과 같이 gradient를 업데이트 (time cycle이 엔진 내의 마지막 time cycle일 때. 즉 continue 하면 failure 하는 상태)
                if current_reward == (self.reward.r_continue_but_failure):
                    """
                    # decision term에 theta를 넣었을 때의 gradient (TD loss - back; original)
                    # weights의 gradient
                    gradient = -2 * (1 - self.td_alpha) * (RL_env.environment['RUL'].iloc[
                                                               state_index] - WX_t) * current_state - 2 * self.td_alpha * (
                                       -(1 / self.td_beta) - WX_t + self.agent.theta) * current_state
                    # theta의 gradient
                    gradient_theta = 2 * self.td_alpha * (-(1 / self.td_beta) - WX_t + self.agent.theta)
                    """

                    # prediction term에 theta를 넣었을 때의 gradient
                    # weights의 gradient
                    gradient = -2 * (1 - self.td_alpha) * (RL_env.environment['RUL'].iloc[
                                                               state_index] - WX_t - self.agent.theta) * current_state - 2 * self.td_alpha * (
                                       -(1 / self.td_beta) - WX_t) * current_state
                    # theta의 gradient
                    gradient_theta = -2 * (1 - self.td_alpha) * (RL_env.environment['RUL'].iloc[
                                                                     state_index] - WX_t - self.agent.theta)

                    # update w, theta
                    self.agent.update_lr_weights_by_gradient(gradient, learning_rate)  # gradient descent for w
                    self.agent.update_theta(gradient_theta, learning_rate)  # gradient descent for theta

                else:
                    time_difference = RL_env.environment['time_cycles'].iloc[next_state_index] - \
                                      RL_env.environment['time_cycles'].iloc[state_index]

                    """
                    # decision term에 theta를 넣었을 때의 gradient (TD loss - back; original)
                    # weights의 gradient
                    gradient = -2 * (1 - self.td_alpha) * (RL_env.environment['RUL'].iloc[
                                                               state_index] - WX_t) * current_state - 2 * self.td_alpha * (
                                       time_difference + max(WX_t_1 - self.agent.theta,
                                                             0) - WX_t + self.agent.theta) * current_state
                    # theta의 gradient
                    gradient_theta = 2 * self.td_alpha * (
                                time_difference + max(WX_t_1 - self.agent.theta, 0) - WX_t + self.agent.theta)
                    # 교수님 코드 (max 안에 있는 y^도 미분한다고 가정했을 때. 예전에 실험해본 결과 큰 차이 x)
                    #gradient_theta = 2 * self.td_alpha * (
                    #        time_difference + self.agent.theta - WX_t) if WX_t_1 < self.agent.theta else 0
                    """

                    # prediction term에 theta를 넣었을 때의 gradient
                    # weights의 gradient
                    gradient = -2 * (1 - self.td_alpha) * (RL_env.environment['RUL'].iloc[
                                                               state_index] - WX_t - self.agent.theta) * current_state - 2 * self.td_alpha * (
                                       time_difference + max(WX_t_1, 0) - WX_t) * current_state

                    # theta의 gradient
                    gradient_theta = -2 * (1 - self.td_alpha) * (RL_env.environment['RUL'].iloc[
                                                                     state_index] - WX_t - self.agent.theta)


                    # update w, theta
                    self.agent.update_lr_weights_by_gradient(gradient, learning_rate)  # gradient descent for w
                    self.agent.update_theta(gradient_theta, learning_rate)  # gradient descent for theta
                """
                # theta의 gradient를 구할때, w처럼 상수 취급을 하지 않고 max를 푼 버전.
                else:
                    time_difference = RL_env.environment['time_cycles'].iloc[next_state_index] - \
                                      RL_env.environment['time_cycles'].iloc[state_index]

                    # weights의 gradient
                    gradient = -2 * (1 - self.td_alpha) * (RL_env.environment['RUL'].iloc[
                                                               state_index] - WX_t) * current_state - 2 * self.td_alpha * (
                                       time_difference + max(WX_t_1 - self.agent.theta,
                                                             0) - WX_t + self.agent.theta) * current_state

                    # theta의 gradient (max operator를 풀기 위해 범위를 나눔)
                    if WX_t_1 - self.agent.theta < 0:
                        gradient_theta = 2 * self.td_alpha * (time_difference - WX_t + self.agent.theta)
                    else:
                        gradient_theta = 0

                    # update w, theta
                    self.agent.update_lr_weights_by_gradient(gradient, learning_rate)  # gradient descent for w
                    self.agent.update_theta(gradient_theta, learning_rate)  # gradient descent for theta
                """

                # 다음 상태로 이동
                state_index = next_state_index

                # 22차원용 코드
                state = RL_env.lr_states.iloc[state_index].values

                num_of_step += 1

                # 그냥 gradient의 크기가 얼마나 줄어드나 확인하기 위한 용도 (학습과는 상관 없음)
                sum_of_gradient += gradient

        # 학습이 잘 되고 있는지 loss를 확인하기 위한 코드 (TD loss)
        mean_sum_of_gradient = sum_of_gradient / num_of_step
        lr_td_training_loss.append(np.dot(mean_sum_of_gradient, mean_sum_of_gradient))

    def train_lr_by_td_loss_random_observation(self, data_sample_index, epoch, alpha, beta, learning_rate):
        # random observation임을 고려해서 td가 실제 흘러간 타임스탭만큼 들어가는 코드.
        # Q-learning (off-policy TD(0)와 equivalent한 update)
        # 이 코드는 RL처럼 매 스탭마다 weight을 업데이트( 함. (action은 계속 continue 하는 버전)
        # 2024.07.16 새롭게 정의한 td loss로 수정. (threshold 반영)
        # config에서 threshold = 0으로 해두면 사실상 결과는 같음.
        state_index = 0
        num_of_step = 0
        sum_of_gradient = 0
        loss_epoch = 0

        train_data = self.sampled_datasets_with_RUL[data_sample_index][0].copy()
        train_data[self.columns_to_scale] = train_data[self.columns_to_scale].apply(self.env.min_max_scaling, axis=0)

        # crucial moment 적용을 위해 RUL이 crucial moment보다 적게 남은 데이터만 필터링함.
        # default = False 임. 학습된 theta가 어떻게 바뀌는지 실험하기 위해 넣은 기능.
        if self.is_crucial_moment == True:
            train_data = train_data[train_data['RUL'] <= self.td_crucial_moment]


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
                next_state = RL_env.lr_states.iloc[next_state_index].values  # max operator 안에 들어감

                # 리워드를 통해 엔진이 바뀌는 것을 알 수 있음.
                current_reward = self.reward.get_reward(state_index, next_state_index, chosen_action,
                                                        RL_env.environment)

                WX_t = np.dot(self.agent.lr_weights_by_td, current_state)  # w * x_t
                WX_t_1 = np.dot(self.agent.lr_weights_by_td, next_state)  # w * x_{t+1}

                # t = tau_i 일 때는 다음과 같이 gradient를 업데이트 (time cycle이 엔진 내의 마지막 time cycle일 때. 즉 continue 하면 failure 하는 상태)
                if current_reward == (self.reward.r_continue_but_failure):
                    # gradient = -2 * (1 - self.td_alpha) * (RL_env.environment['RUL'].iloc[
                    #                                           state_index] - WX_t) * current_state - 2 * self.td_alpha * (
                    #                       -(1 / self.td_beta) - WX_t) * current_state
                    gradient = -2 * (1 - self.td_alpha) * (RL_env.environment['RUL'].iloc[
                                                               state_index] - WX_t) * current_state - 2 * self.td_alpha * (
                                       -(1 / self.td_beta) - WX_t + self.td_simulation_threshold) * current_state

                    self.agent.update_lr_weights_by_gradient(gradient, learning_rate)  # gradient descent



                # t가 '1 <= t < tau_i' 인 경우에는 아래와 같이 gradient를 업데이트 (엔진 내에서 마지막 time cycle이 아닐 때)
                else:
                    time_difference = RL_env.environment['time_cycles'].iloc[next_state_index] - \
                                      RL_env.environment['time_cycles'].iloc[state_index]

                    # gradient = -2 * (1 - self.td_alpha) * (RL_env.environment['RUL'].iloc[
                    #                                           state_index] - WX_t) * current_state - 2 * self.td_alpha * (
                    #                       time_difference + max(WX_t_1, 0) - WX_t) * current_state

                    gradient = -2 * (1 - self.td_alpha) * (RL_env.environment['RUL'].iloc[
                                                               state_index] - WX_t) * current_state - 2 * self.td_alpha * (
                                       time_difference + max(WX_t_1 - self.td_simulation_threshold,
                                                             0) - WX_t + self.td_simulation_threshold) * current_state
                    self.agent.update_lr_weights_by_gradient(gradient, learning_rate)  # gradient descent

                # 다음 상태로 이동
                state_index = next_state_index

                # 22차원용 코드
                state = RL_env.lr_states.iloc[state_index].values

                num_of_step += 1

                # 그냥 gradient의 크기가 얼마나 줄어드나 확인하기 위한 용도 (학습과는 상관 없음)
                sum_of_gradient += gradient

        # 학습이 잘 되고 있는지 loss를 확인하기 위한 코드 (TD loss)
        mean_sum_of_gradient = sum_of_gradient / num_of_step
        lr_td_training_loss.append(np.dot(mean_sum_of_gradient, mean_sum_of_gradient))

    def train_many_lr_by_td_loss_theta(self):  # 샘플 데이터셋 전체를 하나의 epoch로 취급.
        # RL학습시키는데 사용한 코드로 td loss를 반복 학습하는 코드 (처음 학습시킬 때 사용)
        # threshold (theta)도 gradient로 학습하도록 하는 method.

        # Iterate over the number of sample datasets
        for epoch in range(self.max_epoch):
            print(f"Epoch: {epoch + 1}, Theta: {self.agent.theta}")

            # Iterate over the number of sample datasets
            for i in range(self.num_sample_datasets):
                # random observation으로 학습.
                self.train_lr_by_td_loss_w_theta(i, epoch, self.td_alpha, self.td_beta,
                                                 self.td_learning_rate)

        self.env.plot_training_loss(self.max_epoch, self.num_sample_datasets, lr_td_training_loss)

        # Create a dictionary to store both lr_weights_by_td and theta
        save_data = {
            'lr_weights_by_td': self.agent.lr_weights_by_td,
            'theta': self.agent.theta
        }

        # Save the dictionary to a file using pickle
        with open('LR_TD_weight_and_theta.pkl', 'wb') as f:
            pickle.dump(save_data, f)

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
                self.train_lr_by_td_loss_random_observation(i, epoch, self.td_alpha, self.td_beta,
                                                            self.td_learning_rate)

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

        RL_env = Environment(train_data)  # 인스턴스 생성.

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

    def train_RL_new(self, data_sample_index, epsilon, episode):  # method 호출 당, 전체 엔진에 대해 학습이 진행됨.
        # 이 코드도 특정 t시점을 미리 정해두고 그떄까지는 쭉 exploration 하는 코드임. off-policy 설정이 안들어가있음.
        # behavior policy와, target policy 모두 exploration을 하는 코드고 시간이 지날수록 greedy하게 학습되도록 해둠.
        replace_failure = 0
        state_index = 0  # index를 가리키는 pointer로 사용 (episode 마다 초기화)
        total_reward = 0
        total_actual_reward = 0
        num_of_step = 0
        loss_episode = 0

        train_data = self.sampled_datasets_with_RUL[data_sample_index][0].copy()
        train_data[self.columns_to_scale] = train_data[self.columns_to_scale].apply(self.env.min_max_scaling, axis=0)
        train_data.reset_index(drop=True, inplace=True)  # index reset (reset 하지 않으면 state 전이가 되지 않음)

        RL_env = Environment(train_data)

        for unit_num in range(RL_env.max_unit_number):  # unit num : 0, ... , (max_unit_number - 1)
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

                    current_reward = self.reward.get_reward(state_index, next_state_index, chosen_action,
                                                            RL_env.environment)

                    # count 'replace failure'
                    if current_reward == (self.reward.r_continue_but_failure):
                        replace_failure += 1

                    # next action
                    next_chosen_action = 'continue' if next_state_index < RL_env.stateMinIndex(
                        next_state_index) + t_replace else 'replace'

                    # update q-value (Linear Function Approximation)
                    next_state_q = np.dot(self.agent.weights[next_chosen_action],
                                          next_state)  # A' ~ random generated episode
                    current_state_q = np.dot(self.agent.weights[chosen_action],
                                             current_state)  # A  ~ random generated episode

                    # TD target, weight
                    TD_target = current_reward + self.gamma * next_state_q
                    delta_w = self.alpha * (TD_target - current_state_q) * current_state  # current state -> gradient

                    # update weights
                    # self.agent.weights[chosen_action] = self.agent.weights[chosen_action] + delta_w

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
                    current_reward = self.reward.get_reward(state_index, next_state_index, chosen_action,
                                                            RL_env.environment)

                    # count 'replace failure'
                    if current_reward == (self.reward.r_continue_but_failure):
                        replace_failure += 1

                    # update q-value (Linear Function Approximation)
                    next_state_q = max(
                        [np.dot(self.agent.weights[a], next_state) for a in self.agent.actions])  # A' ~ greedy action
                    current_state_q = np.dot(self.agent.weights[chosen_action], current_state)  # A  ~ greedy action

                    TD_target = current_reward + self.gamma * next_state_q
                    delta_w = self.alpha * (TD_target - current_state_q) * current_state

                    # self.agent.weights[chosen_action] = self.agent.weights[chosen_action] + delta_w

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

        train_data.reset_index(drop=True, inplace=True)  # index reset. (리셋하지 않으면 state 전이가 되지 않음)

        RL_env = Environment(train_data)

        for unit_num in range(RL_env.max_unit_number):  # unit num : 0, .... , (max_unit_number - 1)
            state = RL_env.states.iloc[state_index].values

            # random stopping time을 위한 조건 분기 (exploration)
            if np.random.rand() < epsilon:
                t_replace = random.randint(self.min_t_replace, self.max_t_replace)  # 하나의 unit에 대해서만 t_replace를 뽑음.
                while (state_index < RL_env.environment[
                    RL_env.environment['unit_number'] == (RL_env.environment['unit_number'].max() - 2)].index[-1] + 1) \
                        and RL_env.environment['unit_number'].iloc[state_index] == (unit_num + 1):
                    current_state = state
                    chosen_action = 'continue' if state_index < RL_env.stateMinIndex(
                        state_index) + t_replace else 'replace'

                    next_state_index = RL_env.nextStateIndex(chosen_action, state_index)
                    next_state = RL_env.states.iloc[next_state_index].values

                    current_reward = self.reward.get_reward(state_index, next_state_index, chosen_action,
                                                            RL_env.environment)

                    # count 'replace failure'
                    if current_reward == (self.reward.r_continue_but_failure):
                        replace_failure += 1

                    # next action
                    next_chosen_action = 'continue' if next_state_index < RL_env.stateMinIndex(
                        next_state_index) + t_replace else 'replace'

                    # update q-value (Linear Function Approximation)
                    next_state_q = np.dot(self.agent.weights[next_chosen_action],
                                          next_state)  # A' ~ random generated episode
                    current_state_q = np.dot(self.agent.weights[chosen_action],
                                             current_state)  # A  ~ random generated episode

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
                       RL_env.environment[
                           RL_env.environment['unit_number'] == (RL_env.environment['unit_number'].max() - 2)].index[
                           -1] + 1) and RL_env.environment['unit_number'].iloc[state_index] == (unit_num + 1):
                    current_state = state
                    chosen_action = max(self.agent.actions,
                                        key=lambda a: np.dot(self.agent.weights[a], current_state))  # greedy action
                    next_state_index = RL_env.nextStateIndex(chosen_action, state_index)
                    next_state = RL_env.states.iloc[next_state_index].values
                    current_reward = self.reward.get_reward(state_index, next_state_index, chosen_action,
                                                            RL_env.environment)

                    # count 'replace failure'
                    if current_reward == (self.reward.r_continue_but_failure):
                        replace_failure += 1

                    # update q-value (Linear Function Approximation)
                    next_state_q = max(
                        [np.dot(self.agent.weights[a], next_state) for a in self.agent.actions])  # A' ~ greedy action
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
        # average_number_of_observation = number_of_observation / RL_env.max_unit_number

        # best weight 저장
        if average_reward > self.best_average_reward:
            self.best_average_reward = copy(average_reward)
            self.agent.save_best_weights(self.agent.get_weights())  # 이 method를 그대로 활용하려면 위에서 weight을 agent에 save 해뒀어야함.

        average_rewards.append(average_reward)
        training_loss.append(np.abs(loss_episode))
        # average_number_of_observations.append(average_number_of_observation)
        print(
            f"episode : {episode + 1}, replace failure : {replace_failure}, Average Reward : {average_reward}, "
            f"loss : {np.abs(loss_episode)}")

    def train_many_off_policy_RL(self):  # 샘플 데이터셋 전체를 하나의 episode로 취급.
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
        with open('pkl_file/RL_best_weights_continue_240425.pkl', 'wb') as f:
            pickle.dump(self.agent.get_best_weights(), f)

    def train_many_by_saved_weights_off_policy_RL(self):  # 샘플 데이터셋 전체를 하나의 episode로 취급.
        # 사전에 학습된 weights으로 이어서 학습하기 위한 method.
        with open('pkl_file/RL_best_weights_continue_240425.pkl', 'rb') as f:
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
        with open('pkl_file/RL_best_weights_continue_240425.pkl', 'wb') as f:
            pickle.dump(self.agent.get_best_weights(), f)

    def train_many_by_saved_weights_RL(self):  # 샘플 데이터셋 전체를 하나의 episode로 취급.
        # 사전에 학습된 weights으로 이어서 학습하기 위한 method.
        with open('pkl_file/RL_best_weights_continue_240425.pkl', 'rb') as f:
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
        with open('pkl_file/RL_best_weights_continue_240425.pkl', 'wb') as f:
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
        # self.env.plot_number_of_observation(self.max_episodes, average_number_of_observations)

        # Save RL_best_weights to a file using pickle
        with open('RL_best_weights_continue_0.pkl', 'wb') as f:
            pickle.dump(self.agent.get_best_weights(), f)

    def test_RL_random_observation(self, data_sample_index):
        # 매 순간 관측하지 못할 때, 학습된 RL weight으로 test
        global test_average_rewards, test_average_usage_times, test_replace_failures
        replace_failure = 0  # each episode 마다 초기화. 누적시킬 필요는 없음.
        state_index = 0  # state index -> index pointer로 취급하자. (episode 마다 초기화)
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

        full_data.reset_index(drop=True, inplace=True)  # index reset. (리셋하지 않으면 state 전이가 되지 않음)

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
                current_reward = self.reward.get_reward(state_index, next_state_index, chosen_action,
                                                        RL_env.environment)

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

    def test_TD_loss_random_observation(self, data_sample_index, threshold):
        global test_average_rewards, test_average_usage_times, test_replace_failures
        replace_failure = 0  # each episode 마다 초기화. 누적시킬 필요는 없음.
        state_index = 0  # state index -> index pointer로 취급하자. (episode 마다 초기화)
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

        full_data.reset_index(drop=True, inplace=True)  # index reset. (리셋하지 않으면 state 전이가 되지 않음)
        RL_env = Environment(full_data)

        for unit_num in range(RL_env.max_unit_number):  # unit num : 0, .... , (max_unit_number - 1)
            state = RL_env.lr_states.iloc[state_index].values

            while (state_index <
                   RL_env.environment[
                       RL_env.environment['unit_number'] == (RL_env.environment['unit_number'].max() - 2)].index[
                       -1] + 1) and RL_env.environment['unit_number'].iloc[state_index] == (unit_num + 1):
                current_state = state

                # 만약 td_front가 true면 예측한 RUL에 threshold를 더해줌. 즉 퍼포먼스를 q hat으로 보겠다는 의미임.
                if self.is_td_front:
                    threshold_switch = 1 # 1로 설정해서 threshold를 곱하면 그만큼 predicted RUL을 증가시킬 수 있음.
                else:
                    threshold_switch = 0

                if np.dot(self.agent.lr_best_weights['continue'], current_state) + threshold_switch * threshold \
                        > threshold:
                    chosen_action = 'continue'
                else:
                    chosen_action = 'replace'
                next_state_index = RL_env.nextStateIndex(chosen_action, state_index)
                # next_state = RL_env.lr_states.iloc[next_state_index].values
                current_reward = self.reward.get_reward(state_index, next_state_index, chosen_action,
                                                        RL_env.environment)


                if chosen_action == 'replace':
                    print('replace')
                    print(RL_env.environment.iloc[state_index]['time_cycles'])
                    total_operation_time += RL_env.environment.iloc[state_index]['time_cycles']

                # count 'replace failure'
                if current_reward == (self.reward.r_continue_but_failure):
                    print('continue but failure')
                    print(RL_env.environment.iloc[state_index]['time_cycles'])
                    total_operation_time += RL_env.environment.iloc[state_index][
                        'time_cycles']  # 혹시 성능에 차이가 있다면 max time_cycle 쪽 점검
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

    def test_RL(self, data_sample_index):
        global test_average_rewards, test_average_usage_times, test_replace_failures
        replace_failure = 0  # each episode 마다 초기화. 누적시킬 필요는 없음.
        state_index = 0  # state index -> index pointer로 취급하자. (episode 마다 초기화)
        total_reward = 0
        num_of_step = 0
        total_operation_time = 0

        full_data = self.sampled_datasets_with_RUL[data_sample_index][2].copy()
        full_data[self.columns_to_scale] = full_data[self.columns_to_scale].apply(self.env.min_max_scaling, axis=0)
        full_data.reset_index(drop=True, inplace=True)  # index reset. (리셋하지 않으면 state 전이가 되지 않음)

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
                current_reward = self.reward.get_reward(state_index, next_state_index, chosen_action,
                                                        RL_env.environment)

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
        with open('pkl_file/RL_best_weights_continue_240425.pkl', 'rb') as f:
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
        lambda_policy = (-self.REWARD_ACTUAL_REPLACE + failure_probability * (
                    -self.REWARD_ACTUAL_FAILURE + self.REWARD_ACTUAL_REPLACE)) / test_average_usage_time
        beta_policy = lambda_policy / (-self.REWARD_ACTUAL_FAILURE + self.REWARD_ACTUAL_REPLACE)

        print(
            f" Test average reward : {test_average_reward}, Test actual average reward : {test_average_actual_reward},"
            f" Test replace failure : {test_replace_failure}, Test average usage time : {test_average_usage_time}, Failure probability : {failure_probability}, lambda : {lambda_policy}, beta : {beta_policy}")

        # self.env.plot_RL_results_scale_up(average_by_loss_dfs, self.num_dataset, self.loss_labels,
        #                                  test_replace_failure, test_average_usage_time, self.CONTINUE_COST)

    def Q_value_to_RUL(self, data_sample_index, scale):
        # 학습된 continue action에 대한 weights을 이용해 RUL을 예측.
        # replace action에 대한 q-value는 0으로 고정이므로, 이를 threshold로 보면 됨.
        # Q_continue <= Q_replace (0)일 때 replace를 하므로.

        with open('pkl_file/RL_best_weights_beta_00077362686.pkl', 'rb') as f:
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
        # scale = 1이 default.

        # theta가 학습 대상이 아닐 때 code.
        #with open('LR_TD_weight_by_RL_code.pkl', 'rb') as f:
        #    self.td_weight = pickle.load(f)
        #threshold = self.td_simulation_threshold

        # theta도 함께 학습시킬 때 code (theta를 gradient descent로 구할 때).
        with open('LR_TD_weight_and_theta.pkl', 'rb') as f:
            data = pickle.load(f)
        self.td_weight = data['lr_weights_by_td']
        threshold = data['theta']

        full_data = self.sampled_datasets_with_RUL[data_sample_index][2].copy()
        full_data[self.columns_to_scale] = full_data[self.columns_to_scale].apply(self.env.min_max_scaling, axis=0)

        full_data.insert(loc=full_data.columns.get_loc('s_1'), column='s_0', value=1)

        full_data.reset_index(drop=True, inplace=True)
        print("is_td_front")
        print(self.is_td_front)

        self.env.plot_RUL_prediction_by_lr_td_loss(full_data, self.td_weight, scale, threshold, self.is_td_front)
        # self.env.plot_RUL_prediction_by_lr_td_loss(full_data, self.td_weight, scale, 20)

    def plot_lr_td_loss_to_RUL_all_samples(self, scale):
        for i in range(self.num_sample_datasets):
            self.lr_td_loss_to_RUL(i, scale)

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
        # y_train = y_train.clip(upper = 195)

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

        # 4. TD + Crucial moments loss function - Linear Regression (ridge) ##############
        filtered_data = x_train[y_train <= self.td_crucial_moment]
        filtered_labels = y_train[y_train <= self.td_crucial_moment]

        lr4 = Linear_Regression_TD()
        lr4.fit(filtered_data, filtered_labels, 0.5, 10)  # Fitting; fit(X, Y, alpha, lambda)
        y_lr4_train, y_lr4_valid, y_lr4_full = self.env.predict_and_save(lr4, x_train, x_valid, x_full)

        # 5. Directed MSE ################################################################
        y_train_float = y_train.astype(np.float32)  # penalty를 활용할 때 data type을 맞춰주기 위함.
        lr5 = models.Sequential([
            layers.Input(shape=(x_train.shape[1],)),
            layers.Dense(1)  # fully connected layer -> 1 (number of output node)
        ])
        lr5.compile(optimizer='adam', loss=directed_mse_loss)  # model compile
        lr5.fit(x_train, y_train_float, epochs=2500, verbose=0)  # fitting (verbose=2 -> print loss / epoch)
        y_lr5_train, y_lr5_valid, y_lr5_full = self.env.predict_and_save(lr5, x_train, x_valid, x_full)

        # 6. Directed Crucial moments MSE #################################################
        filtered_data = x_train[y_train <= self.directed_crucial_moment]
        filtered_labels = y_train_float[y_train_float <= self.directed_crucial_moment]

        lr6 = models.Sequential([
            layers.Input(shape=(x_train.shape[1],)),
            layers.Dense(1)  # fully connected layer -> 1 (number of output node)
        ])
        lr6.compile(optimizer='adam', loss=directed_mse_loss)  # model compile
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
        # self.env.plot_online_RUL_prediction(merged_full_dfs)

        # simulation by threshold
        full_by_threshold_dfs_list = self.env.random_obs_simulation_by_threshold(merged_full_dfs, self.threshold_values,
                                                                                 -self.REWARD_ACTUAL_REPLACE,
                                                                                 -self.REWARD_ACTUAL_FAILURE)
        full_by_loss_dfs = self.env.calculate_NoF_AUT_by_threshold(full_by_threshold_dfs_list, self.threshold_values)

        # save the learning results to a global variable
        full_by_loss_dfs_list.append(full_by_loss_dfs)

        # self.env.plot_NoF_AUT_by_threshold(full_by_loss_dfs, self.num_dataset)
        # self.env.plot_AC_AUT_by_threshold(full_by_loss_dfs, self.num_dataset)

        # self.env.plot_simulation_results(full_by_loss_dfs, self.num_dataset)
        # self.env.plot_simulation_results_scale_up(full_by_loss_dfs, self.num_dataset, self.loss_labels)

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

    def run_TD_loss_simulation_theta(self, custom_threshold):
        global test_average_rewards, test_average_usage_times, test_replace_failures
        # 이 method는 학습이 완료된 bestweight을 불러와서 테스트 환경에서 실행.
        # 아마도 continue의 reward에 따른 다양한 RL 학습 결과를 테스트 환경에서 실행 후 하나의 plot에 점을 찍어내야 함.
        # 점을 찍을 때, continue의 reward가 무엇이었는지 같이 표시해주면 좋음. reward에 따른 성능을 볼 수 있도록.
        # Load average_by_loss_dfs from the file
        with open('average_by_loss_dfs.pkl', 'rb') as f:
            average_by_loss_dfs = pickle.load(f)
        self.env.plot_simulation_results_scale_up(average_by_loss_dfs, self.num_dataset, self.loss_labels, False)
        with open('LR_TD_weight_and_theta.pkl', 'rb') as f:
            data = pickle.load(f)
        self.agent.lr_best_weights['continue'] = data['lr_weights_by_td']  # continue에 대한 weight.
        self.agent.lr_best_weights['replace'] = np.zeros(22)  # constant 항까지 포함됨 (22-dim)

        if custom_threshold == True:  # 테스트용 theta 사용 (config 파일로)
            self.agent.theta = self.td_simulation_threshold
        elif custom_threshold == False:  # 학습된 theta 사용
            self.agent.theta = data['theta']

        print(self.agent.lr_best_weights)
        print(self.agent.theta)

        for i in range(self.num_sample_datasets):
            self.test_TD_loss_random_observation(i, self.agent.theta)

        print(test_average_usage_times)
        print(self.num_sample_datasets)

        test_average_reward = sum(test_average_rewards) / self.num_sample_datasets
        test_average_actual_reward = sum(test_average_actual_rewards) / self.num_sample_datasets
        test_average_usage_time = sum(test_average_usage_times) / self.num_sample_datasets
        test_replace_failure = sum(test_replace_failures) / self.num_sample_datasets
        failure_probability = test_replace_failure / 100
        lambda_policy = (-self.REWARD_ACTUAL_REPLACE + failure_probability * (
                    -self.REWARD_ACTUAL_FAILURE + self.REWARD_ACTUAL_REPLACE)) / test_average_usage_time
        beta_policy = lambda_policy / (-self.REWARD_ACTUAL_FAILURE + self.REWARD_ACTUAL_REPLACE)

        print(
            f" Test average reward : {test_average_reward}, Test actual average reward : {test_average_actual_reward},"
            f" Test replace failure : {test_replace_failure}, Test average usage time : {test_average_usage_time}, "
            f" Failure probability : {failure_probability}, lambda : {lambda_policy}, beta : {beta_policy}, threshold : {self.agent.theta}")

        self.env.plot_RL_results_scale_up(average_by_loss_dfs, self.num_dataset, self.loss_labels,
                                          test_replace_failure, test_average_usage_time, self.CONTINUE_COST)

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
        self.agent.lr_best_weights['replace'] = np.zeros(22)  # constant 항까지 포함됨. 그래서 21차원이 아니라 22차원.
        print(self.agent.lr_best_weights)

        for i in range(self.num_sample_datasets):
            self.test_TD_loss_random_observation(i, self.td_simulation_threshold)

        print(test_average_usage_times)
        print(self.num_sample_datasets)

        test_average_reward = sum(test_average_rewards) / self.num_sample_datasets
        test_average_actual_reward = sum(test_average_actual_rewards) / self.num_sample_datasets
        test_average_usage_time = sum(test_average_usage_times) / self.num_sample_datasets
        test_replace_failure = sum(test_replace_failures) / self.num_sample_datasets
        failure_probability = test_replace_failure / 100
        lambda_policy = (-self.REWARD_ACTUAL_REPLACE + failure_probability * (
                    -self.REWARD_ACTUAL_FAILURE + self.REWARD_ACTUAL_REPLACE)) / test_average_usage_time
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
        with open('LR_TD_weight_by_RL_code.pkl', 'rb') as f:
            self.agent.lr_best_weights_21['continue'] = pickle.load(f)
        self.agent.lr_best_weights_21['replace'] = np.zeros(21)  # constant 없이 21차원.
        print(self.agent.lr_best_weights)

        for i in range(self.num_sample_datasets):
            self.test_TD_loss_random_observation_21(i, self.td_simulation_threshold)

        print(test_average_usage_times)
        print(self.num_sample_datasets)

        test_average_reward = sum(test_average_rewards) / self.num_sample_datasets
        test_average_actual_reward = sum(test_average_actual_rewards) / self.num_sample_datasets
        test_average_usage_time = sum(test_average_usage_times) / self.num_sample_datasets
        test_replace_failure = sum(test_replace_failures) / self.num_sample_datasets
        failure_probability = test_replace_failure / 100
        lambda_policy = (-self.REWARD_ACTUAL_REPLACE + failure_probability * (
                    -self.REWARD_ACTUAL_FAILURE + self.REWARD_ACTUAL_REPLACE)) / test_average_usage_time
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
        average_by_loss_dfs = self.env.calculate_average_performance_only_MSE(full_by_loss_dfs_list,
                                                                              self.num_sample_datasets)

        # Save average_by_loss_dfs to a file using pickle
        with open('average_by_loss_dfs_only_MSE.pkl', 'wb') as f:
            pickle.dump(average_by_loss_dfs, f)

        self.env.plot_simulation_results_scale_up(average_by_loss_dfs, self.num_dataset, self.loss_labels,
                                                  True)  # True : only MSE

    def plot_results(self):
        # 여러가지 plot들 한번에 출력하는 코드.
        global test_average_rewards, test_average_usage_times, test_replace_failures
        with open('average_by_loss_dfs_only_MSE.pkl', 'rb') as f:
            average_by_loss_dfs = pickle.load(f)
        self.env.plot_simulation_results(average_by_loss_dfs, self.num_dataset)
        # self.env.plot_simulation_results_scale_up(average_by_loss_dfs, self.num_dataset, self.loss_labels)
        self.env.plot_simulation_results_x_y_swap(average_by_loss_dfs, self.num_dataset, self.loss_labels, 100)
        self.env.plot_simulation_results_x_y_swap_cost(average_by_loss_dfs, self.num_dataset, self.loss_labels, 100,
                                                       -self.REWARD_ACTUAL_REPLACE, -self.REWARD_ACTUAL_FAILURE)
        self.env.plot_simulation_results_x_y_swap_cost_scale_up(average_by_loss_dfs, self.num_dataset, self.loss_labels,
                                                                100, -self.REWARD_ACTUAL_REPLACE,
                                                                -self.REWARD_ACTUAL_FAILURE)
        # AUT_Pi, P_failure는 위의 method에서 return하고 저장 후, 아래 method로 전달하자. 지금은 임시로 값을 직접 넣어둠.
        self.env.plot_simulation_results_x_y_swap_point_lambda_2(average_by_loss_dfs, self.num_dataset,
                                                                 self.loss_labels, 100)
        self.env.plot_simulation_results_x_y_swap_point_td_loss(average_by_loss_dfs, self.num_dataset, self.loss_labels,
                                                                100)

    def find_minimum_MSE_to_weight_by_td_loss(self):
        # td-loss로 학습된 weight으로 (22-dim; alpha = 1) MSE를 minimize 하는 threshold (theta)를 찾기 위한 method.
        # LR_TD_weight_by_RL_code_alpha_1 : decision-aware (q-learning) part만 고려해서 학습시킨 weight (threshold = 0 고정)
        with open('pkl_file/LR_TD_weight_by_RL_code_alpha_1.pkl', 'rb') as f:
            self.td_weight = pickle.load(f)

        # 모든 샘플 데이터들을 합쳐서 하나의 데이터 셋으로 만들어서 MSE 측정.
        # 아마 예전에 짜둔 코드가 있을테니 확인해보자.
        combined_data = pd.DataFrame()

        for i in range(self.num_sample_datasets):
            # train dataset으로 하려면 2번 째 index를 바꿔주면 됨. (full data는 2)
            sample_data = self.sampled_datasets_with_RUL[i][2].copy()  # full data (test)
            #sample_data = self.sampled_datasets_with_RUL[i][0].copy()   # train data
            combined_data = pd.concat([combined_data, sample_data], ignore_index=True)

        combined_data[self.columns_to_scale] = combined_data[self.columns_to_scale].apply(
            self.env.min_max_scaling, axis=0)
        # min-max scaling 마친 데이터에 s_0 열을 추가 (s_1 왼쪽에)하고 모든 값을 1로 초기화.
        combined_data.insert(loc=combined_data.columns.get_loc('s_1'), column='s_0', value=1)
        combined_data.reset_index(drop=True, inplace=True)  # index reset

        selected_data = combined_data.iloc[:, 5:27]  # only s_0, ~ s_21 column

        # Add the predicted_RUL column.
        self.td_weight = np.array(self.td_weight)
        # Compute the dot product for each row
        selected_data['predicted_RUL'] = selected_data.apply(lambda row: np.dot(row, self.td_weight), axis=1)
        selected_data['actual_RUL'] = combined_data['RUL'] # predicted RUL의 오른쪽에 actual RUL column 추가.
        selected_data = selected_data.iloc[:, 22:24] # predicted_RUL과 actual RUL만 남김 (불필요한 col 제거).

        # Create a DataFrame to store the results
        mse_results = pd.DataFrame(columns=['theta', 'MSE'])

        # Calculate MSE for predicted_RUL + value where value ranges from 0 to 100
        for value in range(101):
            adjusted_predicted_RUL = selected_data['predicted_RUL'] + value
            print(adjusted_predicted_RUL)
            mse = ((selected_data['actual_RUL'] - adjusted_predicted_RUL) ** 2).mean()
            mse_results = mse_results._append({'theta': value, 'MSE': mse}, ignore_index=True)

        print(mse_results)

        # Find the minimum MSE and corresponding theta
        min_mse = mse_results['MSE'].min()
        optimal_theta = mse_results.loc[mse_results['MSE'] == min_mse, 'theta'].values[0]

        print(f"Optimal theta: {optimal_theta}, Minimum MSE: {min_mse}")

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(mse_results['theta'], mse_results['MSE'], linestyle='-')
        plt.xlabel('Theta')
        plt.ylabel('MSE')
        plt.title('Threshold vs. MSE (alpha = 1)')
        plt.grid(True)

        # Mark the optimal theta on the plot
        plt.axvline(x=optimal_theta, color='r', linestyle='--')

        # Show the optimal theta value on the plot
        plt.text(optimal_theta + 4, min_mse + 200, f'{optimal_theta}', fontsize=12, verticalalignment='bottom',
                 horizontalalignment='center', color='red')
        plt.show()

    def calculate_MSE_weight_by_td_loss(self):
        # td-loss로 학습된 weight으로 (22-dim; alpha = 1) MSE를 minimize 하는 threshold (theta)를 찾기 위한 method.
        # LR_TD_weight_by_RL_code_alpha_1 : decision-aware (q-learning) part만 고려해서 학습시킨 weight (threshold = 0 고정)
        with open('LR_TD_weight_and_theta.pkl', 'rb') as f:
            data = pickle.load(f)

        self.td_weight = data['lr_weights_by_td'] # dictionary에서 td weight만 빼옴 (학습된 threshold도 저장되어있음.)

        # 모든 샘플 데이터들을 합쳐서 하나의 데이터 셋으로 만들어서 MSE 측정.
        # 아마 예전에 짜둔 코드가 있을테니 확인해보자.
        combined_data = pd.DataFrame()

        for i in range(self.num_sample_datasets):
            # train dataset으로 하려면 2번 째 index를 바꿔주면 됨. (full data는 2)
            sample_data = self.sampled_datasets_with_RUL[i][2].copy()  # full data (test)
            # sample_data = self.sampled_datasets_with_RUL[i][0].copy()   # train data
            combined_data = pd.concat([combined_data, sample_data], ignore_index=True)

        combined_data[self.columns_to_scale] = combined_data[self.columns_to_scale].apply(
            self.env.min_max_scaling, axis=0)
        # min-max scaling 마친 데이터에 s_0 열을 추가 (s_1 왼쪽에)하고 모든 값을 1로 초기화.
        combined_data.insert(loc=combined_data.columns.get_loc('s_1'), column='s_0', value=1)
        combined_data.reset_index(drop=True, inplace=True)  # index reset

        selected_data = combined_data.iloc[:, 5:27]  # only s_0, ~ s_21 column

        # Add the predicted_RUL column.
        self.td_weight = np.array(self.td_weight)
        # Compute the dot product for each row
        selected_data['predicted_RUL'] = selected_data.apply(lambda row: np.dot(row, self.td_weight), axis=1)
        selected_data['actual_RUL'] = combined_data['RUL']  # predicted RUL의 오른쪽에 actual RUL column 추가.
        selected_data = selected_data.iloc[:, 22:24]  # predicted_RUL과 actual RUL만 남김 (불필요한 col 제거).

        mse = ((selected_data['actual_RUL'] - selected_data['predicted_RUL']) ** 2).mean()
        print(mse)

    def generate_input_for_DCNN(self, is_train):
        # Original DCNN 적용을 위한 dataset 생성 (1~70 train, 71~100 valid, 1~100 full).
        # 사용하지 않는 센서 column 삭제 (DCNN paper).
        self.drop_columns = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
        self.train_data_DCNN, self.valid_data_DCNN, self.full_data_DCNN = self.env.load_data(self.num_dataset, self.split_unit_number)
        # 사용하지 않는 column 삭제 후, true label column 추가 (RUL)
        self.train_data_DCNN = self.train_data_DCNN.drop(columns=self.drop_columns, errors='ignore')
        self.train_data_DCNN = self.env.data_scaler_only_sensor(self.train_data_DCNN)
        self.train_data_DCNN = self.env.add_RUL_column(self.train_data_DCNN)

        # train_data_DCNN을 만들 때만 ObsTime과 is_last_time_cycle을 생성해야 함.
        obs_time, is_last_time_cycle = self.env.calculate_obs_time_and_is_last_timecycle(self.train_data_DCNN)

        self.valid_data_DCNN = self.valid_data_DCNN.drop(columns=self.drop_columns, errors='ignore')
        self.valid_data_DCNN = self.env.data_scaler_only_sensor(self.valid_data_DCNN)
        self.valid_data_DCNN = self.env.add_RUL_column(self.valid_data_DCNN)

        # 지금 code 기준으로 full data는 안씀 (2024.10.15)
        #self.full_data_DCNN = self.full_data_DCNN.drop(columns=self.drop_columns, errors='ignore')
        #self.full_data_DCNN = self.env.data_scaler_only_sensor(self.full_data_DCNN)
        #self.full_data_DCNN = self.env.add_RUL_column(self.full_data_DCNN)

        self.y_label = self.train_data_DCNN['RUL'] # train이 아닐 때는 아래의 if문에서 valid로 변경.

        # is_train이 true면 train dataset 반환, false면 valid dataset 반환.
        if is_train:
            new_dataset = self.create_time_window_dataset(self.train_data_DCNN, self.DCNN_N_tw)
        else:
            new_dataset = self.create_time_window_dataset(self.valid_data_DCNN, self.DCNN_N_tw)
            self.y_label = self.valid_data_DCNN['RUL']


        x_feature = new_dataset.reshape(-1, self.DCNN_N_tw, new_dataset.shape[2])

        if is_train:
            # train일 때는 return이 4개. train data와 true label, ObsTime, is_last_time_cycle
            return x_feature, self.y_label, obs_time, is_last_time_cycle
        else:
            # test인 경우는 return이 2개.
            return x_feature, self.y_label

    def generate_input_for_DCNN_observe_10(self, is_train):
        # 10%만 관측 가능할때의 input 생성.
        # 사용하지 않는 센서 column 삭제 (DCNN paper).
        self.drop_columns = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']

        # Initialize train_data_DCNN and valid_data_DCNN as empty arrays
        self.train_data_DCNN = np.empty((0, self.DCNN_N_tw, self.DCNN_N_ft))
        self.valid_data_DCNN = np.empty((0, self.DCNN_N_tw, self.DCNN_N_ft))
        obs_time = None
        is_last_time_cycle = None
        y_label_train = None
        y_label_valid = None

        # 샘플링한 데이터셋을 이어붙임.
        for data_sample_index in range(self.num_sample_datasets):
            # train data #############################################################################
            train_data_tmp = self.sampled_datasets_with_RUL[data_sample_index][0].copy() # 0 : train
            train_data_tmp = train_data_tmp.drop(columns='RUL', errors='ignore')
            train_data_tmp = train_data_tmp.drop(columns=self.drop_columns, errors='ignore')
            train_data_tmp = self.env.data_scaler_only_sensor(train_data_tmp)
            train_data_tmp = self.env.add_RUL_column(train_data_tmp)

            obs_time_tmp, is_last_time_cycle_tmp = self.env.calculate_obs_time_and_is_last_timecycle(train_data_tmp)

           # y-label, obs_time, is_last_time_cycle도 concat. (feature와 함께 loss의 input으로 들어가야함)
            y_label_train = pd.concat([y_label_train, train_data_tmp['RUL']])
            obs_time = pd.concat([obs_time, obs_time_tmp], ignore_index=True)
            is_last_time_cycle = pd.concat([is_last_time_cycle, is_last_time_cycle_tmp], ignore_index=True)

            # feature를 deep learning의 input으로 변환.
            train_data_tmp = self.create_time_window_dataset(train_data_tmp, self.DCNN_N_tw)
            self.train_data_DCNN = np.concatenate([self.train_data_DCNN, train_data_tmp], axis=0)

            # valid data #############################################################################
            valid_data_tmp = self.sampled_datasets_with_RUL[data_sample_index][1].copy() # 1 : valid
            valid_data_tmp = valid_data_tmp.drop(columns='RUL', errors='ignore')
            valid_data_tmp = valid_data_tmp.drop(columns=self.drop_columns, errors='ignore')
            valid_data_tmp = self.env.data_scaler_only_sensor(valid_data_tmp)
            valid_data_tmp = self.env.add_RUL_column(valid_data_tmp)

            # valid data의 경우에는 y_label만 필요함 (loss 크기 등, 퍼포먼스 측정용)
            y_label_valid = pd.concat([y_label_valid, valid_data_tmp['RUL']])

            # feature를 deep learning의 input으로 변환.
            valid_data_tmp = self.create_time_window_dataset(valid_data_tmp, self.DCNN_N_tw)
            self.valid_data_DCNN = np.concatenate([self.valid_data_DCNN, valid_data_tmp], axis=0)

            #print(f"Processed sample index: {data_sample_index}")

        # is_train이 true면 train dataset 반환, false면 valid dataset 반환.
        if is_train:
            new_dataset = self.train_data_DCNN
            self.y_label = y_label_train
        else:
            new_dataset = self.valid_data_DCNN
            self.y_label = y_label_valid

        x_feature = new_dataset.reshape(-1, self.DCNN_N_tw, new_dataset.shape[2])

        if is_train:
            # train일 때는 return이 4개. train data와 true label, ObsTime, is_last_time_cycle
            return x_feature, self.y_label, obs_time, is_last_time_cycle
        else:
            # test인 경우는 return이 2개.
            return x_feature, self.y_label

    def create_time_window_dataset(self, df, time_window):
        # DCNN의 input으로 넣을 feature 생성.
        # 결과를 저장할 리스트
        result = []
        # 각 unit_number별로 데이터를 그룹화
        grouped = df.groupby('unit_number')

        # 각 unit_number에 대해 반복
        for _, group in grouped:
            if self.DCNN_is_fully_observe:
                # time_cycles에 대해 정렬 (필요할 경우; sample data를 합친 것에 적용할때는 하면 안됨. 서로 다른 샘플들이 섞임.)
                group = group.sort_values(by='time_cycles')

            # group에서 인덱스를 재설정
            group = group.reset_index(drop=True)

            # group 내에서 time_cycles을 기준으로 time_window를 구성
            for i in range(len(group)):
                # 6번째 열부터 마지막 열까지만 포함
                sensor_data = group.iloc[:, 5:-1]

                # time_window에 해당하는 데이터 가져오기
                if i < time_window - 1:
                    # time_window가 부족한 경우, 0으로 패딩
                    padding = pd.DataFrame(0, index=np.arange(time_window - i - 1), columns=sensor_data.columns)
                    window = pd.concat([padding, sensor_data.iloc[:i + 1]], axis=0).reset_index(drop=True)
                else:
                    # 정상적으로 time_window 크기만큼 데이터 가져오기
                    window = sensor_data.iloc[i - time_window + 1:i + 1].reset_index(drop=True)

                # 결과 리스트에 추가
                result.append(window)

        # 모든 데이터를 리스트에서 결합하여 새로운 DataFrame 생성
        final_result = np.array(result)  # 3차원 데이터로 변환

        return final_result


    def calculated_prediction_and_decision_loss(self):
        # Fixed threshold로 학습을 시켰을 때, prediction loss, decision loss의 영향을 보기 위한 method.
        # threshold에 맞게 loss를 계산해줌 (threshold는 self.td_simulation_threshold 사용).
        # Initialize test data (for iteration).
        full_data = pd.DataFrame()
        # load saved weight
        with open('LR_TD_weight_by_RL_code.pkl', 'rb') as f:
            self.td_weight = pickle.load(f)

        # 샘플링한 데이터셋 병합 (24.10.01 기준 20개)
        for data_sample_index in range(self.num_sample_datasets):
            new_data = self.sampled_datasets_with_RUL[data_sample_index][2].copy()  # 2 means full data (0 train, 1 valid)
            full_data = pd.concat([full_data, new_data], ignore_index=True)

        print(full_data)

        # multiple linear regression setting이라 non-linear setting이랑 feature가 다름.
        full_data[self.columns_to_scale] = full_data[self.columns_to_scale].apply(self.env.min_max_scaling, axis=0)
        # Add a new column 's_0' filled with 1 to the left of 's_1'
        full_data.insert(full_data.columns.get_loc('s_1'), 's_0', 1)
        # Select the columns from index 5 to 27 (6th to 28th columns) for the dot product
        columns_for_prediction = full_data.iloc[:, 5:27]

        # Calculate the predicted_RUL using selected columns and td_weight
        full_data['predicted_RUL'] = np.dot(columns_for_prediction.values, self.td_weight)
        # Initialize the time_difference column with default value 0
        full_data['time_difference'] = 0
        # Iterate through the data to calculate time_difference and set is_terminal_state
        full_data['is_terminal_state'] = 0  # Initialize the is_terminal_state column with default value 0

        # Iterate through the data to calculate time_differenceR
        for i in range(len(full_data) - 1):
            # If the next row has the same unit_number as the current row, calculate time difference
            if full_data.loc[i, 'unit_number'] == full_data.loc[i + 1, 'unit_number']:
                full_data.loc[i, 'time_difference'] = full_data.loc[i + 1, 'time_cycles'] - full_data.loc[
                    i, 'time_cycles']
            else:
                # If unit_number changes, set time_difference to 0
                full_data.loc[i, 'time_difference'] = 0
                full_data.loc[i, 'is_terminal_state'] = 1  # Mark terminal state

        # The last row should have time_difference set to 0
        full_data.loc[len(full_data) - 1, 'time_difference'] = 0
        full_data.loc[len(full_data) - 1, 'is_terminal_state'] = 1

        # Filter the rows where 'is_terminal_state' is 1 (test)
        print(full_data[full_data['is_terminal_state'] == 1])

        # Initialize the decision_loss column with default value 0
        full_data['decision_loss'] = 0

        # Calculate decision_loss for each row
        for i in range(len(full_data)):
            if full_data.loc[i, 'is_terminal_state'] == 0:
                # Calculate the first term of the decision loss
                decision_loss_term1 = self.td_alpha * (1 - full_data.loc[i, 'is_terminal_state']) * (
                        full_data.loc[i, 'time_difference'] +
                        max(full_data.loc[i+1, 'predicted_RUL'] - self.td_simulation_threshold, 0) -
                        full_data.loc[i, 'predicted_RUL'] +
                        self.td_simulation_threshold
                ) ** 2
                full_data.loc[i, 'decision_loss'] = decision_loss_term1
            else:
                # Calculate the second term of the decision loss
                decision_loss_term2 = self.td_alpha * full_data.loc[i, 'is_terminal_state'] * (
                        -1 / self.td_beta - full_data.loc[i, 'predicted_RUL'] +
                        self.td_simulation_threshold
                ) ** 2
                full_data.loc[i, 'decision_loss'] = decision_loss_term2

        # Calculate the average decision_loss
        average_decision_loss = full_data['decision_loss'].mean()

        # Print the average decision_loss
        print("Average Decision Loss:", average_decision_loss)

        print(full_data)

        # Calculate the prediction loss (MSE)
        # (1 - alpha) * (RUL - predicted_RUL)^2 for each row
        full_data['squared_error'] = (full_data['RUL'] - full_data['predicted_RUL']) ** 2
        prediction_loss = (1 - self.td_alpha) * full_data['squared_error'].mean()

        # Print the average prediction loss and decision loss
        print(f"Prediction Loss (MSE): {prediction_loss}")
        print("Average Decision Loss:", average_decision_loss)


    def run_DCNN(self):
        # DCNN 학습, 예측치 저장까지 하나의 method로 구성.
        # 초기 학습용.

        # Model creation
        model = DCNN(self.DCNN_N_tw, self.DCNN_N_ft, self.DCNN_F_N, self.DCNN_F_L, self.DCNN_neurons_fc, self.DCNN_dropout_rate)
        trainer = DCNN_Model(model, self.DCNN_batch_size, self.DCNN_epochs, self.DCNN_is_td_loss, self.td_alpha, self.td_beta, self.td_simulation_threshold)

        # Training을 위한 data 생성.
        if self.DCNN_is_fully_observe: # 모든 데이터 관측 가능할 때.
            x_train, y_train, obs_time, is_last_time_cycle = self.generate_input_for_DCNN(True)  # True면 train data, label 반환 (False면 valid data, label 반환).
        else: # 10% 확률로 관측 가능할 때.
            x_train, y_train, obs_time, is_last_time_cycle = self.generate_input_for_DCNN_observe_10(True)

        # Ensure x_train, y_train, obs_time, is_last_time_cycle is a numpy array (tensor로 변환하려면 numpy array여야 함.)
        x_train = np.ascontiguousarray(np.array(x_train, dtype=np.float32))  # Convert to contiguous numpy array
        # Convert pandas Series to numpy array
        y_train = y_train.to_numpy(dtype=np.float32)
        obs_time = obs_time.to_numpy(dtype=np.float32)
        is_last_time_cycle = is_last_time_cycle.to_numpy(dtype=np.float32)

        # Assuming x_train and y_train are preloaded tensors
        # 여기서 ObsTime, is_last_time_cycle이 인자로 전달되어야 함 (td loss로 학습시키지 않더라도 전달은 함. DeepCNN에서 알아서 처리).
        trainer.train_model(x_train, y_train, obs_time, is_last_time_cycle, is_continue_learning = False)

        # model 학습 후 저장.
        trainer.save_model() # 별도 경로 지정 없이 저장 (현재 파일이 있는 디렉토리)

    def run_continue_DCNN(self):
        # DCNN 학습, 예측치 저장까지 하나의 method로 구성.
        # 이미 학습된 weight을 바탕으로 Loss를 바꿔 이어서 학습하는 코드.

        # Model creation
        model = DCNN(self.DCNN_N_tw, self.DCNN_N_ft, self.DCNN_F_N, self.DCNN_F_L, self.DCNN_neurons_fc, self.DCNN_dropout_rate)
        trainer = DCNN_Model(model, self.DCNN_batch_size, self.DCNN_epochs, self.DCNN_is_td_loss, self.td_alpha, self.td_beta, self.td_simulation_threshold)

        # Training을 위한 data 생성.
        if self.DCNN_is_fully_observe: # 모든 데이터 관측 가능할 때.
            x_train, y_train, obs_time, is_last_time_cycle = self.generate_input_for_DCNN(True)  # True면 train data, label 반환 (False면 valid data, label 반환).
        else: # 10% 확률로 관측 가능할 때.
            x_train, y_train, obs_time, is_last_time_cycle = self.generate_input_for_DCNN_observe_10(True)

        # Ensure x_train, y_train, obs_time, is_last_time_cycle is a numpy array (tensor로 변환하려면 numpy array여야 함.)
        x_train = np.ascontiguousarray(np.array(x_train, dtype=np.float32))  # Convert to contiguous numpy array
        # Convert pandas Series to numpy array
        y_train = y_train.to_numpy(dtype=np.float32)
        obs_time = obs_time.to_numpy(dtype=np.float32)
        is_last_time_cycle = is_last_time_cycle.to_numpy(dtype=np.float32)

        # 모델 학습 전, 현재 weight으로 성능 확인 (beta system의 reward도 함께)
        self.simulation_random_observation_merged_sample_data(self.td_simulation_threshold, is_train_data=True)

        # Assuming x_train and y_train are preloaded tensors
        # 여기서 ObsTime, is_last_time_cycle이 인자로 전달되어야 함 (td loss로 학습시키지 않더라도 전달은 함. DeepCNN에서 알아서 처리).
        trainer.train_model(x_train, y_train, obs_time, is_last_time_cycle, is_continue_learning=True) # 이어서 학습

        # model 학습 후 저장.
        trainer.save_model() # 별도 경로 지정 없이 저장 (현재 파일이 있는 디렉토리)

        # 모델 학습 후, 현재 weight으로 성능 확인 (beta system의 reward도 함께)
        self.simulation_random_observation_merged_sample_data(self.td_simulation_threshold, is_train_data=True)


    def add_predicted_RUL_using_saved_pth_full_observe(self):
        # 전체 관측 가능할 때 전용 method.
        # Model creation
        model = DCNN(self.DCNN_N_tw, self.DCNN_N_ft, self.DCNN_F_N, self.DCNN_F_L, self.DCNN_neurons_fc,
                     self.DCNN_dropout_rate)
        trainer = DCNN_Model(model, self.DCNN_batch_size, self.DCNN_epochs, self.DCNN_is_td_loss, self.td_alpha,
                             self.td_beta, self.td_simulation_threshold)

        # test를 위한 data 생성.
        if self.DCNN_is_fully_observe:
            x_valid, y_valid = self.generate_input_for_DCNN(False)  # False면 valid data, label 반환.
        else:
            x_valid, y_valid = self.generate_input_for_DCNN_observe_10(False)  # False면 valid data, label 반환.

        x_valid = np.ascontiguousarray(np.array(x_valid, dtype=np.float32))  # Convert to contiguous numpy array
        y_valid = y_valid.to_numpy(dtype=np.float32)  # Convert pandas Series to numpy array

        # 모델 로드 후 predict
        trainer.load_model()  # 기본 파일명을 사용하여 로드 (다른 pth load 하려면 바꿔줘야 함. 나중에 기능 추가하자)
        predictions = trainer.predict(x_valid)

        # dataframe 다시 만들기
        self.drop_columns = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
        # 사용하는 것은 valid_dataset만.
        self.train_dataset, self.valid_dataset, self.full_dataset = self.env.load_data(self.num_dataset, self.split_unit_number)

        # 여기서 RUL이 포함된 데이터셋을 load 해야함.
        self.valid_dataset = self.valid_dataset.drop(columns=self.drop_columns, errors='ignore')
        self.valid_dataset = self.env.data_scaler_only_sensor(self.valid_dataset)
        self.valid_dataset = self.env.add_RUL_column(self.valid_dataset)

        self.valid_dataset['predicted_RUL'] = predictions

        return self.valid_dataset

    def add_predicted_RUL_using_saved_pth_partial_observe(self, is_train_data=False):
        # 10% 관측 가능할 때 전용 method.
        # Model creation
        model = DCNN(self.DCNN_N_tw, self.DCNN_N_ft, self.DCNN_F_N, self.DCNN_F_L, self.DCNN_neurons_fc,
                     self.DCNN_dropout_rate)
        trainer = DCNN_Model(model, self.DCNN_batch_size, self.DCNN_epochs, self.DCNN_is_td_loss, self.td_alpha,
                             self.td_beta, self.td_simulation_threshold)

        # test를 위한 data 생성.
        if self.DCNN_is_fully_observe:
            x_valid, y_valid = self.generate_input_for_DCNN(is_train_data)  # is_train_data가 False면 valid data, label 반환.
        else:
            if is_train_data: # 코드의 통일을 위해 x_valid, y_valid라고 뒀지만, train data로 테스트 하는 코드임 (is_train_data = True일 때).
                x_valid, y_valid, obs_time, is_last_time_cycle = self.generate_input_for_DCNN_observe_10(is_train_data)
            else:
                x_valid, y_valid = self.generate_input_for_DCNN_observe_10(is_train_data)  # is_train_data가 False면 valid data, label 반환.

        x_valid = np.ascontiguousarray(np.array(x_valid, dtype=np.float32))  # Convert to contiguous numpy array
        y_valid = y_valid.to_numpy(dtype=np.float32)  # Convert pandas Series to numpy array

        # 모델 로드 후 predict
        trainer.load_model()  # 기본 파일명을 사용하여 로드 (다른 pth load 하려면 바꿔줘야 함. 나중에 기능 추가하자)
        predictions = trainer.predict(x_valid)

        # dataframe 다시 만들기
        self.drop_columns = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
        # Test용 dataset initialize.
        self.valid_dataset = pd.DataFrame()

        # 샘플링한 데이터셋을 이어붙임 (# 10% 관측 가능할 때 전용 method).
        # 전체 데이터에 대한 경우도 적용하고 싶으면 is_partial_observe를 이용해서 valid_dataset을 전체 데이터로 만들자.

        # index에 따라 train, valid, full data로 나뉨.
        train_valid_switching_index = 1 # default는 valid 데이터.
        if is_train_data:
            train_valid_switching_index = 0 # train data로 시뮬레이션 할 때는 index를 0으로 변경

        for data_sample_index in range(self.num_sample_datasets):
            new_data = self.sampled_datasets_with_RUL[data_sample_index][train_valid_switching_index].copy()
            self.valid_dataset = pd.concat([self.valid_dataset, new_data], ignore_index=True)

        self.valid_dataset['predicted_RUL'] = predictions

        return self.valid_dataset

    def simulation_random_observation_merged_sample_data(self, threshold, is_train_data=False):
        # 일단 random observation 상황에서만 짜보자. unit number가 반복되니.
        dataset = self.add_predicted_RUL_using_saved_pth_partial_observe(is_train_data)

        # Data processing
        dataset['predicted_RUL_threshold'] = dataset['predicted_RUL'] - threshold # y^ - threshold
        dataset = dataset.drop(columns = dataset.columns[2:26]) # remove feature columns
        dataset['is_last_time_cycle'] = 0  # is_last_time_cycle 추가 마지막 time cycle이면 1을 저장.
        dataset.loc[len(dataset) - 1, 'is_last_time_cycle'] = 1 # 마지막 인덱스를 1로 설정.

        # RUL 값이 증가하는 시점의 직전 인덱스에 1을 기록 (엔진 내의 마지막 관측치면 1, 아니면 0)
        for i in range(1, len(dataset)):
            if dataset.loc[i, 'RUL'] > dataset.loc[i - 1, 'RUL']:
                dataset.loc[i - 1, 'is_last_time_cycle'] = 1

        # action이 continue면 1, replace면 0
        dataset['is_continue'] = np.where(dataset['predicted_RUL_threshold'] > 0, 1, 0)

        # 위의 코드만으로는 이미 replace가 일어난 이후에는 continue 할 수 없다는 것이 반영되어있지 않음.
        # 따라서 엔진 내에서 replace가 발생 (is_continue = 0)하면, 이후의 is_continue는 모두 0으로 처리)
        for i in range(2, len(dataset)):
            if dataset.loc[i, 'is_continue'] == 1 and dataset.loc[i - 1, 'is_continue'] == 0 and dataset.loc[
                i - 1, 'is_last_time_cycle'] == 0:
                #print(f"event! index : {i}")
                dataset.loc[i, 'is_continue'] = 0

        # 'is_replace_failure' = 'is_last_time_cycle' * 'is_continue' (마지막 사이클에서 continue 한 경우)
        dataset['is_replace_failure'] = dataset['is_last_time_cycle'] * dataset['is_continue']

        # usage_time 열을 추가 (엔진 별 누적 사용 시간).
        dataset['usage_time'] = 0
        for i in range(1, len(dataset)):
            if dataset.loc[i, 'is_continue'] == 0 and dataset.loc[i - 1, 'is_last_time_cycle'] == 0 and dataset.loc[
                i - 1, 'is_continue'] == 1:
                dataset.loc[i, 'usage_time'] = dataset.loc[i, 'time_cycles']

        # 'is_replace_failure'가 1인 인덱스의 'usage_time'을 해당 인덱스의 'time_cycles'로 설정
        dataset.loc[dataset['is_replace_failure'] == 1, 'usage_time'] = dataset['time_cycles']

        total_number_of_engines = dataset['is_last_time_cycle'].sum()
        total_operation_time = dataset['usage_time'].sum()
        replace_failure = dataset['is_replace_failure'].sum()

        average_usage_time_per_engine = total_operation_time / total_number_of_engines
        average_cost_per_engine = (-self.REWARD_ACTUAL_REPLACE * (total_number_of_engines - replace_failure) - self.REWARD_ACTUAL_FAILURE * replace_failure) / total_number_of_engines
        p_failure = replace_failure / total_number_of_engines
        average_cost_per_time = average_cost_per_engine / average_usage_time_per_engine # a.k.a. lambda
        beta = average_cost_per_time / (-self.REWARD_ACTUAL_FAILURE + self.REWARD_ACTUAL_REPLACE) # by formula

        # beta system 내에서 학습 초기와 끝날 때 개선이 이뤄졌는지 확인하기 위한 코드. 따로 반환값으로 추가하진 않음
        average_beta_reward = ( (self.td_beta * (-self.REWARD_ACTUAL_FAILURE)) * average_usage_time_per_engine - (-self.REWARD_ACTUAL_FAILURE) * p_failure) / total_number_of_engines

        print(f"Theta : {threshold}, AUT : {average_usage_time_per_engine}, P_failure : {p_failure}, Lambda : {average_cost_per_time}, Beta : {beta}, Average beta reward : {average_beta_reward}")

        return threshold, average_usage_time_per_engine, p_failure, average_cost_per_time, beta


    def generate_threshold_simulation_data(self, start=0, end=40, step=0.1):
        # simulation_random_observation_merged_sample_data를 threshold를 바꿔가며 실행.
        # theta^*을 찾기 위한 method.
        results_df = pd.DataFrame(
            columns=["threshold", "average_usage_time_per_engine", "p_failure", "average_cost_per_time", "beta"])

        # threshold 값을 start부터 end까지 step 간격으로 증가시키며 반복
        for threshold in [start + x * step for x in range(int((end - start) / step) + 1)]:
            # 각 threshold에 대해 함수 호출
            threshold, avg_usage_time, p_failure, avg_cost_per_time, beta = self.simulation_random_observation_merged_sample_data(
                threshold, is_train_data=False) # Test data를 쓸 때는, is_train_data=False

            # 반환된 값을 한 행으로 데이터프레임에 추가
            new_row = pd.DataFrame({
                "threshold": [threshold],
                "average_usage_time_per_engine": [avg_usage_time],
                "p_failure": [p_failure],
                "average_cost_per_time": [avg_cost_per_time],
                "beta": [beta]
            })

            # concat을 사용하여 새로운 행을 추가
            results_df = pd.concat([results_df, new_row], ignore_index=True)

        self.plot_threshold_vs_cost(results_df)

        return results_df

    def plot_threshold_vs_cost(self, df):
        # average_cost_per_time이 최소가 되는 행을 찾음
        min_cost_row = df.loc[df['average_cost_per_time'].idxmin()]

        # Plot 생성
        plt.figure(figsize=(10, 6))
        plt.plot(df['threshold'], df['average_cost_per_time'], label='Average Cost per Time')

        # 최소값에 해당하는 threshold에 빨간 점 표시
        plt.scatter(min_cost_row['threshold'], min_cost_row['average_cost_per_time'], color='red',
                    label=f"Min Cost at threshold={min_cost_row['threshold']:.2f}")

        # Labeling
        plt.xlabel('Threshold')
        plt.ylabel('Average Cost per Time')
        plt.title('Threshold vs. Average Cost per Time')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 최소 average_cost_per_time에 해당하는 행의 값 출력
        print("Row with minimum average_cost_per_time:")
        print(min_cost_row)

    def plot_RUL_prediction_using_saved_pth(self, is_partial_observe):
        # predicted RUL column을 추가.
        # 여기서 is_partial_obeserve에 따라 다른 method로 데이터 프레임을 생성.
        if is_partial_observe:
            valid_data_with_predicted_RUL = self.add_predicted_RUL_using_saved_pth_partial_observe()
        else:
            valid_data_with_predicted_RUL = self.add_predicted_RUL_using_saved_pth_full_observe()

        # 마지막 샘플만 RUL plot (valid 1개 샘플 사이즈는 650. 나중에 수정; 10% 관측의 경우)
        if is_partial_observe:
            #last_sample = valid_data_with_predicted_RUL.tail(650) # 10%만 관측 가능할 때.
            #last_sample = valid_data_with_predicted_RUL.tail(1300) # 20%만 관측 가능할 때,
            #last_sample = valid_data_with_predicted_RUL.tail(1950) # 30% 관측 가능,
            #last_sample = valid_data_with_predicted_RUL.tail(3250)  # 50% 관측 가능

            # dataset 3
            last_sample = valid_data_with_predicted_RUL.tail(1433)  # 20% 관측 가능,
            #last_sample = valid_data_with_predicted_RUL.tail(2150)  # 30% 관측 가능,
        else:
            last_sample = valid_data_with_predicted_RUL.tail(6501) # 전체 데이터 관측 가능할 때.

        print(last_sample)
        self.env.plot_RUL_prediction_by_DCNN(last_sample, self.td_simulation_threshold)

        # valid data의 MSE 출력.
        mse_value = self.calculate_MSE_loss(valid_data_with_predicted_RUL)
        print(f"MSE between RUL and predicted_RUL: {mse_value}")

    def calculate_MSE_loss(self, dataset):
        true_rul = torch.tensor(dataset['RUL'].values, dtype=torch.float32)
        predicted_rul = torch.tensor(dataset['predicted_RUL'].values, dtype=torch.float32)
        mse = torch.mean((true_rul - predicted_rul) ** 2)
        return mse.item()

    def plot_prediction_decision_loss_by_threshold(self):
        # threshold에 따른 prediction loss, decision loss, time average cost 비교 plot.
        # 실험 데이터를 직접 입력해서 plot만 그려주는 method.
        threshold = [0, 10, 20, 30, 40,
                     50, 60, 80, 90, 110,
                     120, 130]
        prediction_loss = [1896.54, 1898.44, 1900.74, 1903.42, 1906.57,
                           1910.23, 1914.54, 1925.23, 1931.95, 1948.82,
                           1959.02, 1970.70]
        decision_loss = [2599.73, 2534.09, 2470.38, 2408.89, 2350.02,
                         2294.21, 2241.94, 2150.04, 2111.85, 2056.01,
                         2041.24, 2037.54]
        average_cost = [18.0255, 14.5555, 12.2323, 10.1881, 8.5900,
                        7.8224, 7.2498, 6.9000, 7.2524, 9.3326,
                        11.1962, 13.6109]

        self.env.plot_prediction_decision_loss_and_cost_by_threshold(threshold, prediction_loss, decision_loss,
                                                            average_cost)

    def plot_cost_by_beta(self):
        # beta, timeaverage cost를 입력 후, beta에 따른 lambda를 plot.
        # lambda를 minimize 하는 beta를 찾는 것을 시각화.
        # 실험 데이터를 직접 입력해서 plot만 그려주는 method.
        beta = [0.00200, 0.00150, 0.00137, 0.00120, 0.00110,
                0.00100, 0.00050, 0.00034, 0.00025, 0.00021, 0.00020,
                0.00019, 0.000175, 0.00015, 0.00010, 0.00005]
        time_average_cost = [7.2220, 7.1582, 7.1749, 7.1114, 7.1255,
                             7.1116, 7.0815, 7.0719, 7.0808, 7.0597, 7.0341,
                             7.0341, 7.0371, 7.0379, 7.0477, 7.0553]
        self.env.plot_time_average_cost_by_beta(beta, time_average_cost)



"""generate instance"""
#run_sim = RunSimulation('config_009.ini')   # 전체 관측, (MSE) or (TD Loss, alpha 0.1, theta 0)
#run_sim = RunSimulation('config_010.ini')  # 10% 관측, TD Loss, alpha 0.1, theta 0
#run_sim = RunSimulation('config_011.ini')  # 10% 관측, MSE Loss (1000 epoch)
#run_sim = RunSimulation('config_012.ini')  # 10% 관측, TD Loss, alpha 0.9, theta 0
#run_sim = RunSimulation('config_013.ini')  # 10% 관측, TD Loss, alpha 0.5, theta 0
#run_sim = RunSimulation('config_014.ini')   # 10% 관측, TD Loss, alpha 1.0, theta 0, beta 0.000684
#run_sim = RunSimulation('config_015.ini')   # 10% 관측, TD Loss, alpha 1.0, theta 0, beta 0.001370
#run_sim = RunSimulation('config_016.ini')   # 10% 관측, TD Loss, alpha 1.0, theta 42.7, beta 0.001370
#run_sim = RunSimulation('config_017.ini')   # 10% 관측, TD Loss, alpha 0.1, theta 42.7, beta 0.001370
#run_sim = RunSimulation('config_018.ini')   # 10% 관측, TD Loss, alpha 0.1, theta 42.7, beta 0.001370
#run_sim = RunSimulation('config_019.ini')   # 10% 관측, TD Loss, alpha 0.1, theta 58.8, beta 0.000701 (2000 epoch)
#run_sim = RunSimulation('config_020.ini')   # 10% 관측, TD Loss, alpha 1.0, theta 41.8, beta 0.000687 (2000 epoch)
#run_sim = RunSimulation('config_021.ini')   # 10% 관측, TD Loss, alpha 0.1, theta 41.8, beta 0.000687 (2000 epoch)
#run_sim = RunSimulation('config_022.ini')   # 10% 관측, TD Loss, alpha 0.3, theta 41.8, beta 0.000687 (2000 epoch)
#run_sim = RunSimulation('config_023.ini')   # 10% 관측, TD Loss, alpha 0.9, theta 41.8, beta 0.000687 (2000 epoch)

# epoch (1,000 -> 2,000으로 변경)
#run_sim = RunSimulation('config_024.ini')   # 10% 관측, MSE Loss (2000 epoch)
#run_sim = RunSimulation('config_025.ini')   # 10% 관측, TD, alpha 0.1, theta 37.6, beta 0.000681

# Cost 수정 후 테스트 (MSE는 1000 epoch 학습시킨 것으로 사용)
#run_sim = RunSimulation('config_026.ini')   # 10% 관측, failure 5,000, TD alpha 0.1, theta 30.3, beta 0.001466
#run_sim = RunSimulation('config_027.ini')   # 10% 관측, failure 10,000, replace 500, TD alpha 0.1, theta 45.6, beta 0.000336


# Observation Probability 수정 후 테스트 (MSE는 1000 epoch 학습; Cost는 그대로)
#run_sim = RunSimulation('config_028.ini')   # 30% 관측, MSE
#run_sim = RunSimulation('config_029.ini')   # 30% 관측, TD alpha 0.1, theta 19.6, beta 0.000612
#run_sim = RunSimulation('config_030.ini')   # 20% 관측, MSE
#run_sim = RunSimulation('config_031.ini')   # 20% 관측, TD alpha 0.1, theta 25.6, beta 0.000646
#run_sim = RunSimulation('config_032.ini')   # 50% 관측, MSE
#run_sim = RunSimulation('config_033.ini')   # 50% 관측, TD alpha 0.1, theta 12.5, beta 0.000586

# Dataset 3
# run_sim = RunSimulation('config_034.ini')  # 30% 관측 MSE
#run_sim = RunSimulation('config_035.ini')  # 30% 관측, TD alpha 0.1, theta 28.3, beta 0.000498
#run_sim = RunSimulation('config_036.ini')  # 20% 관측 MSE
run_sim = RunSimulation('config_037.ini')  # 20% 관측 MSE, TD alpha 0.1,

""" ###############################
Deep Convolution Neural Network
"""
#run_sim.run_DCNN()  # DCNN 학습.
run_sim.run_continue_DCNN() # 이미 학습된 weight(dcnn_model.pth)을 이어서 학습하는 코드

#run_sim.simulation_random_observation_merged_sample_data(41.8) # 학습한 모델로, 인자로 넣은 threshold 에서 테스트

#run_sim.plot_RUL_prediction_using_saved_pth(is_partial_observe = False) # 학습된 모델로 RUL prediction 수행 (모든 데이터 관측 가능).
run_sim.plot_RUL_prediction_using_saved_pth(is_partial_observe = True) # 학습된 모델로 RUL prediction 수행 (n % 데이터만 관측 가능).

# 최적의 threshold 찾기
run_sim.generate_threshold_simulation_data() # Find optimal theta (in train data).
#run_sim.generate_threshold_simulation_data() # Find optimal theta (in test data).


""" ################################
Linear Regression Simulation
"""
# run_sim.run_many()
# run_sim.plot_results()

# MSE만 사용해서 LR 시뮬레이션 하는 코드. 다른 loss function들은 이제 필요 없음.
# run_sim.run_many_only_MSE()
# run_sim.plot_lr_td_loss_to_RUL_all_samples_21(1) # RUl prediction plot (21차원용)
# run_sim.plot_lr_td_loss_to_RUL_all_samples(1) # RUl prediction plot
# run_sim.plot_results() # plot 그리는 코드 (퍼포먼스 비교용)

""" #################################
Reinforcement Learning (value-based)
"""
# Weights를 처음 학습시킬때만 실행.
# run_sim.train_many_off_policy_RL()

# 학습된 weights를 바탕으로 이어서 학습하기 위한 method.
# run_sim_1.train_many_by_saved_weights_off_policy_RL()
# run_sim.train_many_by_saved_weights_off_policy_RL()

# 저장된 weights으로 전체 엔진에 대한 test 수행.
# run_sim.run_RL_simulation()
# run_sim_1.run_RL_simulation()


""" RL 코드를 기반으로 한, TD loss로 Linear regression 학습. """
#run_sim.train_many_lr_by_td_loss()              # 2024.10.02에 마지막으로 씀. threshold 별 DL, PL 비교하기 위해 학슴 진행.
# run_sim.train_continue_many_lr_by_td_loss()   # 이미 학습된 weight을 이어서 학습시킬 때 사용.
#run_sim.run_TD_loss_simulation()                # 학습 결과 시뮬레이션.
#run_sim.calculated_prediction_and_decision_loss() # prediction loss, decision loss 계산.

#run_sim.plot_prediction_decision_loss_by_threshold() # 학습된 weight으로 prediction loss, decision loss 계산
#run_sim.plot_cost_by_beta() # beta에 따른 time average cost 출력용.

# 잠깐 블럭처리
# theta도 함께 학습
#run_sim.train_many_lr_by_td_loss_theta()
#run_sim.run_TD_loss_simulation_theta(False)  # 학습된 threshold로 성능 테스트
#run_sim.run_TD_loss_simulation_theta(True)  # Config에 지정된 threshold로 성능 테스트

# 학습된 weight으로 MSE 계산
#run_sim.calculate_MSE_weight_by_td_loss()

# RUL prediction
#run_sim.plot_lr_td_loss_to_RUL_all_samples(1) # RUl prediction plot


# MSE가 최소가 되는 threshold 찾기.
#run_sim.find_minimum_MSE_to_weight_by_td_loss()

# 21차원으로 학습시키는 코드, 테스트
# run_sim.train_many_lr_by_td_loss_21()
# run_sim.run_TD_loss_simulation_21()


# TD Loss 학습 코드.
"""
run_sim.train_many_lr_by_DA_TD_loss()
#run_sim.train_continue_many_lr_by_DA_td_loss()

#run_sim.calaulate_solution_of_lr_DA_TD_loss(80)  # lambda (ridge)
run_sim.run_TD_loss_simulation()
run_sim.plot_lr_td_loss_to_RUL_all_samples(1)
"""
