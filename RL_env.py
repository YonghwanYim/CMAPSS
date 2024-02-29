
import pickle
from copy import copy
import numpy as np
import configparser
from simulation_env import SimulationEnvironment

# generate configuration instance
config = configparser.ConfigParser()

# RL environment 구현.
# 데이터셋 정의부터, state 변경 등등..
# agent는 따로 어떻게 파일로 만들지도 생각하자.
# 알고리즘과는 분리해야 함. value-based, policy-based 모두 적용할 수 있도록.

# Load average_by_loss_dfs from the file
with open('average_by_loss_dfs.pkl', 'rb') as f:
    average_by_loss_dfs = pickle.load(f)

#print(average_by_loss_dfs)
# 이 데이터를 가지고 plot을 그리는 method는 따로 만들어야함.

class RLParams:
    def __init__(self, config_path):
        config.read(config_path)

        self.REPLACE_COST = int(config['SimulationSettings']['REPLACE_COST'])
        self.FAILURE_COST = int(config['SimulationSettings']['FAILURE_COST'])

        self.gamma = float(config['RL_Settings']['discount_factor'])
        self.alpha = float(config['RL_Settings']['learning_rate'])
        self.initial_epsilon = float(config['RL_Settings']['initial_epsilon'])
        self.epsilon_delta = float(config['RL_Settings']['epsilon_delta'])
        self.min_epsilon = float(config['RL_Settings']['min_epsilon'])
        self.max_episodes = int(config['RL_Settings']['max_episodes'])
        self.observation_probability = float(config['SimulationSettings']['observation_probability'])

class Environment:
    def __init__(self, config_path):
        config.read(config_path)
        simul_env = SimulationEnvironment()
        self.dataset_number = int(config['SimulationSettings']['num_dataset'])
        dataset_path = simul_env.dataset_paths[self.dataset_number]
        self.train_data, self.valid_data, self.full_data = self.simul_env.load_data(self.num_dataset, self.split_unit_number)
        self.data = simul_env.add_RUL_column(self.data)


class Rewards:
    def __init__(self, r_continue=0, r_continue_but_failure=-10000, r_replace=-1000, config_path):
        config.read(config_path)
        self.r_continue = r_continue
        self.r_continue_but_failure = r_continue_but_failure
        self.r_replace = r_replace

    def calculate_reward(self, current_index, next_index, action, environment):
        current_unit_number = environment.data['unit_number'].iloc[current_index]
        next_unit_number = environment.data['unit_number'].iloc[next_index]

        if action == 'continue':
            return self.r_continue if current_unit_number == next_unit_number else self.r_continue_but_failure
        elif action == 'replace':
            return self.r_replace

class RLUtilities:
    def __init__(self, environment, agent, rewards):
        self.environment = environment
        self.agent = agent
        self.rewards = rewards

    def state_min_index(self, current_index):
        current_unit_number = self.environment.data['unit_number'].iloc[current_index]
        return self.environment.data[self.environment.data['unit_number'] == current_unit_number].index[0]


class Agent:
    def __init__(self, actions=["continue", "replace"]):
        self.actions = actions
        self.weights = {action: np.random.normal(loc=0, scale=0.5, size=22) for action in actions}
        self.best_weights = {action: np.random.normal(loc=0, scale=0.5, size=22) for action in actions}

    def get_weights(self):
        return self.weights

    def get_best_weights(self):
        return self.best_weights


if __name__ == "__main__":
    # Example usage
    train_data = ...  # Your training data

