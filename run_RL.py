import pickle
from copy import copy
import numpy as np
from simulation_env import SimulationEnvironment
from RL_env import RLParams
from RL_env import Environment


# 여기서 학습된 pkl 파일로 그래프를 그릴 수 있어야 함. ML 학습이 따로 필요하지 않음.

if __name__ == "__main__":
    # Example usage
    #train_data = ...  # Your training data

    rl_params = RLParams('config1.ini')
    print(rl_params.observation_probability)
    environment = Environment('config1.ini')
    #environment = Environment(train_data)
    #agent = Agent()
    #rewards = Rewards()
    #rl_utils = RLUtilities(environment, agent, rewards)