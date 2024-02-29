import pickle
from copy import copy
import numpy as np
from simulation_env import SimulationEnvironment
from RL_env import RLParams
from RL_env import Environment

if __name__ == "__main__":
    # Example usage
    #train_data = ...  # Your training data

    rl_params = RLParams('config1.ini')
    print(rl_params)
    #environment = Environment(train_data)
    #agent = Agent()
    #rewards = Rewards()
    #rl_utils = RLUtilities(environment, agent, rewards)