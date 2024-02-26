import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import sklearn
from sklearn.metrics import mean_squared_error, r2_score
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import warnings
np.random.seed(34)
warnings.filterwarnings('ignore')


class SimulationEnvironment():
    def __int__(self):
        self.index_names = ['unit_number', 'time_cycles']
        self.setting_names = ['setting_1', 'setting_2', 'setting_3']
        self.sensor_names = ['s_{}'.format(i + 1) for i in range(0, 21)]
        self.col_names = self.index_names + self.setting_names + self.sensor_names

    def load_data(self, number_of_dataset):



env = SimulationEnvironment()
env.hello()
