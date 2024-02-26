import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from sklearn.linear_model import LinearRegression
import sklearn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random


class SimulationEnvironment():
    def __init__(self):
        self.index_names = ['unit_number', 'time_cycles']
        self.setting_names = ['setting_1', 'setting_2', 'setting_3']
        self.sensor_names = ['s_{}'.format(i + 1) for i in range(0, 21)]
        self.col_names = self.index_names + self.setting_names + self.sensor_names
        self.dataset_paths = {
            1: 'dataset/train_FD001.txt',
            2: 'dataset/train_FD002.txt',
            3: 'dataset/train_FD003.txt',
            4: 'dataset/train_FD004.txt'
        }

    def load_data(self, dataset_number, split_number):
        if dataset_number not in self.dataset_paths:
            print("Invalid dataset number. Please provide a number between 1 and 4.")
            return None

        dataset_path = self.dataset_paths[dataset_number]
        data = pd.read_csv(dataset_path, sep='\s+', header=None, index_col=False, names=self.col_names)

        # dataset의 unit_number를 기준으로 train, test, full data로 나누기
        train = data[data['unit_number'] <= split_number] # train data (fitting)
        test = data[data['unit_number'] > split_number]   # test data
        full = data.copy()                                # full data

        return train, test, full

    def sampling_datasets(self, num_sample_datasets, observation_probability, train, test, full):
        # 샘플링 결과를 저장할 리스트 초기화
        sampled_data_list = []

        # num_sample_datasets만큼 sampling
        for i in range(num_sample_datasets):    # i means random seed
            # 샘플링 수행
            sampled_train = train.sample(frac=observation_probability, random_state=i)
            sampled_test = test.sample(frac=observation_probability, random_state=i)
            sampled_full = full.sample(frac=observation_probability, random_state=i)

            # Sorting
            sampled_train = sampled_train.sort_values(by=['unit_number', 'time_cycles'])
            sampled_test = sampled_test.sort_values(by=['unit_number', 'time_cycles'])
            sampled_full = sampled_full.sort_values(by=['unit_number', 'time_cycles'])

            # 샘플링 결과를 리스트에 추가
            sampled_data_list.append((sampled_train.copy(), sampled_test.copy(), sampled_full.copy()))

        return sampled_data_list

    def add_RUL_column(self, dataframe):
        train_grouped_by_unit = dataframe.groupby(by='unit_number')
        max_time_cycles = train_grouped_by_unit['time_cycles'].max()
        merged = dataframe.merge(max_time_cycles.to_frame(name='max_time_cycle'), left_on='unit_number', right_index=True)
        merged["RUL"] = merged["max_time_cycle"] - merged['time_cycles']
        merged = merged.drop("max_time_cycle", axis=1)
        return merged



