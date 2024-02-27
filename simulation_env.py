import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import sklearn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import random

# tensorflow.keras 오류 수정하자. loss 5,6 학습할 때 필요함.
from tensorflow.keras import layers, models



class SimulationEnvironment():
    def __init__(self):
        self.index_names = ['unit_number', 'time_cycles']
        self.setting_names = ['setting_1', 'setting_2', 'setting_3']
        self.drop_lables = self.index_names + self.setting_names
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

        # dataset의 unit_number를 기준으로 train, valid, full data로 나누기
        train = data[data['unit_number'] <= split_number]  # train data (fitting)
        valid = data[data['unit_number'] > split_number]   # valid data
        full = data.copy()                                 # full data

        return train, valid, full

    def sampling_datasets(self, num_sample_datasets, observation_probability, train, valid, full):
        # 샘플링 결과를 저장할 리스트 초기화
        sampled_data_list = []

        # num_sample_datasets만큼 sampling
        for i in range(num_sample_datasets):    # i means random seed
            # 샘플링 수행
            sampled_train = train.sample(frac=observation_probability, random_state=i)
            sampled_valid = valid.sample(frac=observation_probability, random_state=i)
            sampled_full = full.sample(frac=observation_probability, random_state=i)

            # Sorting
            sampled_train = sampled_train.sort_values(by=['unit_number', 'time_cycles'])
            sampled_valid = sampled_valid.sort_values(by=['unit_number', 'time_cycles'])
            sampled_full = sampled_full.sort_values(by=['unit_number', 'time_cycles'])

            # 샘플링 결과를 리스트에 추가
            sampled_data_list.append((sampled_train.copy(), sampled_valid.copy(), sampled_full.copy()))

        return sampled_data_list

    def add_RUL_column(self, dataframe):
        train_grouped_by_unit = dataframe.groupby(by='unit_number')
        max_time_cycles = train_grouped_by_unit['time_cycles'].max()
        merged = dataframe.merge(max_time_cycles.to_frame(name='max_time_cycle'), left_on='unit_number', right_index=True)
        merged["RUL"] = merged["max_time_cycle"] - merged['time_cycles']
        merged = merged.drop("max_time_cycle", axis=1)
        return merged

    def add_RUL_column_to_sampled_datasets(self, sampled_datasets):
        # 새로운 샘플링 결과를 저장할 리스트 초기화
        sampled_data_with_RUL_list = []

        # 각각의 샘플링 데이터셋에 RUL 열 추가
        for sampled_train, sampled_valid, sampled_full in sampled_datasets:
            sampled_train_with_RUL = self.add_RUL_column(sampled_train)
            sampled_valid_with_RUL = self.add_RUL_column(sampled_valid)
            sampled_full_with_RUL = self.add_RUL_column(sampled_full)

            # 결과를 리스트에 추가
            sampled_data_with_RUL_list.append((sampled_train_with_RUL, sampled_valid_with_RUL, sampled_full_with_RUL))

        return sampled_data_with_RUL_list

    def drop_labels_from_train_data(self, train_data): # for train data
        train_dataset_index_names = train_data[['unit_number', 'time_cycles']]   # save index name (for scaling)
        dropped_train_dataset = train_data.drop(columns=self.drop_lables).copy() # split dataset
        tmp_x_train, tmp_x_test, tmp_y_train, tmp_y_test = train_test_split(dropped_train_dataset, dropped_train_dataset['RUL'], test_size=0.01, shuffle=False)

        return train_dataset_index_names, tmp_x_train, tmp_x_test, tmp_y_train, tmp_y_test

    def drop_labels_from_data(self, data): # for valid, full data
        dataset_index_names = data[['unit_number', 'time_cycles']]   # save index name (for scaling)
        dropped_dataset = data.drop(columns=self.drop_lables).copy() # split dataset
        labels_data = data[['RUL']]                                  # save true RUL

        return dataset_index_names, dropped_dataset, labels_data

    def data_scaler(self, data):
        scaler = MinMaxScaler()  # MinMax Scaler
        # Droping the target variable (True RUL)
        data.drop(columns=['RUL'], inplace=True)  # inplace -> true면 원본 삭제 (current option)  # inplace -> False면 원본은 유지
        scaled_data = scaler.fit_transform(data)

        return scaled_data









