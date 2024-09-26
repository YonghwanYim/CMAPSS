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
import tensorflow as tf
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

    def add_RUL_column_to_datasets(self, dataset): # for Reinforcement Learning (only full data)
        # 새로운 샘플링 결과를 저장할 리스트 초기화
        data_with_RUL = []
        for sampled_full in dataset:
            data_with_RUL = self.add_RUL_column(sampled_full)

        return data_with_RUL


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

    def data_scaler_only_sensor(self, data):
        # 첫 5개의 column은 스케일링 하지 않음.
        non_sensor_columns = data.iloc[:, :5]
        # 6번째 열부터 끝까지 센서 데이터만 스케일링 적용
        sensor_columns = data.iloc[:, 5:]

        scaler = MinMaxScaler() # MinMax Scaler
        scaled_sensor_data = scaler.fit_transform(sensor_columns)
        #print(scaled_sensor_data)
        #print(scaled_sensor_data.shape) # 정상적으로 동작한다면 size가 14로 끝남.

        scaled_data = pd.concat([non_sensor_columns, pd.DataFrame(scaled_sensor_data, columns=sensor_columns.columns,
                                                                  index=sensor_columns.index)], axis=1)
        #print(scaled_data) # 여기서는 다시 합쳐진 데이터로 나와야 함.

        return scaled_data

    def min_max_scaling(self, column):
        # for reinforcement learning environment.
        min_value = column.min()
        max_value = column.max()
        scaled_column = (column - min_value) / (max_value - min_value)

        # NaN -> 0
        scaled_column = scaled_column.fillna(0)

        return scaled_column

    def merge_dataframe_only_MSE(self, index_names, y_labels, df1):
        y_lr_dfs = [pd.DataFrame(df1, columns=['predicted RUL'])]
        merged_dfs = []

        for i, y_lr_df in enumerate(y_lr_dfs, start=1):
            merged_df = pd.concat([index_names, y_lr_df, y_labels], axis=1)
            merged_dfs.append(merged_df)

        return merged_dfs

    def merge_dataframe(self, index_names, y_labels, df1, df2, df3, df4, df5, df6):
        y_lr_dfs = [pd.DataFrame(df1, columns=['predicted RUL']),
                    pd.DataFrame(df2, columns=['predicted RUL']),
                    pd.DataFrame(df3, columns=['predicted RUL']),
                    pd.DataFrame(df4, columns=['predicted RUL']),
                    pd.DataFrame(df5, columns=['predicted RUL']),
                    pd.DataFrame(df6, columns=['predicted RUL'])]
        merged_dfs = []

        for i, y_lr_df in enumerate(y_lr_dfs, start=1):
            merged_df = pd.concat([index_names, y_lr_df, y_labels], axis=1)
            merged_dfs.append(merged_df)

        return merged_dfs

    def plot_online_RUL_prediction(self, merged_dfs): # 지금 코드 구조에서 바로 쓰지는 못함. 수정해서 써야함.
        for i, merged_df in enumerate(merged_dfs, start=1):
            # unit_number을 기준으로 그룹화
            grouped = merged_df.groupby('unit_number')

            # set subplot
            fig, ax = plt.subplots(figsize=(16, 9))

            # unit_number 별로 그래프 그리기
            for unit, group in grouped:
                ax.plot(group['RUL'], group['predicted RUL'], label=f'Unit {unit}')

            ax.set_xlabel('Remaining Useful Life')
            ax.set_ylabel('Predicted RUL')
            ax.set_title(f'Predicted RUL by Unit Number - Loss function {i}')
            # ax.legend(loc='upper right')  # 범례 추가

            # 그래프 출력 설정
            plt.xlim(350, 0)  # reverse the x-axis so RUL counts down to zero
            plt.ylim(-50, 200)
            plt.show()

    def plot_RUL_prediction_by_q_value(self, environment, weights_by_RL, scale):

        # Calculate predicted RUL by multiplying 's_1' to 's_21' columns with weights_by_RL for all rows
        predicted_RUL_by_Q = np.dot(environment.iloc[:, 5:26], weights_by_RL)

        # Apply scale to predicted RUL values
        predicted_RUL_by_Q *= scale

        # Add a new column 'predicted_RUL_by_Q' to the environment DataFrame
        environment['predicted_RUL_by_Q'] = predicted_RUL_by_Q

        grouped = environment.groupby('unit_number')

        # Set subplot
        fig, ax = plt.subplots(figsize=(16, 9))

        # Plot for each unit_number
        for unit, group in grouped:
            #ax.plot(group['RUL'], group['predicted_RUL_by_Q'], label=f'Unit {unit}')
            ax.plot(group['RUL'], group['predicted_RUL_by_Q'])

        # Draw a red dashed line at y=0
        ax.axhline(y=0, color='r', linestyle='--', label='Q_replace')

        ax.set_xlabel('Remaining Useful Life')
        ax.set_ylabel('Predicted RUL (Q-value)')
        ax.set_title('Predicted RUL by Unit Number - Q-value')
        ax.legend(loc='upper right')  # Add legend

        # Plot settings
        plt.xlim(350, 0)  # Reverse the x-axis so RUL counts down to zero
        #plt.ylim(-50, 200)
        plt.show()

    def plot_RUL_prediction_by_lr_td_loss(self, environment, weights, scale, threshold, is_td_front):

        # Calculate predicted RUL by multiplying 's_0' to 's_21' columns with weights_by_RL for all rows
        predicted_RUL = np.dot(environment.iloc[:, 5:27], weights)

        # Apply scale to predicted RUL values
        predicted_RUL *= scale

        # If is_td_front is True, then predicted_RUL = predicted_RUL + threshold
        print(predicted_RUL)
        print(is_td_front)
        if is_td_front:
            predicted_RUL += threshold # TD loss (front) version. y-axis -> q-hat

        print(predicted_RUL)

        # Add a new column 'predicted_RUL_by_Q' to the environment DataFrame
        environment['predicted_RUL'] = predicted_RUL

        grouped = environment.groupby('unit_number')

        # Set subplot
        fig, ax = plt.subplots(figsize=(16, 9))

        # Plot for each unit_number
        for unit, group in grouped:
            #ax.plot(group['RUL'], group['predicted_RUL_by_Q'], label=f'Unit {unit}')
            ax.plot(group['RUL'], group['predicted_RUL'])

        # Draw a red dashed line at y=threshold
        ax.axhline(y=threshold, color='r', linestyle='--', label='threshold')

        # Plot a purple dashed line with slope -45 degrees passing through (0, 0)
        x_vals = np.linspace(0, 350, 100)  # x-values from -350 to 0
        y_vals = x_vals  # y-values corresponding to y = x
        ax.plot(x_vals, y_vals, 'b--', linewidth=2, label='y = x')  # Plot purple dashed line

        ax.set_xlabel('Remaining Useful Life')
        ax.set_ylabel('Predicted RUL')
        ax.set_title('Predicted RUL by Unit Number')
        ax.legend(loc='upper right')  # Add legend

        # Plot settings
        plt.xlim(350, 0)  # Reverse the x-axis so RUL counts down to zero
        plt.ylim(-350, 400)
        #plt.ylim(-50, 200)
        plt.show()

    def plot_RUL_prediction_by_DCNN(self, environment, threshold):
        # 엔진별로 group.
        grouped = environment.groupby('unit_number')

        # Set subplot
        fig, ax = plt.subplots(figsize=(16, 9))

        # Plot for each unit_number
        for unit, group in grouped:
            ax.plot(group['RUL'], group['predicted_RUL'])

        # Draw a red dashed line at y=threshold
        ax.axhline(y=threshold, color='r', linestyle='--', label='threshold')

        # Plot a purple dashed line with slope -45 degrees passing through (0, 0)
        x_vals = np.linspace(0, 350, 100)  # x-values from -350 to 0
        y_vals = x_vals  # y-values corresponding to y = x
        ax.plot(x_vals, y_vals, 'b--', linewidth=2, label='y = x')  # Plot purple dashed line

        ax.set_xlabel('Remaining Useful Life')
        ax.set_ylabel('Predicted RUL')
        ax.set_title('Predicted RUL by Unit Number')
        ax.legend(loc='upper right')  # Add legend

        # Plot settings
        plt.xlim(350, 0)  # Reverse the x-axis so RUL counts down to zero
        plt.ylim(-350, 400)
        #plt.ylim(-50, 200)
        plt.show()

    def plot_RUL_prediction_by_lr_td_loss_21(self, environment, weights, scale, threshold):

        # Calculate predicted RUL by multiplying 's_0' to 's_21' columns with weights_by_RL for all rows
        predicted_RUL = np.dot(environment.iloc[:, 5:26], weights)
        print(environment)

        # Apply scale to predicted RUL values
        predicted_RUL *= scale

        # Add a new column 'predicted_RUL_by_Q' to the environment DataFrame
        environment['predicted_RUL'] = predicted_RUL

        grouped = environment.groupby('unit_number')

        # Set subplot
        fig, ax = plt.subplots(figsize=(16, 9))

        # Plot for each unit_number
        for unit, group in grouped:
            #ax.plot(group['RUL'], group['predicted_RUL_by_Q'], label=f'Unit {unit}')
            ax.plot(group['RUL'], group['predicted_RUL'])

        # Draw a red dashed line at y=threshold
        ax.axhline(y=threshold, color='r', linestyle='--', label='threshold')

        ax.set_xlabel('Remaining Useful Life')
        ax.set_ylabel('Predicted RUL')
        ax.set_title('Predicted RUL by Unit Number')
        ax.legend(loc='upper right')  # Add legend

        # Plot settings
        plt.xlim(350, 0)  # Reverse the x-axis so RUL counts down to zero
        plt.ylim(-350, 400)
        plt.show()



    def random_obs_simulation_by_threshold(self, merged_dfs, threshold_values, REPLACE_COST, FAILURE_COST):
        """ A method that takes dataframes for each loss function as input, simulates each threshold,
            and returns a list of dataframes for each threshold.

        Args:
            merged_dfs (list) : a list of dataframes for each loss function.
            threshold_values (list) : a list of thresholds in integer format.
            REPLACE_COST (int) : The cost incurred when replacing the engine.
            FAILURE_COST (int) : The cost incurred when engine replacement fails.

        Note:
            It may raise a FutureWarning due to the .append method, but it's safe to ignore as there's no issue.

        :return:
            by_threshold_dfs_list (list) : a list of dataframes for each threshold.
        """

        by_threshold_dfs_list = []

        for merged_df in merged_dfs:
            by_threshold_dfs = []

            for threshold_value in threshold_values:
                cumulative_operation_time = 0
                total_cost = 0
                average_cost = 0  # total cost / operation time
                by_threshold_df = pd.DataFrame(
                    columns=['unit_number', 'actual operation time', 'actual RUL', 'Cumulative Operation Time',
                             'Total Cost'])

                grouped = merged_df.groupby('unit_number')

                for unit, group in grouped:
                    operation_time = None
                    # max_RUL = group['RUL'].max() # RUL으로 계산하면 초기 time_step이 0이 아닌 경우에 max_RUL이 낮게 측정되는 문제가 있음.
                    max_RUL = group['time_cycles'].max()  # 'RUL'을 'time_cycles'로 바꿈.

                    for index, row in group.iterrows():
                        if row['predicted RUL'] <= threshold_value:
                            operation_time = row['time_cycles']
                            cumulative_operation_time += operation_time
                            total_cost += REPLACE_COST
                            break
                    """
                    # 이부분을 자세히 살펴보자. 여기서 문제가 생겼을 것임.
                    for index, row in group.iterrows():
                        if row['predicted RUL'] <= threshold_value:
                            if row['time_cycles'] > max_RUL:
                                operation_time = max_RUL
                            else:
                                operation_time = row['time_cycles']
                            cumulative_operation_time += operation_time
                            total_cost += REPLACE_COST
                            break
                    """
                    # When the operation time is None, it corresponds to the case of a replacement failure.
                    if pd.isna(operation_time):
                        cumulative_operation_time += max_RUL
                        total_cost += FAILURE_COST

                    average_cost = total_cost / cumulative_operation_time  # calculate average cost

                    by_threshold_df = by_threshold_df._append(
                        {'unit_number': unit, 'actual operation time': operation_time,
                         'actual RUL': max_RUL, 'Cumulative Operation Time': cumulative_operation_time,
                         'Average Cost': average_cost, 'Total Cost': total_cost}, ignore_index=True)

                by_threshold_dfs.append(by_threshold_df)
            by_threshold_dfs_list.append(by_threshold_dfs)

        return by_threshold_dfs_list

    def calculate_NoF_AUT_by_threshold(self, by_threshold_dfs_list, threshold_values):
        """ Create a method that takes a list of dataframes, each containing dataframes for different thresholds.
            The method calculates the number of replace failures and the average usage time for each dataframe.
            Finally, it returns a list of dataframes for each loss function.

        Args:
            by_threshold_dfs_list (list) : a list of dataframes for each threshold.
            threshold_values (list) : a list of thresholds in integer format.

        :return:
            by_loss_func_dfs (list) : a list of dataframes for each loss function.
        """

        nan_counts_list = []
        average_cost_lists = []
        average_usage_time_lists = []
        by_loss_func_dfs = []

        # 각 by_threshold_dfs 리스트 내의 by_threshold_df에 대해 반복 수행
        for by_threshold_dfs in by_threshold_dfs_list:
            nan_counts = []
            average_cost_list = []
            average_usage_time_list = []

            # 각 dataframe마다 수행.
            for df in by_threshold_dfs:
                average_usage_time = df['Cumulative Operation Time'].iloc[-1] / (len(df.index))  # 전체 엔진에 대한 평균 사용 시간 저장.
                average_usage_time_list.append(average_usage_time)

                average_cost_by_threshold = df['Average Cost'].iloc[-1]  # 'Average Cost'의 열에서 마지막 값을 가져옴.
                average_cost_list.append(average_cost_by_threshold)

                # 'actual operation time' column의 NaN 개수 count 후 리스트에 추가. index의 끝까지 Nan을 count.
                nan_count = df.loc[:, 'actual operation time'].isna().sum()
                nan_counts.append(nan_count)

            nan_counts_list.append(nan_counts)
            average_cost_lists.append(average_cost_list)
            average_usage_time_lists.append(average_usage_time_list)

        for i, by_threshold_dfs in enumerate(by_threshold_dfs_list, start=1):
            by_loss_func_df = pd.DataFrame(
                {'Threshold': threshold_values, 'Number of replace failures': nan_counts_list[i - 1],
                 'Total Cost': average_cost_lists[i - 1],
                 'Average usage time': average_usage_time_lists[i - 1]})
            by_loss_func_dfs.append(by_loss_func_df)

        return by_loss_func_dfs

    def plot_NoF_AUT_by_threshold(self, by_loss_dfs, dataset_number):
        """ A method to plot the comparison between the number of failures and average usage time.

        Args:
            by_loss_dfs (list) : a list of dataframes for each threshold.
            dataset_number (int): the number of the dataset (1, 2, 3, or 4).
        """

        fig, axs = plt.subplots(len(by_loss_dfs), 1,
                                figsize=(10, 6 * len(by_loss_dfs)))

        for i, by_loss_df in enumerate(by_loss_dfs, start=1):
            ax1 = axs[i - 1]

            # y-axis (left) (Number of failures)
            color = 'tab:blue'
            ax1.set_xlabel('Threshold')
            ax1.set_ylabel('Number of replace failures', color=color)
            ax1.plot(by_loss_df['Threshold'],
                     by_loss_df['Number of replace failures'], color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.yaxis.set_major_locator(MaxNLocator(integer=True))  # 눈금을 정수로 표시
            if dataset_number == 1:
                ax1.set_ylim(-1, 104)
            elif dataset_number == 2:
                ax1.set_ylim(-5, 270)
            elif dataset_number == 3:
                ax1.set_ylim(-1, 104)
            elif dataset_number == 4:
                ax1.set_ylim(-5, 260)

            # y-axis (right) (Average Usage Time)
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Average Usage Time', color=color)
            ax2.plot(by_loss_df['Threshold'], by_loss_df['Average usage time'],
                     color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.yaxis.set_major_locator(MaxNLocator(integer=True))  # 눈금을 정수로 표시
            if dataset_number == 1:
                ax2.set_ylim(150, 220)
            elif dataset_number == 2:
                ax2.set_ylim(135, 220)
            elif dataset_number == 3:
                ax2.set_ylim(190, 260)
            elif dataset_number == 4:
                ax2.set_ylim(180, 260)

            ax1.set_title(f'Number of Failures & Average Usage Time - Loss function {i} (Dataset {dataset_number})')

        plt.tight_layout()  # subplot interval
        plt.show()

    def plot_AC_AUT_by_threshold(self, by_loss_dfs, dataset_number):
        """ A method to plot the comparison between average cost and average usage time.

        Args:
            by_loss_dfs (list) : a list of dataframes for each threshold.
            dataset_number (int): the number of the dataset (1, 2, 3, or 4).
        """

        fig, axs = plt.subplots(len(by_loss_dfs), 1,
                                figsize=(10, 6 * len(by_loss_dfs)))

        for i, by_loss_df in enumerate(by_loss_dfs, start=1):
            ax1 = axs[i - 1]

            # y-axis (left) (Average Cost)
            color = 'tab:green'
            ax1.set_xlabel('Threshold')
            ax1.set_ylabel('Average Cost', color=color)
            ax1.plot(by_loss_df['Threshold'],
                     by_loss_df['Total Cost'], color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.yaxis.set_major_locator(MaxNLocator(integer=True))  # 눈금을 정수로 표시
            if dataset_number == 1:
                ax1.set_ylim(0, 60)
            elif dataset_number == 2:
                ax1.set_ylim(0, 60)
            elif dataset_number == 3:
                ax1.set_ylim(0, 60)
            elif dataset_number == 4:
                ax1.set_ylim(0, 60)

            # y-axis (right) (Average Usage Time)
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Average Usage Time', color=color)
            ax2.plot(by_loss_df['Threshold'], by_loss_df['Average usage time'],
                     color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.yaxis.set_major_locator(MaxNLocator(integer=True))  # 눈금을 정수로 표시
            if dataset_number == 1:
                ax2.set_ylim(150, 220)
            elif dataset_number == 2:
                ax2.set_ylim(135, 220)
            elif dataset_number == 3:
                ax2.set_ylim(190, 260)
            elif dataset_number == 4:
                ax2.set_ylim(180, 260)

            ax1.set_title(f'Average Cost & Average Usage Time - Loss function {i} (Dataset {dataset_number})')

        plt.tight_layout()  # subplot interval
        plt.show()
    def plot_simulation_results(self, by_loss_dfs, dataset_number):
        """ Displaying simulation results in curve forms for each loss function

        Args:
            by_loss_dfs (list) : a list of dataframes for each threshold.
            dataset_number (int): the number of the dataset (1, 2, 3, or 4).
        """

        # Create a list of colors for each dataframe
        colors = ['blue', 'green', 'red', 'yellow', 'pink', 'purple']

        fig, ax = plt.subplots(figsize=(10, 6))

        # Loop through the dataframes and plot scatter points with different colors
        for i, by_loss_df in enumerate(by_loss_dfs):
            ax.plot(
                by_loss_df['Number of replace failures'],
                by_loss_df['Average usage time'],
                '+-',
                label=f'Loss function {i + 1}',
                color=colors[i],
                alpha=0.5
            )

        # Set labels and title based on dataset number
        dataset_name = f"Dataset {dataset_number}"
        ax.set_title(f'Number of Failures vs. Average usage time ({dataset_name})')

        # Set y-axis and x-axis limits based on dataset number
        if dataset_number == 1:
            ax.set_ylim(0, 200)
            ax.set_xlim(-5, 105)
        elif dataset_number == 2:
            ax.set_ylim(135, 220)
            ax.set_xlim(-5, 265)
        elif dataset_number == 3:
            ax.set_ylim(190, 248)
            ax.set_xlim(-5, 105)
        elif dataset_number == 4:
            ax.set_ylim(180, 260)
            ax.set_xlim(-5, 255)

        # Set labels and legend
        ax.set_xlabel('Number of replace failures')
        ax.set_ylabel('Average usage time')
        ax.legend()

        ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # Set integer ticks for y-axes

        ax.set_aspect('equal')   # Set aspect ratio to make the plot square

        plt.tight_layout()
        plt.show()

    def plot_simulation_results_scale_up(self, by_loss_dfs, dataset_number, loss_labels, isMSE):
        """ Displaying simulation results in curve forms for each loss function

        Args:
            by_loss_dfs (list) : a list of dataframes for each threshold.
            dataset_number (int): the number of the dataset (1, 2, 3, or 4).
            loss_labels (list) : a list of labels for the loss.
        """

        # Create a list of colors for each dataframe
        if isMSE == True:
            colors = ['blue']
        else:
            colors = ['blue', 'green', 'red', 'yellow', 'pink', 'purple']

        fig, ax = plt.subplots(figsize=(6, 6))

        # Loop through the dataframes and plot scatter points with different colors
        for i, by_loss_df in enumerate(by_loss_dfs):
            ax.plot(
                by_loss_df['Number of replace failures'],
                by_loss_df['Average usage time'],
                '+-',
                label=loss_labels[i],  # use the custom label
                color=colors[i],
                alpha=0.5
            )

        # Set labels and title based on dataset number
        dataset_name = f"Dataset {dataset_number}"
        ax.set_title(f'Number of Failures vs. Average usage time ({dataset_name})')

        # Set y-axis and x-axis limits based on dataset number
        if dataset_number == 1:
            ax.set_ylim(176, 189)
            ax.set_xlim(-0.5, 60)
        elif dataset_number == 2: # Tentative value
            ax.set_ylim(135, 220)
            ax.set_xlim(-5, 265)
        elif dataset_number == 3: # Tentative value
            ax.set_ylim(190, 248)
            ax.set_xlim(-5, 105)
        elif dataset_number == 4: # Tentative value
            ax.set_ylim(180, 260)
            ax.set_xlim(-5, 255)

        # Set labels and legend
        ax.set_xlabel('Number of replace failures')
        ax.set_ylabel('Average usage time')
        ax.legend()

        ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # Set integer ticks for y-axes

        plt.tight_layout()
        plt.show()


    def plot_simulation_results_x_y_swap(self, by_loss_dfs, dataset_number, loss_labels, max_engine):
        """ Displaying simulation results in curve forms for each loss function (swap x, y axis)

        Args:
            by_loss_dfs (list) : a list of dataframes for each threshold.
            dataset_number (int): the number of the dataset (1, 2, 3, or 4).
            loss_labels (list) : a list of labels for the loss.
            max_engine (int) : a number of engines
        """

        # Create a list of colors for each dataframe
        colors = ['blue', 'green', 'red', 'yellow', 'pink', 'purple']

        fig, ax = plt.subplots(figsize=(6, 6))

        # Loop through the dataframes and plot scatter points with different colors
        for i, by_loss_df in enumerate(by_loss_dfs):
            ax.plot(
                by_loss_df['Average usage time'],
                by_loss_df['Number of replace failures'] / max_engine, # Divide by 'max_engine' to get the failure rate
                '+-',
                label=loss_labels[i],  # use the custom label
                color=colors[i],
                alpha=0.5
            )

        # Set labels and title based on dataset number
        dataset_name = f"Dataset {dataset_number}"
        ax.set_title(f'Average usage time vs. Failure rate ({dataset_name})')

        # Set y-axis and x-axis limits based on dataset number
        if dataset_number == 1:
            #ax.set_xlim(176, 189)
            ax.set_xlim(0, 205)
            ax.set_ylim(0, 1)
        elif dataset_number == 2: # Tentative value
            ax.set_ylim(135, 220)
            ax.set_xlim(-5, 265)
        elif dataset_number == 3: # Tentative value
            ax.set_ylim(190, 248)
            ax.set_xlim(-5, 105)
        elif dataset_number == 4: # Tentative value
            ax.set_ylim(180, 260)
            ax.set_xlim(-5, 255)

        # Set labels and legend
        ax.set_xlabel('Average usage time')
        ax.set_ylabel('Failure rate')
        ax.legend()

        #ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # Set integer ticks for y-axes

        plt.tight_layout()
        plt.show()

    def plot_simulation_results_x_y_swap_point_lambda(self, by_loss_dfs, dataset_number, loss_labels, max_engine, AUT_pi, failure_rate_pi):
        """ Displaying simulation results in curve forms for each loss function (swap x, y axis)

        Args:
            by_loss_dfs (list) : a list of dataframes for each threshold.
            dataset_number (int): the number of the dataset (1, 2, 3, or 4).
            loss_labels (list) : a list of labels for the loss.
            max_engine (int) : a number of engines.
            AUT_pi (float) : average usage time of optimal point.
            failure_rate_pi (float) : failure rate of optimal point.
        """

        # Create a list of colors for each dataframe
        colors = ['blue', 'green', 'red', 'yellow', 'pink', 'purple']

        fig, ax = plt.subplots(figsize=(6, 6))

        # Loop through the dataframes and plot scatter points with different colors
        for i, by_loss_df in enumerate(by_loss_dfs):
            ax.plot(
                by_loss_df['Average usage time'],
                by_loss_df['Number of replace failures'] / max_engine, # Divide by 'max_engine' to get the failure rate
                '+-',
                label=loss_labels[i],  # use the custom label
                color=colors[i],
                alpha=0.5
            )

        # Plot the point (AUT_pi, failure_rate_pi)
        ax.plot(AUT_pi, failure_rate_pi, 'ro', label='optimal point')

        # Plot the line connecting origin and point (AUT_pi, failure_rate_pi)
        #beta = failure_rate_pi / AUT_pi
        beta = 0.00078352

        # Plot line connecting origin and min point
        x_vals = np.array([0, 200])
        y_vals = beta * x_vals - 0.11111159
        ax.plot(x_vals, y_vals, 'r--', label=f'Beta: {beta:.8f}')

        # Set labels and title based on dataset number
        dataset_name = f"Dataset {dataset_number}"
        ax.set_title(f'Average usage time vs. Failure rate ({dataset_name})')

        # Set y-axis and x-axis limits based on dataset number
        if dataset_number == 1:
            #ax.set_xlim(0, 200)
            #ax.set_ylim(0, 1)
            ax.set_xlim(120, 170)
            ax.set_ylim(0, 0.03)
        elif dataset_number == 2: # Tentative value
            ax.set_ylim(135, 220)
            ax.set_xlim(-5, 265)
        elif dataset_number == 3: # Tentative value
            ax.set_ylim(190, 248)
            ax.set_xlim(-5, 105)
        elif dataset_number == 4: # Tentative value
            ax.set_ylim(180, 260)
            ax.set_xlim(-5, 255)

        # Set labels and legend
        ax.set_xlabel('Average usage time')
        ax.set_ylabel('Failure rate')
        ax.legend()

        #ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # Set integer ticks for y-axes

        plt.tight_layout()
        plt.show()

    def plot_simulation_results_x_y_swap_point_lambda_2(self, by_loss_dfs, dataset_number, loss_labels, max_engine):
        """ Displaying simulation results in curve forms for each loss function (swap x, y axis)

        Args:
            by_loss_dfs (list) : a list of dataframes for each threshold.
            dataset_number (int): the number of the dataset (1, 2, 3, or 4).
            loss_labels (list) : a list of labels for the loss.
        """

        # Create a list of colors for each dataframe
        colors = ['blue', 'green', 'red', 'yellow', 'pink', 'purple']

        fig, ax = plt.subplots(figsize=(6, 6))

        # Loop through the dataframes and plot scatter points with different colors
        for i, by_loss_df in enumerate(by_loss_dfs):
            ax.plot(
                by_loss_df['Average usage time'],
                by_loss_df['Number of replace failures'] / max_engine,  # Divide by 'max_engine' to get the failure rate
                '+-',
                label=loss_labels[i],  # use the custom label
                color=colors[i],
                alpha=0.5
            )

        # Plot the points (AUT, failure_rate)
        points = [(159.90, 0.0125), (161.2544, 0.0139), (178.56, 0.046), (193.5445, 0.271), (195.644, 0.4529), (160.411, 0.0135)]
        labels = ['alpha 1.0 (lambda : 6.957364)', 'alpha 0.9', 'alpha 0.5', 'alpha 0.1', 'alpha 0.0', 'Optimal of MSE (lambda : 6.991415)']

        #points = [(159.90, 0.0125), (160.411, 0.0135)]
        #labels = ['Optimal policy of RL (lambda : 6.957364)', 'Optimal of MSE (lambda : 6.991415)']

        for i, (point, label) in enumerate(zip(points, labels)):
            ax.plot(point[0], point[1], marker='o', markersize=6, label=label, color='C{}'.format(i))  # each point
            #ax.plot(point[0], point[1], marker='o', markersize=6, color='C{}'.format(i))

        #for i, (point, label) in enumerate(zip(points, labels)):
        #    #ax.plot(point[0], point[1], marker='o', markersize=6, label=label, color='C{}'.format(i))  # each point
        #    ax.plot(point[0], point[1], marker='o', markersize=6, color='C{}'.format(i))

        # Add legend
        ax.legend(fontsize='1')

        # Plot the line connecting origin and point (AUT_pi, failure_rate_pi)
        beta = 0.0007730405

        # Plot line connecting origin and min point
        x_vals = np.array([0, 205])
        y_vals = beta * x_vals - 0.11110917595
        ax.plot(x_vals, y_vals, 'r--', label=f'Beta: {beta:.9f}')

        # Set labels and title based on dataset number
        dataset_name = f"Dataset {dataset_number}"
        ax.set_title(f'Average usage time vs. Failure rate ({dataset_name})')

        # Set y-axis and x-axis limits based on dataset number
        if dataset_number == 1:
            #ax.set_xlim(0, 200)
            #ax.set_ylim(0, 0.4)
            #ax.set_xlim(160, 200)
            ax.set_xlim(150, 170)
            #ax.set_xlim(130, 170)
            ax.set_ylim(-0.01, 0.03)
        elif dataset_number == 2:  # Tentative value
            ax.set_ylim(135, 220)
            ax.set_xlim(-5, 265)
        elif dataset_number == 3:  # Tentative value
            ax.set_ylim(190, 248)
            ax.set_xlim(-5, 105)
        elif dataset_number == 4:  # Tentative value
            ax.set_ylim(180, 260)
            ax.set_xlim(-5, 255)

        # Set labels and legend
        ax.set_xlabel('Average usage time')
        ax.set_ylabel('Failure rate')
        ax.legend()

        # ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # Set integer ticks for y-axes

        plt.tight_layout()
        plt.show()

    def plot_simulation_results_x_y_swap_point_td_loss(self, by_loss_dfs, dataset_number, loss_labels, max_engine):
        """ Displaying simulation results in curve forms for each loss function (swap x, y axis)

        Args:
            by_loss_dfs (list) : a list of dataframes for each threshold.
            dataset_number (int): the number of the dataset (1, 2, 3, or 4).
            loss_labels (list) : a list of labels for the loss.
        """

        # Create a list of colors for each dataframe
        colors = ['blue', 'green', 'red', 'yellow', 'pink', 'purple']

        fig, ax = plt.subplots(figsize=(6, 6))

        # Loop through the dataframes and plot scatter points with different colors
        for i, by_loss_df in enumerate(by_loss_dfs):
            ax.plot(
                by_loss_df['Average usage time'],
                by_loss_df['Number of replace failures'] / max_engine,  # Divide by 'max_engine' to get the failure rate
                '+-',
                label=loss_labels[i],  # use the custom label
                color=colors[i],
                alpha=0.5
            )

        # Plot the points (AUT, failure_rate)
        #points = [(159.135, 0.012), (151.239, 0.0255), (197.3625, 1), (197.3625, 1), (197.163, 0.785), (193.440, 0.249)] # for td ratio loss
        #points = [(159.135, 0.012), (197.3625, 1), (197.3625, 1), (197.3575, 0.984), (196.919, 0.699), (193.440, 0.249)] # for td loss
        #labels = ['beta : 0.00077362', 'alpha 1.0', 'alpha 0.8', 'alpha 0.5', 'alpha 0.2', 'alpha 0.0']
        points = [(159.90, 0.0125), (161.2544, 0.0139), (178.56, 0.046), (193.5445, 0.271), (195.644, 0.4529), (160.411, 0.0135)]
        labels = ['alpha 1.0 (lambda : 6.957364)', 'alpha 0.9', 'alpha 0.5', 'alpha 0.1', 'alpha 0.0', 'Optimal of MSE (lambda : 6.991415)']
        beta = [0.0007730405]

        for i, (point, label) in enumerate(zip(points, labels)):
            ax.plot(point[0], point[1], marker='o', markersize=6, label=label, color='C{}'.format(i))  # each point
            #ax.plot(point[0], point[1], marker='o', markersize=6, color='C{}'.format(i))

        # Add legend
        ax.legend(fontsize='1')

        # Plot the line connecting origin and point (AUT_pi, failure_rate_pi)
        beta = 0.0007736268

        # Plot line connecting origin and min point
        x_vals = np.array([0, 205])
        y_vals = beta * x_vals - 0.11110917595
        ax.plot(x_vals, y_vals, 'r--', label=f'Beta: {beta:.9f}')

        # Set labels and title based on dataset number
        dataset_name = f"Dataset {dataset_number}"
        ax.set_title(f'Average usage time vs. Failure rate ({dataset_name})')

        # Set y-axis and x-axis limits based on dataset number
        if dataset_number == 1:
            #ax.set_xlim(0, 200)
            ax.set_ylim(0, 1)
            #ax.set_xlim(160, 200)
            #ax.set_xlim(150, 170)
            #ax.set_xlim(130, 170)
            #ax.set_ylim(-0.01, 0.03)
        elif dataset_number == 2:  # Tentative value
            ax.set_ylim(135, 220)
            ax.set_xlim(-5, 265)
        elif dataset_number == 3:  # Tentative value
            ax.set_ylim(190, 248)
            ax.set_xlim(-5, 105)
        elif dataset_number == 4:  # Tentative value
            ax.set_ylim(180, 260)
            ax.set_xlim(-5, 255)

        # Set labels and legend
        ax.set_xlabel('Average usage time')
        ax.set_ylabel('Failure rate')
        ax.legend()

        # ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # Set integer ticks for y-axes

        plt.tight_layout()
        plt.show()

    def plot_simulation_results_x_y_swap_cost(self, by_loss_dfs, dataset_number, loss_labels, max_engine,
                                              c_replace, c_failure):
        """ Displaying simulation results in curve forms for each loss function (swap x, y axis)

        Args:
            by_loss_dfs (list) : a list of dataframes for each threshold.
            dataset_number (int): the number of the dataset (1, 2, 3, or 4).
            loss_labels (list) : a list of labels for the loss.
            max_engine (int) : a number of engines
            c_replace (int) : replacement cost
            c_failure (int) : System failure cost
        """

        # Create a list of colors for each dataframe
        colors = ['blue', 'green', 'red', 'yellow', 'pink', 'purple']

        fig, ax = plt.subplots(figsize=(6, 6))

        # Initialize variables to store the minimum actual cost per usage time
        min_actual_cost_per_usage_time = float('inf')
        min_point = None

        # Loop through the dataframes and plot scatter points with different colors
        for i, by_loss_df in enumerate(by_loss_dfs):
            # Divide by 'max_engine' to get the failure rate
            failure_rate = by_loss_df['Number of replace failures'] / max_engine
            actual_cost = c_replace + failure_rate * (c_failure - c_replace)

            ax.plot(
                by_loss_df['Average usage time'],
                actual_cost,
                '+-',
                label=loss_labels[i],  # use the custom label
                color=colors[i],
                alpha=0.5
            )

            # Calculate actual cost per usage time
            actual_cost_per_usage_time = actual_cost / by_loss_df['Average usage time']

            # Find the minimum actual cost per usage time
            min_index = np.argmin(actual_cost_per_usage_time)
            min_cost = actual_cost_per_usage_time[min_index]

            if min_cost < min_actual_cost_per_usage_time:
                min_actual_cost_per_usage_time = min_cost
                min_point = (by_loss_df['Average usage time'].iloc[min_index], actual_cost.iloc[min_index])

        # Plot the minimum point
        if min_point:
            ax.plot(min_point[0], min_point[1], 'ro', label='Minimum cost per time (MSE)')

            # Calculate slope of the line connecting the origin and the min point
            slope = min_point[1] / min_point[0]

            # Plot line connecting origin and min point
            x_vals = np.array([0, 200])
            # x_vals = np.linspace(0, min_point[0], 100)
            y_vals = slope * x_vals
            ax.plot(x_vals, y_vals, 'r--', label=f'Lambda: {slope:.7f}')

            # Print coordinates of the min point and slope
            min_text = f"({min_point[0]:.3f}, {min_point[1]:.3f})"
            ax.text(min_point[0], min_point[1], min_text, verticalalignment='bottom', horizontalalignment='right')



        # Set labels and title based on dataset number
        dataset_name = f"Dataset {dataset_number}"
        ax.set_title(f'Average usage time vs. Actual Cost ({dataset_name})')

        # Set y-axis and x-axis limits based on dataset number
        if dataset_number == 1:
            #ax.set_xlim(176, 189)
            ax.set_xlim(0, 205)
            #ax.set_ylim(0, 1)
        elif dataset_number == 2: # Tentative value
            ax.set_ylim(135, 220)
            ax.set_xlim(-5, 265)
        elif dataset_number == 3: # Tentative value
            ax.set_ylim(190, 248)
            ax.set_xlim(-5, 105)
        elif dataset_number == 4: # Tentative value
            ax.set_ylim(180, 260)
            ax.set_xlim(-5, 255)

        # Set labels and legend
        ax.set_xlabel('Average usage time')
        ax.set_ylabel('Actual Cost')
        ax.legend()

        #ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # Set integer ticks for y-axes

        plt.tight_layout()
        plt.show()
    def plot_simulation_results_x_y_swap_cost_scale_up(self, by_loss_dfs, dataset_number, loss_labels, max_engine,
                                              c_replace, c_failure):
        """ Displaying simulation results in curve forms for each loss function (swap x, y axis)

        Args:
            by_loss_dfs (list) : a list of dataframes for each threshold.
            dataset_number (int): the number of the dataset (1, 2, 3, or 4).
            loss_labels (list) : a list of labels for the loss.
            max_engine (int) : a number of engines
            c_replace (int) : replacement cost
            c_failure (int) : System failure cost
        """

        # Create a list of colors for each dataframe
        colors = ['blue', 'green', 'red', 'yellow', 'pink', 'purple']

        fig, ax = plt.subplots(figsize=(6, 6))

        # Initialize variables to store the minimum actual cost per usage time
        min_actual_cost_per_usage_time = float('inf')
        min_point = None

        # Loop through the dataframes and plot scatter points with different colors
        for i, by_loss_df in enumerate(by_loss_dfs):
            # Divide by 'max_engine' to get the failure rate
            failure_rate = by_loss_df['Number of replace failures'] / max_engine
            actual_cost = c_replace + failure_rate * (c_failure - c_replace)

            ax.plot(
                by_loss_df['Average usage time'],
                actual_cost,
                '+-',
                label=loss_labels[i],  # use the custom label
                color=colors[i],
                alpha=0.5
            )

            # Calculate actual cost per usage time
            actual_cost_per_usage_time = actual_cost / by_loss_df['Average usage time']

            # Find the minimum actual cost per usage time
            min_index = np.argmin(actual_cost_per_usage_time)
            min_cost = actual_cost_per_usage_time[min_index]

            if min_cost < min_actual_cost_per_usage_time:
                min_actual_cost_per_usage_time = min_cost
                min_point = (by_loss_df['Average usage time'].iloc[min_index], actual_cost.iloc[min_index])

            # Plot the minimum point
        if min_point:
            ax.plot(min_point[0], min_point[1], 'ro', label='Minimum cost per time (MSE)')

            # Calculate slope of the line connecting the origin and the min point
            slope = min_point[1] / min_point[0]

            # Plot line connecting origin and min point
            x_vals = np.array([0, 200])
            #x_vals = np.linspace(0, min_point[0], 100)
            y_vals = slope * x_vals
            ax.plot(x_vals, y_vals, 'r--', label=f'Lambda: {slope:.7f}')

            # Print coordinates of the min point and slope
            min_text = f"({min_point[0]:.3f}, {min_point[1]:.3f})"
            ax.text(min_point[0], min_point[1], min_text, verticalalignment='bottom', horizontalalignment='right')


        # Set labels and title based on dataset number
        dataset_name = f"Dataset {dataset_number}"
        ax.set_title(f'Average usage time vs. Actual Cost ({dataset_name})')

        # Set y-axis and x-axis limits based on dataset number
        if dataset_number == 1:
            #ax.set_xlim(120, 205)
            ax.set_xlim(140, 170)
            ax.set_ylim(1000, 1250)
            #ax.set_ylim(0, 2000)
        elif dataset_number == 2: # Tentative value
            ax.set_ylim(135, 220)
            ax.set_xlim(-5, 265)
        elif dataset_number == 3: # Tentative value
            ax.set_ylim(190, 248)
            ax.set_xlim(-5, 105)
        elif dataset_number == 4: # Tentative value
            ax.set_ylim(180, 260)
            ax.set_xlim(-5, 255)

        # Set labels and legend
        ax.set_xlabel('Average usage time')
        ax.set_ylabel('Actual Cost')
        ax.legend()

        #ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # Set integer ticks for y-axes

        plt.tight_layout()
        plt.show()

    def plot_RL_results_scale_up(self, by_loss_dfs, dataset_number, loss_labels, number_of_failure, average_usage_time, r_continue):
        """ Displaying simulation results in curve forms for each loss function

        Args:
            by_loss_dfs (list) : a list of dataframes for each threshold.
            dataset_number (int): the number of the dataset (1, 2, 3, or 4).
            loss_labels (list) : a list of labels for the loss.
        """

        # Create a list of colors for each dataframe
        colors = ['blue', 'green', 'red', 'yellow', 'pink', 'purple']

        fig, ax = plt.subplots(figsize=(6, 6))

        # Loop through the dataframes and plot scatter points with different colors
        for i, by_loss_df in enumerate(by_loss_dfs):
            ax.plot(
                by_loss_df['Number of replace failures'],
                by_loss_df['Average usage time'],
                '+-',
                label=loss_labels[i],  # use the custom label
                color=colors[i],
                alpha=0.5
            )
        print("Number of failures:", number_of_failure)
        print("Average usage time:", average_usage_time)

        # Plot additional points for number_of_failure and average_usage_time
        ax.scatter(number_of_failure, average_usage_time, color='black', label=r_continue)

        # Set labels and title based on dataset number
        dataset_name = f"Dataset {dataset_number}"
        ax.set_title(f'Number of Failures vs. Average usage time ({dataset_name})')

        # Set y-axis and x-axis limits based on dataset number
        if dataset_number == 1:
            ax.set_ylim(176, 189)
            ax.set_ylim(85, 105)
            ax.set_xlim(-0.5, 60)
        elif dataset_number == 2: # Tentative value
            ax.set_ylim(135, 220)
            ax.set_xlim(-5, 265)
        elif dataset_number == 3: # Tentative value
            ax.set_ylim(190, 248)
            ax.set_xlim(-5, 105)
        elif dataset_number == 4: # Tentative value
            ax.set_ylim(180, 260)
            ax.set_xlim(-5, 255)

        # Set labels and legend
        ax.set_xlabel('Number of replace failures')
        ax.set_ylabel('Average usage time')
        ax.legend()

        ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # Set integer ticks for y-axes

        plt.tight_layout()
        plt.show()

    def calculate_average_performance(self, by_loss_dfs_list, number_of_samples):
        """ A method that takes results for multiple samples as input,
            calculates the average for each loss function, and returns the final list.

        Args:
            by_loss_dfs_list (list) : a list of dataframes for each sample.

        :return:
            by_loss_func_dfs (list) : a list of dataframes for each loss function.
        """
        result_dfs = []

        for col_idx in range(5):  # 열 인덱스 0부터 5까지
            sum_df = by_loss_dfs_list[0][col_idx].copy()

            for sample_idx in range(1, number_of_samples):
                sum_df += by_loss_dfs_list[sample_idx][col_idx]

            result_df = sum_df / number_of_samples
            result_dfs.append(result_df)

        return result_dfs

    def calculate_average_performance_only_MSE(self, by_loss_dfs_list, number_of_samples):
        """ A method that takes results for multiple samples as input,
            calculates the average for each loss function, and returns the final list.

        Args:
            by_loss_dfs_list (list) : a list of dataframes for each sample.

        :return:
            by_loss_func_dfs (list) : a list of dataframes for each loss function.
        """
        result_dfs = []

        for col_idx in range(1):  # 열 인덱스 0부터 5까지
            sum_df = by_loss_dfs_list[0][col_idx].copy()

            for sample_idx in range(1, number_of_samples):
                sum_df += by_loss_dfs_list[sample_idx][col_idx]

            result_df = sum_df / number_of_samples
            result_dfs.append(result_df)

        return result_dfs


    def predict_and_save(self, model_instance, train_data, valid_data, full_data):
        """ A method that returns the predictions for each dataset.

        Args:
            model_instance (instance) : instance of a prediction model.
            train_data (dataframe) : train data.
            valid_data (dataframe) : train data.
            full_data (dataframe) : train + valid data.

        :return:
            train_predictions (dataframe)
            valid_predictions (dataframe)
            full_predictions (dataframe)
        """
        train_predictions = model_instance.predict(train_data)
        valid_predictions = model_instance.predict(valid_data)
        full_predictions = model_instance.predict(full_data)

        return train_predictions, valid_predictions, full_predictions

    """
    def plot_average_reward(self, max_episodes, average_rewards):
        # Reinforcement Learning
        plt.plot(range(1, max_episodes + 1), average_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Average Reward per Episode')
        plt.show()
    """

    def plot_average_reward(self, max_episodes, number_of_sample, average_rewards):
        # Reinforcement Learning
        averaged_rewards = [sum(average_rewards[i:i + number_of_sample]) / number_of_sample for i in range(0, len(average_rewards), number_of_sample)]
        plt.plot(range(1, len(averaged_rewards) + 1), averaged_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Average Reward per Episode (1 episode : all samples)')
        plt.show()
    def plot_actual_average_reward(self, max_episodes, number_of_sample, average_rewards):
        # Reinforcement Learning
        averaged_rewards = [sum(average_rewards[i:i + number_of_sample]) / number_of_sample for i in range(0, len(average_rewards), number_of_sample)]
        plt.plot(range(1, len(averaged_rewards) + 1), averaged_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Actual Average Reward')
        plt.title('Actual Average Reward per Episode (1 episode : all samples)')
        plt.show()

    def plot_training_loss(self, max_episodes, number_of_sample, training_loss):
        # Reinforcement Learning
        averaged_loss = [sum(training_loss[i:i + number_of_sample]) / number_of_sample for i in range(0, len(training_loss), number_of_sample)]
        plt.plot(range(1, len(averaged_loss) + 1), averaged_loss)
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Loss per Episode (1 episode : all samples)')
        plt.show()

    def plot_number_of_observation(self, max_episodes, average_number_of_observations):
        plt.plot(range(1, max_episodes + 1), average_number_of_observations)
        plt.xlabel('Episode')
        plt.ylabel('Average Number of Observations')
        plt.title('Average Number of Observations per Episode')
        plt.show()

