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

    def min_max_scaling(self, column):
        # for reinforcement learning environment.
        min_value = column.min()
        max_value = column.max()
        scaled_column = (column - min_value) / (max_value - min_value)

        # NaN -> 0
        scaled_column = scaled_column.fillna(0)

        return scaled_column

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

    def plot_online_RUL_prediction(self, merged_dfs):
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
                    max_RUL = group['RUL'].max()
                    """
                    for index, row in group.iterrows():
                        if row['predicted RUL'] <= threshold_value:
                            operation_time = row['time_cycles']
                            cumulative_operation_time += operation_time
                            total_cost += REPLACE_COST
                            break
                    """
                    for index, row in group.iterrows():
                        if row['predicted RUL'] <= threshold_value:
                            if row['time_cycles'] > max_RUL:
                                operation_time = max_RUL
                            else:
                                operation_time = row['time_cycles']
                            cumulative_operation_time += operation_time
                            total_cost += REPLACE_COST
                            break

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
            ax.set_ylim(150, 190)
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

    def plot_simulation_results_scale_up(self, by_loss_dfs, dataset_number, loss_labels):
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

        for col_idx in range(6):  # 열 인덱스 0부터 5까지
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

    def plot_average_reward(self, max_episodes, average_rewards):
        # Reinforcement Learning
        plt.plot(range(1, max_episodes + 1), average_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Average Reward per Episode')
        plt.show()

    def plot_training_loss(self, max_episodes, training_loss):
        # Reinforcement Learning
        plt.plot(range(1, max_episodes + 1), training_loss)
        plt.xlabel('Episode')
        plt.ylabel('loss')
        plt.title('Loss per Episode')
        plt.show()

    def plot_number_of_observation(self, max_episodes, average_number_of_observations):
        plt.plot(range(1, max_episodes + 1), average_number_of_observations)
        plt.xlabel('Episode')
        plt.ylabel('Average Number of Observations')
        plt.title('Average Number of Observations per Episode')
        plt.show()










