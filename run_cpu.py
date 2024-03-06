# General lib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import configparser
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pickle        # it is suitable for saving Python objects.
import warnings

# Custom .py
from linear_regression_TD import Linear_Regression_TD
from simulation_env import SimulationEnvironment
from loss import directed_mse_loss
from loss import different_td_loss
from loss import previous_prediction, previous_true_label  # use global variable

# Filter out the warning
warnings.filterwarnings("ignore", message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*")

# generate configuration instance
config = configparser.ConfigParser()

# a list to store dataframes containing simulation results for each sample
full_by_loss_dfs_list = []
average_by_loss_dfs = []

class RunCPU():
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

        self.td_alpha = float(config['SimulationSettings']['td_alpha'])


        # constant of simulation
        self.threshold_start = int(config['SimulationSettings']['threshold_start'])
        self.threshold_end = int(config['SimulationSettings']['threshold_end'])
        self.threshold_values = list(range(self.threshold_start, self.threshold_end + 1))

        # Define cost
        self.REPLACE_COST = int(config['SimulationSettings']['REPLACE_COST'])
        self.FAILURE_COST = int(config['SimulationSettings']['FAILURE_COST'])

        # class instance 생성
        self.env = SimulationEnvironment()
        # dataset 분할
        self.train_data, self.valid_data, self.full_data = self.env.load_data(self.num_dataset, self.split_unit_number)
        # sampling
        self.sampled_datasets = self.env.sampling_datasets(self.num_sample_datasets, self.observation_probability,
                                                      self.train_data, self.valid_data, self.full_data)
        # sampled_datasets에 RUL column 추가.
        self.sampled_datasets_with_RUL = self.env.add_RUL_column_to_sampled_datasets(self.sampled_datasets)

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
        y_train = y_train.clip(upper = 195)

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
                                                                      self.REPLACE_COST, self.FAILURE_COST)
        # Organize learning results by loss function
        full_by_loss_dfs = self.env.calculate_NoF_AUT_by_threshold(full_by_threshold_dfs_list, self.threshold_values)

        # save the learning results to a global variable
        full_by_loss_dfs_list.append(full_by_loss_dfs)

        #self.env.plot_NoF_AUT_by_threshold(full_by_loss_dfs, self.num_dataset)
        #self.env.plot_AC_AUT_by_threshold(full_by_loss_dfs, self.num_dataset)

        #self.env.plot_simulation_results(full_by_loss_dfs, self.num_dataset)
        #self.env.plot_simulation_results_scale_up(full_by_loss_dfs, self.num_dataset, self.loss_labels)

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

runCPU = RunCPU('config1.ini')
runCPU.run_many()

"""
   * Later, you can load the file to retrieve the data.

# Load average_by_loss_dfs from the file
with open('average_by_loss_dfs.pkl', 'rb') as f:
    average_by_loss_dfs = pickle.load(f)

"""




