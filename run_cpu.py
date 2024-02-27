# General lib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import configparser
import tensorflow as tf
from tensorflow.keras import layers, models

# Custom .py
from linear_regression_TD import Linear_Regression_TD
from simulation_env import SimulationEnvironment
from loss import directed_mse_loss


# generate configuration instance
config = configparser.ConfigParser()

class RunCPU():
    def __init__(self, config_path):
        config.read(config_path)
        # number of dataset (1, 2, 3, 4)
        self.num_dataset = int(config['SimulationSettings']['num_dataset'])
        # The boundary value of unit numbers for dividing the train and valid datasets
        self.split_unit_number = int(config['SimulationSettings']['split_unit_number'])
        # Number of sample datasets
        self.num_sample_datasets = int(config['SimulationSettings']['num_sample_datasets'])
        # Randomly extract only a subset of observational probability data from the entire dataset
        self.observation_probability = float(config['SimulationSettings']['observation_probability'])
        # Constant for crucial_moment loss
        self.crucial_moment = int(config['SimulationSettings']['crucial_moment'])

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

        y_lr1_train = lr1.predict(x_train)  # Prediction on train data
        y_lr1_valid = lr1.predict(x_valid)  # Prediction on validation data
        y_lr1_full = lr1.predict(x_full)    # Prediction on full data

        # 2. Crucial moments loss function - Linear Regression ##########################
        # Filter and save only data that is less than crucial moment constant.
        filtered_data = x_train[y_train <= self.crucial_moment]
        filtered_labels = y_train[y_train <= self.crucial_moment]

        lr2 = LinearRegression()
        lr2.fit(filtered_data, filtered_labels)  # Fitting

        y_lr2_train = lr2.predict(x_train)  # Prediction on train data
        y_lr2_valid = lr2.predict(x_valid)  # Prediction on validation data
        y_lr2_full = lr2.predict(x_full)    # Prediction on full data

        # 3. TD Loss function - Linear Regression (ridge) ################################
        lr3 = Linear_Regression_TD()
        lr3.fit(x_train, y_train, 0.5, 10)  # Fitting; fit(X, Y, alpha, lambda)

        y_lr3_train = lr3.predict(x_train)  # Prediction on train data
        y_lr3_valid = lr3.predict(x_valid)  # Prediction on validation data
        y_lr3_full = lr3.predict(x_full)    # Prediction on full data

        # 4. TD + Crucial moments loss function - Linear Regression (ridge) ##############
        lr4 = Linear_Regression_TD()
        lr4.fit(filtered_data, filtered_labels, 0.5, 10)  # Fitting; fit(X, Y, alpha, lambda)

        y_lr4_train = lr4.predict(x_train)  # Prediction on train data
        y_lr4_valid = lr4.predict(x_valid)  # Prediction on validation data
        y_lr4_full = lr4.predict(x_full)    # Prediction on full data

        # 5. Directed MSE





    def run_many(self):
        # run_lr_simulation()을 data sample만큼 반복수행.
        # 나중에 코드 추가하자.
        # 여기서 sample 수 만큼 실행하고, 평균을 낸 후 파일로 저장하는 것 까지는 다뤄주면 좋음.
        print("test")





runCPU = RunCPU('config1.ini')
# test
runCPU.run_lr_simulation(0)



