import tensorflow as tf
import numpy as np
"""
This file is not used at the moment, so it can be ignored.
Version of the TD Loss function without the max operator.
"""

# simulation.py 에서 사용할 때, data sample의 수 만큼 처음에 초기화해두고 가져다 쓰도록 구현.
class DecisionAwareTD:
    def __init__(self, dataset, beta, weight, alpha):
        self.dataset = dataset
        self.beta = beta
        self.weight = weight
        self.alpha = alpha  # (prediction - decision-aware) ratio
        self.X_t_minus_1 = dataset
        self.delete_row1_dataset = dataset
        self.features = [f's_{i}' for i in range(0, 22)]

    def preprocessing(self):
        self.add_TD_column_to_dataset()  # TD column 추가 (마지막 행의 - 1/beta 도 포함)
        self.delete_first_row_of_engines()
        self.save_X_t_minus_1()

    def add_TD_column_to_dataset(self):
        # time-step의 차를 저장. 첫번째 행은 0으로 저장됨.
        self.dataset['TD'] = self.dataset['time_cycles'].diff().fillna(0)

        # 각 unit_number의 첫 번째 행은 TD를 0으로 설정 (서로 다른 엔진 간 TD를 계산하지 않도록 하기 위함)
        self.dataset.loc[self.dataset.groupby('unit_number').head(1).index, 'TD'] = 0

        # 각 unit_number의 마지막 행에만 TD 대신 '-1/베타' 로 바꿈. 이렇게 해야 하나의 행렬로 gradient를 계산할 수 있음.
        self.dataset.loc[self.dataset.groupby('unit_number').tail(1).index, 'TD'] = -(1 / self.beta)

    def delete_first_row_of_engines(self):
        self.delete_row1_dataset = self.dataset[self.dataset.groupby('unit_number').cumcount() != 0]
        self.delete_row1_dataset.reset_index(drop=True, inplace=True)  # index reset

        # s_1 열 왼쪽에 s_0 열을 새롭게 추가하고 모든 값을 1로 초기화
        self.delete_row1_dataset.insert(loc=self.delete_row1_dataset.columns.get_loc('s_1'), column='s_0', value=1)

    def save_X_t_minus_1(self):
        # 각 unit_number의 마지막 행을 제외한 새로운 데이터 프레임.
        temp_dataset = self.dataset.drop(self.dataset.groupby('unit_number').tail(1).index)
        temp_dataset.insert(loc=temp_dataset.columns.get_loc('s_1'), column='s_0', value=1)

        # 필요한 column만 선택 (s_0부터 s_21까지)
        self.X_t_minus_1 = temp_dataset[self.features]
        self.X_t_minus_1.reset_index(drop=True, inplace=True)  # index reset

    def calculate_gradient(self):
        self.X_t = self.delete_row1_dataset[self.features]
        self.Y = self.delete_row1_dataset['RUL']
        self.Y_X = self.Y.T.dot(self.X_t) # 이 값이 압도적으로 큼.
        self.TD = self.delete_row1_dataset['TD']
        self.row_size = self.delete_row1_dataset.shape[0] # row의 수를 저장해, gradient를 구할 때 나눠주기 위함.
        #print(self.Y.T)
        #print(self.Y_X)

        self.W_X_X = self.weight.T.dot(self.X_t.T.dot(self.X_t))
        #print(self.weight.T)
        #print(self.W_X_X)

        self.X_t_diff_X_t_minus_1 = self.X_t - self.X_t_minus_1
        #print(self.X_t_diff_X_t_minus_1)

        self.W_X_diff_square = self.weight.T.dot(self.X_t_diff_X_t_minus_1.T.dot(self.X_t_diff_X_t_minus_1))
        #print(self.W_X_diff_square)

        gradient = -2 * self.Y_X + 2 * self.W_X_X + 2 * self.alpha * (self.W_X_diff_square + self.TD.T.dot(self.X_t_diff_X_t_minus_1))
        mean_gradient = gradient / self.row_size

        return mean_gradient

    def calculate_gradient_only_TD(self):
        self.X_t = self.delete_row1_dataset[self.features]
        self.Y = self.delete_row1_dataset['RUL']
        self.Y_X = self.Y.T.dot(self.X_t) # 이 값이 압도적으로 큼.
        self.TD = self.delete_row1_dataset['TD']
        self.row_size = self.delete_row1_dataset.shape[0] # row의 수를 저장해, gradient를 구할 때 나눠주기 위함.
        #print(self.Y.T)
        #print(self.Y_X)

        self.W_X_X = self.weight.T.dot(self.X_t.T.dot(self.X_t))
        #print(self.weight.T)
        #print(self.W_X_X)

        self.X_t_diff_X_t_minus_1 = self.X_t - self.X_t_minus_1
        #print(self.X_t_diff_X_t_minus_1)

        self.W_X_diff_square = self.weight.T.dot(self.X_t_diff_X_t_minus_1.T.dot(self.X_t_diff_X_t_minus_1))
        #print(self.W_X_diff_square)

        gradient = 2 * self.alpha * (self.W_X_diff_square + self.TD.T.dot(self.X_t_diff_X_t_minus_1))
        #gradient = - 2 * self.alpha * (self.weight.T.dot(self.X_t_diff_X_t_minus_1.T) + self.TD.T).dot(self.X_t)
        mean_gradient = gradient / self.row_size

        return mean_gradient

    def calculate_closed_form_solution(self, lambd):
        # Directly derive the closed-form solution.
        self.X_t = self.delete_row1_dataset[self.features]
        self.Y = self.delete_row1_dataset['RUL']
        self.Y_X = self.Y.T.dot(self.X_t) # 이 값이 압도적으로 큼.
        self.TD = self.delete_row1_dataset['TD']
        self.row_size, self.col_size = self.X_t.shape
        self.lambda_ridge = lambd


        self.W_X_X = self.weight.T.dot(self.X_t.T.dot(self.X_t))
        self.X_t_diff_X_t_minus_1 = self.X_t - self.X_t_minus_1
        self.W_X_diff_square = self.weight.T.dot(self.X_t_diff_X_t_minus_1.T.dot(self.X_t_diff_X_t_minus_1))

        self.identity = np.identity(n=self.col_size)  # Ridge regression
        self.lambda_identity = self.lambda_ridge * self.identity  # lambda * I

        second_term = self.X_t.T.dot(self.X_t) + self.alpha * self.X_t_diff_X_t_minus_1.T.dot(self.X_t_diff_X_t_minus_1)

        # compute the (Moore-Penrose) pseudo-inverse of a matrix.
        inverse_second_term = np.linalg.inv(second_term + self.lambda_identity)

        solution = (self.Y_X - self.alpha * self.TD.T.dot(self.X_t_diff_X_t_minus_1)).dot(inverse_second_term)

        return solution

    def calculate_closed_form_solution_ratio(self, lambd):
        # prediction loss term에도 (1-alpha)를 곱해줘서, decision, prediction loss의 총 합이 1이 되도록 바꿈.
        self.X_t = self.delete_row1_dataset[self.features]
        self.Y = self.delete_row1_dataset['RUL']
        self.Y_X = self.Y.T.dot(self.X_t) # 이 값이 압도적으로 큼.
        self.TD = self.delete_row1_dataset['TD']
        self.row_size, self.col_size = self.X_t.shape
        self.lambda_ridge = lambd

        self.W_X_X = self.weight.T.dot(self.X_t.T.dot(self.X_t))
        self.X_t_diff_X_t_minus_1 = self.X_t - self.X_t_minus_1
        self.W_X_diff_square = self.weight.T.dot(self.X_t_diff_X_t_minus_1.T.dot(self.X_t_diff_X_t_minus_1))

        self.identity = np.identity(n=self.col_size)  # Ridge regression
        self.lambda_identity = self.lambda_ridge * self.identity  # lambda * I

        second_term = (1 - self.alpha) * self.X_t.T.dot(self.X_t) + self.alpha * self.X_t_diff_X_t_minus_1.T.dot(self.X_t_diff_X_t_minus_1)

        # compute the (Moore-Penrose) pseudo-inverse of a matrix.
        inverse_second_term = np.linalg.pinv(second_term + self.lambda_identity)

        solution = ((1 - self.alpha) * self.Y_X - self.alpha * self.TD.T.dot(self.X_t_diff_X_t_minus_1)).dot(inverse_second_term)

        return solution
