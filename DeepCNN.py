"""
* Author : Yonghwan Yim
* DCNN Version 1
* Final update : 2024.09.26
"""
# Custom TD Loss function
from CustomTDLoss import CustomTDLoss

# directory를 자동으로 가져오기 위함.
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

# Example of calling the training function
# Assuming x_train and y_train are PyTorch tensors with proper dimensions
# x_train: (N, 30, 14), y_train: (N,)
# x_train should be permuted to (N, 14, 30) because PyTorch expects (batch_size, in_channels, seq_length)
# x_train = x_train.permute(0, 2, 1)
# train_model(model, criterion, optimizer, x_train, y_train, epochs)

# 모델 정의는 독립적으로 클래스화
# 논문에서 다룬것과 동일. piecewise linear 셋팅만 제외하고.
class DCNN(nn.Module):
    def __init__(self, N_tw=30, N_ft=14, FN=10, FL=10, neurons_fc=100, dropout_rate=0.5):
        super(DCNN, self).__init__() # DCNN class는 PyTorch의 nn.Module 상속받음. 따라서 parent class를 initialize
        self.conv1 = nn.Conv1d(in_channels=N_ft, out_channels=FN, kernel_size=FL, padding='same')
        self.conv2 = nn.Conv1d(in_channels=FN, out_channels=FN, kernel_size=FL, padding='same')
        self.conv3 = nn.Conv1d(in_channels=FN, out_channels=FN, kernel_size=FL, padding='same')
        self.conv4 = nn.Conv1d(in_channels=FN, out_channels=FN, kernel_size=FL, padding='same')
        self.conv_high = nn.Conv1d(in_channels=FN, out_channels=1, kernel_size=3, padding='same')

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(N_tw, neurons_fc)
        self.fc2 = nn.Linear(neurons_fc, 1)

        # Xavier initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        x = torch.tanh(self.conv4(x))
        x = torch.tanh(self.conv_high(x))

        x = self.flatten(x)
        x = self.dropout(x)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


# 훈련 로직은 별도의 클래스로 정의
class DCNN_Model:
    def __init__(self, model, batch_size=512, epochs=250, base_dir='./'):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.base_dir = base_dir

        # Loss function and optimizer
        self.criterion = nn.MSELoss() # 나중에 MSE 대신 TD loss (Custom)으로 바꿔야 함. loss를 class로 정의하는게 편함
        # self.criterion = CustomTDLoss() # create an instance of the TD loss.
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # MPS 가능 여부 출력 (for Mac silicon chip)
        print(f"PyTorch version: {torch.__version__}")
        print(f"Is built to support MPS devices: {torch.backends.mps.is_built()}")
        print(f"Is MPS device available: {torch.backends.mps.is_available()}")

    def lr_schedule(self, epoch):
        if epoch < 200:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.001
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.0001

    def train_model(self, x_train, y_train):
        # Training loop (assuming x_train is N × 30 × 14 and y_train is N)
        self.model.train()

        # MPS로 mac M1 GPU 사용 (mac 전용 코드).
        device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
        print(f"device: {device}")
        self.model.to(device) # torch의 backend를 mps로 설정.

        try:
            x_train = torch.tensor(x_train, device=device, dtype=torch.float32)
            y_train = torch.tensor(y_train, device=device, dtype=torch.float32)
        except Exception as e:
            print("Error converting to tensor:", e)

        print(x_train)
        print(y_train)

        for epoch in range(self.epochs):
            # Adjust learning rate
            self.lr_schedule(epoch)

            for i in range(0, len(x_train), self.batch_size):
                # Get the batch data
                x_batch = x_train[i:i + self.batch_size]
                y_batch = y_train[i:i + self.batch_size]

                # input data를 model에 전달하기 전에 CPU로 이동. #################
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                ###########################################################

                # test (batch shape 보는 콛,)
                #print(f"x_batch shape: {x_batch.shape}")
                #print(f"y_batch shape: {y_batch.shape}")

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                try:
                    # Forward pass
                    outputs = self.model(x_batch.permute(0, 2, 1)) # Ensure correct shape for Conv1D
                except Exception as e:
                    print(f"Error during forward pass: {e}")

                # Forward pass
                #outputs = self.model(x_batch.permute(0, 2, 1))  # Ensure correct shape for Conv1D
                #outputs = self.model(x_batch) # 에러가 난 코드. Pytorch에서 사용할 수 있는 데이터 타입으로 변환.
                loss = self.criterion(outputs.squeeze(), y_batch) # squeeze로 output의 차원을 줄임. loss 계산시 차원이 일치하도록.

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{self.epochs}], Loss: {loss.item():.4f}')

    def save_model(self, filename='dcnn_model.pth'):
        """모델의 state_dict를 현재 파일의 디렉토리에 저장"""
        save_path = os.path.join(self.base_dir, filename)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    # 모델 로드 기능 수정
    def load_model(self, filename='dcnn_model.pth'):
        """현재 파일의 디렉토리에서 모델의 state_dict를 로드"""
        load_path = os.path.join(self.base_dir, filename)
        self.model.load_state_dict(torch.load(load_path))
        print(f"Model loaded from {load_path}")

    def predict(self, x_test):
        self.model.eval() # model을 evaluation mode로 변환. (dropout, bathnorm 등 비활성화)

        # x_test가 numpy.ndarray일 경우, torch.Tensor로 변환
        if isinstance(x_test, np.ndarray):
            x_test = torch.from_numpy(x_test)

        # MPS로 mac M1 GPU 사용 (mac 전용 코드).
        device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
        print(f"device: {device}")
        self.model.to(device) # torch의 backend를 mps로 설정.

        try:
            x_test = torch.tensor(x_test, device=device, dtype=torch.float32)
        except Exception as e:
            print("Error converting to tensor:", e)

        # 배치 차원이 없으면 차원을 추가 (batch_size = 1로 변환); 이 과정 꼭 필요. input shape 때문에 에러 발생.
        if len(x_test.shape) == 2:  # (sequence_length, channels)의 형태라면
            x_test = x_test.unsqueeze(0)  # (1, sequence_length, channels) -> batch 차원 추가

        # 입력 데이터를 (batch_size, channels, sequence_length) 형태로 변환
        # (batch_size, sequence_length, channels) -> (batch_size, channels, sequence_length)
        x_test = x_test.permute(0, 2, 1)

        # Prediction
        with torch.no_grad():
            #predictions = self.model(x_test)
            predictions = self.model(x_test).cpu().numpy()  # 예측 후 numpy 배열로 변환 (MPS -> CPU로 바꿔야 numpy로 변환 가능)

        # predictions를 tensor에서 list로 변환. 원래 데이터에 predictions column을 추가하기 위해.
        #predictions = predictions.cpu().numpy().tolist()

        return predictions.flatten()






