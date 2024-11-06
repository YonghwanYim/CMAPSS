"""
* Author : Yonghwan Yim
* Final update : 2024.10.16
"""
# Custom TD Loss function
from CustomTDLoss import CustomTDLoss

# directory를 자동으로 가져오기 위함.
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

# x_train: (N, 30, 14), y_train: (N,)
# x_train should be permuted to (N, 14, 30) because PyTorch expects (batch_size, in_channels, seq_length)

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
    def __init__(self, model, batch_size=512, epochs=250, is_td_loss=False, alpha = 0, beta = 0, theta = 0, base_dir='./'):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.is_td_loss = is_td_loss
        self.alpha = alpha    # ratio
        self.beta = beta
        self.theta = theta    # threshold
        self.base_dir = base_dir

        self.loss_fn = None # loss initialization

        # Loss function and optimizer
        if self.is_td_loss:
            # create an instance of the TD loss.
            self.loss_fn = CustomTDLoss(self.alpha, self.beta, self.theta)
        else:
            # 아닌 경우 MSE로 학습.
            self.loss_fn = nn.MSELoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # MPS 가능 여부 출력 (for Mac silicon chip)
        print(f"PyTorch version: {torch.__version__}")
        print(f"Is built to support MPS devices: {torch.backends.mps.is_built()}")
        print(f"Is MPS device available: {torch.backends.mps.is_available()}")

    def lr_schedule(self, epoch):
        if epoch < 500:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.001
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.0001

    def train_model(self, x_train, y_train, obs_time, is_last_time_cycle, is_continue_learning=False):
        # Training loop (assuming x_train is N × 30 × 14 and y_train is N)
        self.model.train()
        #self.model.eval() # dropout 영향 test용 코드.

        # MPS로 mac M1 GPU 사용 (mac 전용 코드).
        device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
        print(f"device: {device}")
        self.model.to(device) # torch의 backend를 mps로 설정.

        try:
            x_train = torch.tensor(x_train, device=device, dtype=torch.float32)
            y_train = torch.tensor(y_train, device=device, dtype=torch.float32)
            obs_time = torch.tensor(obs_time, device=device, dtype=torch.float32)
            is_last_time_cycle = torch.tensor(is_last_time_cycle, device=device, dtype=torch.float32)
        except Exception as e:
            print("Error converting to tensor:", e)

        print('x_train')
        print(x_train)
        print('y_train')
        print(y_train)

        if is_continue_learning: # 이미 학습된 weight을 이어서 학습시킬 때 사용하는 코드 (MSE Loss로 학습시킨 모델 이어서 할 때)
            self.load_model(filename='dcnn_model.pth')

        for epoch in range(self.epochs):
            # Adjust learning rate
            self.lr_schedule(epoch)

            for i in range(0, len(x_train), self.batch_size):
                # Get the batch data
                x_batch = None  # 블록 바깥에서 미리 정의.
                """ 
                 TD loss 적용시. t+1의 prediction까지 알아야 하니 forward 패스에서만 batch보다 1 크게 넣어줌.
                 이렇게 하지 않고, [i+1 : i+1+self.batch_size]로 넣어서 \hat{y}_{t+1}를 구하면,
                 training 중에는 drop out으로 인해 다른 결과가 출력됨. 즉, 중복되는 index인
                 [i+1 : i+self.batch_size]까지의 outputs (Prediction)이 같지 않아짐.
                 """
                if self.is_td_loss : # td loss 사용시 x_batch 생성하는 코드
                    if i + self.batch_size < len(x_train):
                        # 마지막 batch가 아닐 때, 원래 batch보다 index를 1 크게 가져옴 (t+1 예측을 위해)
                        x_batch = x_train[i:i + self.batch_size + 1]
                    else:
                        # 마지막 배치: 크기를 맞추기 위해 마지막 값을 복사 (loss 계산시 마지막 값은 사용하지 않음.)
                        # Indicator function으로 마지막 값 (엔진의 끝)은 0으로 처리하도록 CustomTDLoss에서 정의.
                        x_batch = x_train[i:]  # 남은 데이터를 가져옴.
                        last_row = x_batch[-1].unsqueeze(0) # 마지막 값을 가져옴.
                        x_batch = torch.cat([x_batch, last_row], dim=0) # 맨 끝에 마지막 값을 복사.

                else:
                    # 일반적인 loss function (loss에 y^{t+1}이 사용되지 않는) 사용시.
                    x_batch = x_train[i:i + self.batch_size]  # 일반적인 loss function 사용 시.

                y_batch = y_train[i:i + self.batch_size] # true label은 t+1이 필요 없음.
                obs_time_batch = obs_time[i:i + self.batch_size]
                is_last_time_cycle_batch = is_last_time_cycle[i:i + self.batch_size]

                # input data를 model에 전달하기 전에 GPU (or CPU)로 이동. #################
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                obs_time_batch = obs_time_batch.to(device)
                is_last_time_cycle_batch = is_last_time_cycle_batch.to(device)

                """ 이 부분이 있어야 y^{t+1}을 loss 내에서 계산 가능. """

                # test (batch shape 보는 코드)
                #print(f"x_batch shape: {x_batch.shape}")
                #print(f"y_batch shape: {y_batch.shape}")
                #print("x_batch")
                #print(x_batch)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                try:
                    # Forward pass
                    outputs = self.model(x_batch.permute(0, 2, 1)) # Ensure correct shape for Conv1D

                except Exception as e:
                    print(f"Error during forward pass: {e}")

                #print('outputs.squeeze()')
                #print(outputs.squeeze())

                # squeeze로 output의 차원을 줄임. loss 계산시 차원이 일치하도록 함. 인자로 ObsTime, last_TC가 들어감.
                if self.is_td_loss:
                    loss = self.loss_fn(outputs.squeeze(), y_batch, obs_time_batch, is_last_time_cycle_batch)
                else:
                    loss = self.loss_fn(outputs.squeeze(), y_batch)

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

        return predictions.flatten()






