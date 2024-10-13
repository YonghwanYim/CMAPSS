import torch
import torch.nn as nn

class CustomTDLoss(nn.Module):
    def __init__(self):
        super(CustomTDLoss, self).__init__()
        # 여기서 DCNN 클래스 내의 값으로 어떻게 접근할까?
        # outputs_t_1 등등. 함수의 인자로 받아올 수는 없는데,
        # method를 따로 정의해서 매번 넣어줘야 하나? (아마도 그렇게 하는게..)
        # 일단 forward의 인자는 건드리면 안됨.

    def forward(self, outputs, targets):
        # 여기에 unit_number 내에서 마지막 타임스텝인지 여부를 판단할 수 있는 0, 1 값이 들어와야 함. 좀 더 고민
        # 여기서 데이터셋의 index를 판단할 수 잇어야..
        # ObsTime{t+1, t}와 max(y^_{i,t+1}, 0)을 어떻게 가져오지?..
        obs_max = 1

        # Loss function 내의 ObsTime + max()는 미분하지 않고 상수로 처리하도록 하는 코드.
        with torch.no_grad():  # torch.no_grad로 구현할 수 있음.
            constant_part = torch.mean(targets) # 이 코드는 예시일 뿐 수정해야 함.

        prediction_loss = torch.mean((outputs - targets) ** 2) # MSE loss



        td_loss = prediction_loss + constant_part


        return td_loss






# how to use this loss function.

# td_loss = CustomTDLoss()
# 학습하는 모델 내에서는 아래처럼 불러와서 쓰면됨.
# loss = td_loss(outputs, targets)

