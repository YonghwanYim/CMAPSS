import torch.nn as nn

class CustomTDLoss(nn.Module):
    def __init__(self):
        super(CustomTDLoss, self).__init__()

    def forward(self, inputs, targets):
        # 여기에 unit_number 내에서 마지막 타임스텝인지 여부를 판단할 수 있는 0, 1 값이 들어와야 함. 좀 더 고민
        td_loss = inputs + targets


        return td_loss.mean()






# how to use this loss function.

# td_loss = CustomTDLoss()
# 학습하는 모델 내에서는 아래처럼 불러와서 쓰면됨.
# loss = td_loss(outputs, targets)

