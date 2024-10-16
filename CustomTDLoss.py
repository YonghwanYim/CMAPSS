import torch
import torch.nn as nn

class CustomTDLoss(nn.Module):
    def __init__(self, alpha, beta, theta):
        """
        :param alpha: The reflection ratio of the decision loss within the TD loss (float; 0~1).
        :param beta: The value of the last time cycle compared to using the engine for a single time cycle (float).
        :param theta: Threshold (Not a target for learning)
        """
        super(CustomTDLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.theta = theta

    def forward(self, outputs, targets, obs_time, is_last_time_cycle):
        """
        :param outputs: Predictions (tensor size : batch_size + 1; type : float)
        :param targets: True labels (tensor size : batch_size; type : float)
        :param obs_time: It means that 'time_cycles[i+1] - time_cycles[i]'
        :param is_last_time_cycle: Indicates 1 if it's the last time cycle, and 0 otherwise.
                                    (tensor size : batch_size; type : float)
        :return: td loss
        """
        # batch size에 관계 없이 동적으로 slicing 되도록 하는 변수 batch_size 정의.
        batch_size = outputs.size(0) # tensor의 첫 번째 차원인 batch size. 실제 batch size보다 1 큼.

        # Slicing
        outputs_t = outputs[:batch_size - 1] # 입력으로 받은 tensor에서 마지막 값만 빼고 slicing. (y^{t})
        outputs_t_1 = outputs[1:]            # 입력으로 받은 tensor에서 첫번째 값만 빼고 slicing. (y^{t+1})

        """ Loss function 내의 'ObsTime{t+1,t} + max(y^_{t+1} - theta, 0)'는 미분하지 않고 상수로 처리하도록 하는 코드
            이 부분은 Q-learning에서의 target에 해당되므로 미분하지 않음. (RL에서도 상수 취급함)
        """
        with torch.no_grad():  # torch.no_grad로 상수 취급.
            obs_time_plus_max = obs_time + torch.max(outputs_t_1 - self.theta, torch.tensor(0.0))

        # Calculate MSE loss (prediction loss term)
        prediction_loss = torch.mean((targets - outputs_t) ** 2)

        # Calulate decision loss (a.k.a. td-loss)
        decision_loss = torch.mean( (1 - is_last_time_cycle) * ((obs_time_plus_max - outputs_t + self.theta) ** 2) +
                        is_last_time_cycle * ((-(1 / self.beta) - outputs_t + self.theta ) ** 2) )

        td_loss = (1 - self.alpha) * prediction_loss + self.alpha * decision_loss

        return td_loss
