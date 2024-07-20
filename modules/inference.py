import os
from modules import train
import torch
from torch.utils.data import DataLoader, TensorDataset

input_size = 5
hidden_size = 100
output_size = 1
learning_rate = 0.01

# LSTM 모델 생성
class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, 1)

    def reset_hidden_state(self, batch_size):
        self.hidden = (
            torch.zeros(1, batch_size, self.hidden_size),
            torch.zeros(1, batch_size, self.hidden_size)
        )

    def forward(self, x):
        x, _status = self.lstm(x)
        x = self.linear(x[:, -1])
        return x
            
def LoadModelFromHive(model_name):
    print("불러올 모델 이름 : ", model_name)
    
    model = LSTM(input_size, hidden_size)
    model.load_state_dict(torch.load(model_name))
    model.eval()
    print(model.eval())