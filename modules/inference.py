import os
from modules import train
from modules import db
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

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
    
    return model

def SeparateData(df):
    try:
        # 데이터 분할 / 학습 데이터셋 & 테스트 데이터셋 7:3 비율
        
        train_size = int(len(df) * 0.7)
        train_set = df[0:train_size]
        test_set = df[train_size:]
        
        # 데이터 스케일링 후 변경 
    except:
        print("Data Separate Failed EP2999")
    return train_set, test_set

def InferenceModel(model, databaseID):
    data = pd.DataFrame(db.getDatabaseInfo(databaseID))
    data = data.drop([0, 1, 2], axis=1)
    # data = data.iloc[-1]

    print("Excution Data : ", data)
    data = data[::-1]
    
    seq_len = 60
    batch = 100
    
    train_set, test_set = SeparateData(data)
    
    data_loader = train.TranslationTensorType(train_set, test_set, seq_len, batch)
    
    print(train_set)
    try:
        for i, (batch_sequence, batch_target) in enumerate(data_loader):
            model.reset_hidden_state(batch_target.size(0))
            pred = model(batch_sequence)
            # print(pred[-1]) 
            break 
    except:
        print("Inference Failed EP3200")
    return pred[-1]