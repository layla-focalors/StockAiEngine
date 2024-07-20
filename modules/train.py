import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

from modules import db

def RunTrain(databaseID):
    data = pd.DataFrame(db.getDatabaseInfo(databaseID))
    data = data.drop([0, 1, 2], axis=1)
    # print(data)
    data = data[::-1]
    
    # data days for using train
    seq_len = 60
    batch = 100
    
    train_set, test_set = SeparateData(data)
    
    dataloader = TranslationTensorType(train_set, test_set, seq_len, batch)
    # print(data)
    return None

def SeparateData(df):
    # 데이터 분할 / 학습 데이터셋 & 테스트 데이터셋 7:3 비율
    
    train_size = int(len(df) * 0.7)
    train_set = df[0:train_size]
    test_set = df[train_size:]
    
    # 데이터 스캐일링 후 변경 
    train_set, test_set = ScaleData(train_set, test_set)
    
    return train_set, test_set
    
    
def ScaleData(train_set, test_set):
    # 입력
    scaler_x = MinMaxScaler()
    scaler_x.fit(train_set.iloc[:, :-1])
    
    train_set.iloc[:, :-1] = scaler_x.transform(train_set.iloc[:, :-1])
    test_set.iloc[:, :-1] = scaler_x.transform(test_set.iloc[:, :-1])
    
    # 출력
    scaler_y = MinMaxScaler()
    scaler_y.fit(train_set.iloc[:, [-1]])

    train_set.iloc[:, -1] = scaler_y.transform(train_set.iloc[:, [-1]])
    test_set.iloc[:, -1] = scaler_y.transform(test_set.iloc[:, [-1]])
    
    return train_set, test_set

def TranslationTensorType(train_set, test_set, seq_len, batch):
    def build_dataset(time_series, seq_length):
        dataX = []
        dataY = []
        for i in range(0, len(time_series) - seq_length):
            _x = time_series[i:i + seq_length, :]
            _y = time_series[i + seq_length, [-1]]
            dataX.append(_x)
            dataY.append(_y)
        return np.array(dataX), np.array(dataY)
    
    trainX, trainY = build_dataset(np.array(train_set), seq_len)
    testX, testY = build_dataset(np.array(test_set), seq_len)
    
    trainX_tensor = torch.FloatTensor(trainX)
    trainY_tensor = torch.FloatTensor(trainY)
    
    testX_tensor = torch.FloatTensor(testX)
    testY_tensor = torch.FloatTensor(testY)
    
    dataset = TensorDataset(trainX_tensor, trainY_tensor)
    
    data_loader = DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=True)
    return data_loader