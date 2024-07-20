import pandas as pd
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

from modules import db

def RunTrain(databaseID, stockID):
    data = pd.DataFrame(db.getDatabaseInfo(databaseID))
    data = data.drop([0, 1, 2], axis=1)
    # print(data)
    data = data[::-1]
    
    # data days for using train
    seq_len = 60
    batch = 100
    
    train_set, test_set = SeparateData(data)
    
    dataloader = TranslationTensorType(train_set, test_set, seq_len, batch)
    
    model, model_name = TrainModel_LSTM(dataloader, seq_len, 100, stockID=stockID)
    print(model)
    # print(data)
    return model, model_name

def SeparateData(df):
    try:
        # 데이터 분할 / 학습 데이터셋 & 테스트 데이터셋 7:3 비율
        
        train_size = int(len(df) * 0.7)
        train_set = df[0:train_size]
        test_set = df[train_size:]
        
        # 데이터 스캐일링 후 변경 
        train_set, test_set = ScaleData(train_set, test_set)
    except:
        print("Data Separate Failed EP2999")
    return train_set, test_set
    
    
def ScaleData(train_set, test_set):
    try:
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
    except:
        print("Data Scale Failed EP3000")
            
    return train_set, test_set

def TranslationTensorType(train_set, test_set, seq_len, batch):
    try:
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
    except:
        print("Data Translation Failed EP3100")
    return data_loader

def TrainModel_LSTM(data_loader, seq_len, epochs, stockID):
    
    try:
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
        
        # trainNode1
        model = LSTM(input_size, hidden_size)
        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_hist = np.zeros(epochs)
        
        for epoch in range(epochs):
            model.reset_hidden_state(batch_size=data_loader.batch_size) 
            for i, data in enumerate(data_loader):
                x, y = data
                optimizer.zero_grad()
                output = model(x)
                loss = loss_function(output, y)
                loss.backward()
                optimizer.step()
                
                if i % 100 == 0:
                    print("Epoch: %d, Batch: %d, Loss: %1.5f, ModelINF: %s" % (epoch, i, loss.item(), stockID))
                    
            train_hist[epoch] = loss.item()
    except:
        print("Model Train Failed EP3110")
    # 모델 저장 & 기록
    try:
        ModelLossDraw(train_hist, stockID)
    except:
        print("Model Loss Draw Failed EP3111")
        
    try:
        model_name = SaveModelPth(model, stockID)
    except:
        print("Model Save Failed EP3112")
            
    return model, model_name
    
def ModelLossDraw(train_hist, stockID):
    plt.figure(figsize=(12, 6))
    plt.plot(train_hist, label="Training loss")
    plt.title("Loss at each epoch")
    plt.legend()
    plt.savefig(f"./output/{int(datetime.datetime.now().timestamp())}_train_loss-{stockID}.png")
    
    return None

def SaveModelPth(model, stockID):
    model_name = f"./model/{int(datetime.datetime.now().timestamp())}_model-{stockID}.pth"
    torch.save(model.state_dict(), model_name)
    
    return model_name

def PredictTestDataset(model, testX_tensor, scaler_y, stockID):
    model.eval()
    test_predict = model(testX_tensor)
    test_predict = scaler_y.inverse_transform(test_predict.detach().numpy())
    
    DrawTestDatasetOut(test_predict, stockID)
    
    return test_predict

def DrawTestDatasetOut(test_predict, stockID):
    fig = plt.figure(facecolor='white', figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.plot(test_predict, label='Predict')
    ax.legend()
    plt.show()
    plt.savefig(f"./output/{int(datetime.datetime.now().timestamp())}_predict-{stockID}.png")