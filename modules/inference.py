import os
from modules import train
import torch
from torch.utils.data import DataLoader, TensorDataset

def LoadModelFromHive(model_name):
    print("불러올 모델 이름 : ", model_name)
    
    model = train.TrainModel_LSTM().LSTM()
    model.load_state_dict(torch.load(model_name))
    model.eval()
    print(model.eval())