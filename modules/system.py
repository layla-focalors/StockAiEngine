import platform
import os

def getSystemInfomation():
    return platform.uname().system

def MakeEnvironment():
    os.system("pip install yahoo-fin")
    
def DeleteOldModels():
    file_list = os.listdir('./model')
    model_list = []
    
    for file in file_list:
        if file.endswith('.pth'):
            model_list.append(file)
    
    if len(model_list) >= 10:
        model_list.sort(key=lambda x: int(''.join(filter(str.isdigit, x))), reverse=True)
        for i in range(1, len(model_list)):
            os.remove(f"./models/{model_list[i]}")
        print("오래된 모델의 개수가 10개를 초과하여 자동 삭제되었습니다.")
    return None
    
def DeleteImageRequire():
    file_list = os.listdir('./output')
    image_list = []
    
    for file in file_list:
        if file.endswith('.jpg') or file.endswith('.png'):
            image_list.append(file)
    
    if len(image_list) >= 10:
        image_list.sort(key=lambda x: int(''.join(filter(str.isdigit, x))), reverse=True)
        for i in range(1, len(image_list)):
            os.remove(f"./output/{image_list[i]}")
        print("오래된 이미지의 개수가 10개를 초과하여 자동 삭제되었습니다.")
    
def DeleteDatabaseRequire(databaseID):
    file_list = os.listdir('./database')
    new_file_list = []
    
    for file in file_list:
        if not str(databaseID) in file:
            new_file_list.append(file)
            
    if len(new_file_list) >= 10:
        for i in range(0, len(new_file_list)):
            os.remove(f"./database/{new_file_list[i]}")  
        print("오래된 데이터베이스의 개수가 10개를 초과하여 자동 삭제되었습니다.")
    return None  
