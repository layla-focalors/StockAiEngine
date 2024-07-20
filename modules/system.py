import platform
import os

def getSystemInfomation():
    return platform.uname().system

def MakeEnvironment():
    os.system("pip install yahoo-fin")
    
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