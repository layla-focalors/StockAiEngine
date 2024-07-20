from modules import system
from modules import version
from modules import db
from modules import croller
from modules import train

import datetime

def getSystem():
    print(f"Running analysis : Running Time {datetime.datetime.now()}")
    print(f"System Information : {system.getSystemInfomation()}")
    print(f"Version Information : {version.getProcessVersion()}")
    return None

def ExcuteTask(data):
    output = """\n분석에 활용할 거래소, 종목을 입력하십시오.\n주의 이 프로세스는 개발 버전으로 시스템의 최종 퀄리티가 아닙니다. \n이에 예측 실패, 오류 및 각종 시스템 에러가 발생할 수 있습니다.\n입력 방법 : (거래소):(종목)\n입력 예시 : nasdaq:aapl / kospi:005930"""
    
    if version.getProcessVersion()[0:3] == "Dev":
        # dev
        print(output)
        MultiInference("Dev")
    else:
        # release
        while True:
            out = MultiInference()
            if out == "stop":
                break
    
def MultiInference(env):
    if env == "Dev":
        user_input = "nasdaq:aapl"
    else:
        user_input = input("분석에 활용할 거래소, 종목을 입력하십시오. (종료 : exit) : ")
        
        if user_input == "exit":
            return "stop"
        elif ":" not in user_input:
            return None
    
    dt = user_input.split(":")
    
    trade = dt[0]
    stock = dt[1]
    
    print(f"거래소 : {trade}, 종목 : {stock}")
    
    # db 생성, databaseID 반환
    databaseID = db.CreateTableAndDatabase()
    croller.RunCroller(trade, stock, databaseID)
    
    # 오래된 DB 정리
    system.DeleteDatabaseRequire(databaseID)
    
    # 학습 실행
    train.RunTrain(databaseID, stock)
    
    