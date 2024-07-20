import os
import yahoo_fin.stock_info as si
import pandas as pd
from modules import db

def RunCroller(trade, stock, databaseID):
    print("Now DownLoad Stock Data With SearchMod . . .")
    
    DATA = si.get_data(f'{stock}', start_date='2021-11-15', interval="1d").reset_index()
    
    # db 데이터 초기화
    db.InitCrollerData(databaseID, DATA)
    
    