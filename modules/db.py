import datetime
import sqlite3

def CreateTableAndDatabase():
    unix = str(datetime.datetime.now().timestamp()).split(".")[0]
    conn = sqlite3.connect(f"./database/stock-{unix}.db")
    cur = conn.cursor()
    
    sql = """CREATE TABLE IF NOT EXISTS stock (id INTEGER PRIMARY KEY AUTOINCREMENT, 
    date DATE,
    ticker TEXT,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL)"""
    
    cur.execute(sql)
    
    conn.commit()
    conn.close()
    
    return unix
        
def InitCrollerData(databaseID, DATA):
    conn = sqlite3.connect(f"./database/stock-{databaseID}.db")
    cur = conn.cursor()
    
    for i in range(len(DATA)):
        date = DATA["index"][i]
        ticker = DATA["ticker"][i]
        open = DATA["open"][i]
        high = DATA["high"][i]
        low = DATA["low"][i]
        close = DATA["close"][i]
        volume = DATA["volume"][i]
        
        sql = f"""INSERT INTO stock (date, ticker, open, high, low, close, volume) VALUES ('{date}', '{ticker}', {open}, {high}, {low}, {close}, {volume})"""
        cur.execute(sql)
    
    conn.commit()
    conn.close()
    
    print("Data Insert Complete")