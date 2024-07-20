import datetime
import sqlite3
import os

def CreateTableAndDatabase():
    unix = datetime.datetime.now().strftime("%s")
    conn = sqlite3.connect(f'./database/stock-{unix}.db')
    cur = conn.cursor()
    
    sql = """CREATE TABLE IF NOT EXISTS stock (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        stock_name TEXT,
        price INTEGER,
        d0 DATE,
        d1 DATE,
        d2 DATE,
        d3 DATE,
        d4 DATE,
        d5 DATE,
        d6 DATE,
        d7 DATE)"""
    cur.execute(sql)
    
    conn.commit()
    conn.close()
    
    return unix

def RemoveDatabase(unix):
    os.chdir("./database")
    os.remove(f"stock-{unix}.db")
    os.chdir("../")