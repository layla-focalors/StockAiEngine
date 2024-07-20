import platform
import os

def getSystemInfomation():
    return platform.uname().system

def MakeEnvironment():
    os.system("pip install yahoo-fin")