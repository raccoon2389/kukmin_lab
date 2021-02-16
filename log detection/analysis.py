import pandas as pd
import numpy as np

logs = pd.DataFrame()

with open("log detection/dataset/logs.log","r",encoding="utf-8") as f:
    lines = f.readlines()
    for l in lines:
        l = str(l)
        l = l.replace('"','').split(" ")
        # l = list(l)
        
        print(l)
        log = pd.Series(l,dtype="str")
        # print(log.shape)
        logs.append(log,ignore_index=True)
    print(logs.head(10))
logs.to_csv('log detection/dataset/logs.csv')


# s = pd.read_csv('log detection/dataset/logs.csv',sep=' ',dtype=str,engine='python',encoding='utf-8')

# print(s.head(5))