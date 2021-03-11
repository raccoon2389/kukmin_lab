import pandas as pd
import numpy as np
from glob import glob


path2pkl = glob('kisti_http/data/*.pkl')
# print(path2pkl)
l=[]
# for idx,path in enumerate(path2pkl):
#     path = path.split("\\")[-1]
#     df = pd.DataFrame(pd.read_pickle(f"kisti_http/data/{path}"),index=None)
#     pay = df.loc[df["decodePayload"].str.contains('GET' or 'POST',na=False)]
    
#     pay.to_pickle(f"kisti_http\\data\\Requests\\{path}")
#     l.append(pay)


for idx,path in enumerate(path2pkl):
    path = path.split("\\")[-1]
    d = pd.read_pickle(f"kisti_http/data/Requests/{path}")
    print(len(d))
    l.append(d)

'''
df = pd.concat(l,sort=False)
df.to_pickle(f"kisti_http\\data\\Requests\\stacked.pkl")
print(df["decodePayload"])
df.loc[:,"decodePayload"].to_csv('kisti_http/data/Requests/req.csv')
'''