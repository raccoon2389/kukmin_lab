from math import isnan
import numpy as np
import pandas as pd
import urllib.parse as parse
import re
import pickle

F_PATH = ['req2log2','req2logANOMAL2']

path_voca = []

for P in F_PATH:
    req = pd.read_csv(f'log detection/dataset/{P}.csv',index_col=0)

    for idx,row in req.iterrows():
        path = row["Path:"]

        parsed =parse.urlsplit(path)
        # print(parsed.path)
        path = parsed.path.split("/")[1:]
        # print(path)
        # quit()
        for i in path:
            if not i in path_voca:
                path_voca.append(i)

vocab = {tkn: i+2 for i, tkn in enumerate(path_voca)}  # 단어 집합의 각 단어에 고유한 정수 맵핑.
vocab['<unk>'] = 0
vocab['<pad>'] = 1
# print(vocab.items())
with open('log detection/dataset/vocab.pkl',"wb") as f:
    pickle.dump(vocab,f)
f.close



with open('log detection/dataset/vocab.pkl',"rb") as f:
    df = pd.read_pickle(f)
f.close

v_list = df.items()

# print(len(df))
l=[]
max=0
for P in F_PATH:

    req = pd.read_csv(f'log detection/dataset/{P}.csv',index_col=0)
    arr = np.zeros((req.shape[0],6))+1
    for r_idx,row in req.iterrows():
        path = row["Path:"]
        parsed =parse.urlsplit(path)

        path = parsed.path.split("/")[1:]
        
        
        if len(path)>max:max=len(path)
        for c_idx, cont in enumerate(path):
            arr[r_idx,c_idx]=df.get(cont)
    np.save(f"log detection/dataset/voca/{P}.npy",arr)
    # print(arr)
    l.append(arr)
np.save('log detection/dataset/voca/concat.npy',np.concatenate((l[0],l[1]),axis=0))
print(max)