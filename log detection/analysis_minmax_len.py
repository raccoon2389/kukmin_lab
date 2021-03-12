from math import isnan
import numpy as np
import pandas as pd
import urllib.parse as parse
import re
F_PATH = ['req2log2','req2logANOMAL2']

all = re.compile("[^a-zA-Z0-9_]")
dot = re.compile('[.]')
dig = re.compile("[0-9]")
char = re.compile("[a-zA-Z]")
letter = re.compile("[a-zA-Z0-9_]")

req_n = pd.read_csv(f'log detection/dataset/req2log2.csv',index_col=0)
req_a = pd.read_csv(f'log detection/dataset/req2logANOMAL2.csv',index_col=0)

req = pd.concat((req_n,req_a),ignore_index=True)
line=""
header =""

for i in range(3): line += req.iloc[:,i].map(str)

for i in range(3,14):header += req.iloc[:,i].map(str)

arg = req.loc[:,"Arg:"]
measure = np.vectorize(len)
df = line.map(str) + arg.map(str)
# df.to_csv('pnn/data/req.csv')
# quit()
l = []
sets = [line,header,arg]
words = ['Line','Header','Argument']

for s,w in zip(sets,words):
    m = measure(s.values.astype(str))
    l1 = m.min(axis=0)
    l2 = m.max(axis=0)
    print(f"{w} min / max:\t\t{l1} / {l2}")

# Line      min / max:  33 / 897
# Header    min / max:  323 / 323
# Argument  min / max:  3 / 836