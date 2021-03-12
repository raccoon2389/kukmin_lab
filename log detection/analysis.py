import pandas as pd
import numpy as np
from pandas.io.pytables import IndexCol


df = pd.read_csv('log detection/dataset/req2log2.csv',index_col=0)
df2 = pd.read_csv("log detection/dataset/req2logANOMAL2.csv",index_col=0)

df3 = pd.concat((df,df2),ignore_index=True)
print(df3)

df4 = df3["Path:"]+df3["Arg:"].map(str) 
df4.to_csv('for_pnn.csv')
