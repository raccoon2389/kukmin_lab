import pandas as pd
import numpy as np
from pandas.io.pytables import IndexCol


df = pd.read_csv('log detection/dataset/req2log2.csv',index_col=0)

print(df.iloc[:,1].unique(),df.iloc[:,0].value_counts())