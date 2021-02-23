import pandas as pd
import numpy as np

log = pd.read_csv('log detection/dataset/EClog HTTP_level e_commerce data based on server access logs for an online store/eclog.csv',header=0,index_col=None)
print(log.loc[:,'Uri'].unique().size)