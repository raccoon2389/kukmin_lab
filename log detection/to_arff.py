import numpy as np
import pandas as pd
import arff

x = np.load('log detection/dataset/req2logANOMAL2.npy')
label = np.zeros((x.shape[0],1))
x = np.concatenate([x,label],axis=1)

y = np.load('log detection/dataset/req2log2.npy')
label = np.zeros((y.shape[0],1))+1
y = np.concatenate([y,label],axis=1)

x = np.concatenate([x,y],axis=0)

arff.dump('normal.arff',x,names=["Len_req", "Len_arg", "Num_arg", "Len_path"," Num_sp_char_path","num_digit_arg", "num_letter_arg", "num_letter_char_arg","Maximum_bytes","Label"])