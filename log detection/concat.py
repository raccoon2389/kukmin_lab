import pandas as pd
import numpy as np
import tqdm

##### Load data from numpy ######

train_x = np.load('log detection/dataset/req2log2.npy')
test_x = np.load('log detection/dataset/req2logTEST2.npy')
anomal_x = np.load('log detection/dataset/req2logANOMAL2.npy')

train_y = np.zeros((train_x.shape[0]))
anomal_y = np.zeros((anomal_x.shape[0])) + 1
train_data = np.concatenate([train_x,anomal_x],axis=0)

train_label = np.concatenate([train_y,anomal_y]).reshape(-1,1)

df_x = pd.DataFrame(train_x)
df_y = pd.DataFrame(anomal_x)
df_x.to_csv('norm2.csv')
df_y.to_csv('anomal2.csv')
# print(df_x)
# quit()

l=[]
for i in tqdm.tqdm(df_x.iterrows()):
    # print("".join(str(i)))
    a=""
    for i2 in i[1]:
        # print(i2)
        # print((i[i2]))
        # break
        a += ","+str(i2)
    
    l.append(a)
df1 = pd.DataFrame(l,index=None,columns=None)
print(df1.shape)
df1.to_csv('log detection/dataset/nom.csv')

l=[]
for i in tqdm.tqdm(df_y.iterrows()):
    # print("".join(str(i)))
    a=""
    for i2 in i[1]:
        # print(i2)
        # print((i[i2]))
        # break
        a += ","+str(i2)
    
    l.append(a)
df2 = pd.DataFrame(l,index=None,columns=None)
print(df2.shape)
df2.to_csv('log detection/dataset/ano.csv')


######### Find Unique ################
normal = pd.read_csv('log detection/dataset/nom.csv',index_col=0,header=None)
anomal = pd.read_csv('log detection/dataset/ano.csv',index_col=0,header=None)

# print(pd.value_counts(df2.iloc[:,0]))
uni_nor = normal.iloc[:,0].unique()
uni_ano = anomal.iloc[:,0].unique()
l=[]
for i in uni_nor:
    for a in uni_ano:
        # print(i,a)
        # break
        if i == a:
            # print(i)
            l.append(i)
# print(l)
# print(len(l))
l = l[1:]
s = set(l)
print(len(s))
c=0
arr = np.zeros((len(l),2),dtype=np.int)
for idx,i in enumerate(l):
    # print(i)
    count = normal.iloc[:,0]==i
    c_nor = np.sum(count)
    count = anomal.iloc[:,0]==i
    c_ano = np.sum(count)
    arr[idx] = [c_nor,c_ano]
    # print(count)
    # print(c)
    # break
# print(arr,np.sum(arr,axis=0))
# print(l)
overlap = np.array(l).reshape((-1,1))
a = np.concatenate([overlap,arr],axis=1)
print(a)
pd.DataFrame(a,)