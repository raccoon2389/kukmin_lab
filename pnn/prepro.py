import torch
import pandas as pd
import numpy as np
import tqdm

INPUT_SIZE = 10

OUTPUT_SIZE =18

BATCH_SIZE=256

HIDDEN_SIZE = 20

df = pd.read_csv('pnn/data/req.csv',index_col=0).values
# print(df)


def str2ascii(data):
    # print(data)
    arr = [ord(c) for c in data[0]]
    arr = torch.LongTensor(arr).view((-1,))
    return arr

def padding(data,max):
    print(data)
    print(max)
    seq_tensor=torch.zeros((len(data),max)).long()
    for idx, seqs in tqdm.tqdm(enumerate(data)):
        # print(seq_tensor.shape)
        # print(seq_tensor[idx,:])
        # print(data[idx])
        seq_tensor[idx, 0]= torch.LongTensor(seqs)
    return seq_tensor

def to_seq(data):
    # print(data)
    seqs=[]
    seq_max=0
    for seq in data:
        seqs.append(str2ascii(seq))
        # print(seqs)
        if seq_max<len(seq):
            seq_max = len(seq)
    
    global max_seq 
    max_seq= seq_max
    return padding(seqs,seq_max)

y = np.load('pnn/data/label.npy').reshape((-1,1))
y = pd.DataFrame(y)
df = to_seq(df)
print(df.shape)

x = pd.DataFrame(df)
pd.concat((x,y),axis=1,ignore_index=True).to_csv('pnn/data/labeled.csv')

