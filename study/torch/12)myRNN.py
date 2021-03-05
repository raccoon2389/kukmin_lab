import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import dataloader,dataset
from torch.autograd.variable import Variable
import pandas as pd
import os

DIR_IN='study/data/names/'


files = os.listdir(DIR_IN)
files =[p.replace(".txt","") for p in files ]
# print(files)
# index = [p for p in range(len(files))]
INPUT_SIZE = 10

OUTPUT_SIZE =18

BATCH_SIZE=256

HIDDEN_SIZE = 20

def str2ascii(data):
    arr = [ord(c) for c in data]
    arr = torch.LongTensor(arr)
    return arr

def padding(data,max):
    # print(data)
    seq_tensor=torch.zeros((len(data),max)).long()
    for idx, seqs in enumerate(data):
        # print(seq_tensor.shape)
        # print(seq_tensor[idx,:])
        # print(data[idx])
        seq_tensor[idx, :seqs.size(0)]= torch.LongTensor(seqs)
    return seq_tensor

def to_seq(data):
    # print(data)
    seqs=[]
    seq_max=0
    for seq in data:
        seqs.append(str2ascii(seq))
        # print(len(seq))
        if seq_max<len(seq):
            seq_max = len(seq)
    
    global max_seq 
    max_seq= seq_max
    return padding(seqs,seq_max)



class Set(dataset.Dataset):
    def __init__(self,switch) -> None:
        super().__init__()
        if switch=="train":
            f = pd.read_csv('study/data/names_train.csv',header=None,index_col=None,encoding='utf-8')
        else:
            f = pd.read_csv('study/data/names_train.csv',header=None,index_col=None,encoding='utf-8')
        self.x_data = to_seq(f.iloc[:,0].to_numpy())
        self.y_data = f.iloc[:,1].to_numpy()
        self.y_data = torch.LongTensor([files.index(p) for p in self.y_data])
        print(self.y_data.size())
        self.len = len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.len

class Model(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,n_layer=1):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layer

        self.embed = nn.Embedding(128,hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layer)
        self.fc1 = nn.Linear(hidden_size,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,output_size)

    def forward(self,x):
        batch_size = x.size(0)
        x= x.t()
        # print(x[:,0])
        x= self.embed(x)
        self.gru.flatten_parameters()
        x,_ = self.gru(x)
        x = self.fc1(x[-1])
        return x
    
train_set = Set("train")
test_set = Set("test")

train_loader = dataloader.DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
test_loader = dataloader.DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=True)

model = Model(INPUT_SIZE,HIDDEN_SIZE,OUTPUT_SIZE)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(),lr=0.1)

def train():
    total_loss = 0
    for epoch in range(100):
        for i,data in enumerate(train_loader,0):
            inputs , labels = data
            # print(inputs)
            inputs, labels = Variable(inputs), Variable(labels)
            # print(inputs)
            y_pred = model(inputs)
            
            loss = criterion(y_pred,labels)
            total_loss += loss
            print(f"epoch : {epoch+1} \nloss : {loss}\t{i*BATCH_SIZE} / {BATCH_SIZE}")
            model.zero_grad()
            loss.backward()
            optim.step()
    return total_loss

def train():
    total_loss = 0
    for epoch in range(100):
        for i,data in enumerate(train_loader,0):
            inputs , labels = data
            # print(inputs)
            inputs, labels = Variable(inputs), Variable(labels)
            # print(inputs)
            y_pred = model(inputs)
            
            loss = criterion(y_pred,labels)
            total_loss += loss
            print(f"epoch : {epoch+1} \nloss : {loss}\t{i*BATCH_SIZE} / {BATCH_SIZE}")
            model.zero_grad()
            loss.backward()
            optim.step()
    return total_loss
