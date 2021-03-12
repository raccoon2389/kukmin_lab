import torch
from torch.autograd.variable import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader,dataset
import numpy as np
import pandas as pd
from tqdm import tqdm


with open('log detection/dataset/vocab.pkl',"rb") as f:
    df = pd.read_pickle(f)
f.close

path_con = np.load('log detection/dataset/voca/concat.npy')

path_con = torch.LongTensor(path_con)

train_x = np.load('log detection/dataset/req2log2.npy')
anomal_x = np.load('log detection/dataset/req2logANOMAL2.npy')

input_lengths = torch.LongTensor([torch.max(torch.nonzero(path_con[i, :].data))+1 for i in range(path_con.size(0))])
input_lengths, sorted_idx = input_lengths.sort(0, descending=True)
path_con = path_con[sorted_idx]


INPUT_SIZE = path_con.shape[1]
BATCH_SIZE = 32
OUTPUT_SIZE = 1
HIDDEN_SIZE = 64
N_LAYER = 1

class Dataset_(dataset.Dataset):
    def __init__(self) -> None:
        super(Dataset_,self).__init__()
        self.data = torch.LongTensor(path_con)
        train_y = np.zeros((train_x.shape[0]))
        anomal_y = np.zeros((anomal_x.shape[0])) + 1
        self.train_label = np.concatenate([train_y,anomal_y]).reshape(-1,1)
        self.train_label = torch.Tensor(self.train_label)
        self.len = len(self.data)
    def __getitem__(self, index):
        return self.data[index],self.train_label[index]
    def __len__(self):
        return self.len



class Model(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,n_layer):
        super(Model,self).__init__()
        self.embed = nn.Embedding(len(df),hidden_size,padding_idx=1)
        self.gru = nn.GRU(hidden_size,hidden_size,n_layer,dropout=0.2,bidirectional=False)
        self.fc1 = nn.Linear(hidden_size,hidden_size*2)
        self.out =  nn.Linear(hidden_size*2,output_size)
    def forward(self,x):
        # print(x)
        # x=x.t()
        x = self.embed(x)
        self.gru.flatten_parameters()
        x, _= self.gru(x)
        x= self.fc1(x[:,-1])
        x = F.relu(x)
        out = self.out(x)
        return out

def loader(batch_size):
    sets = Dataset_()
    return dataloader.DataLoader(sets,batch_size=batch_size,shuffle=True,num_workers=4)

def train():
    train_loader = loader(BATCH_SIZE)
    model = Model(input_size=INPUT_SIZE,hidden_size=HIDDEN_SIZE,output_size=OUTPUT_SIZE,n_layer=N_LAYER)
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adamax(model.parameters())
    for epoch in tqdm(range(100)):
        bat_loss=0
        for batch_idx, (data,target) in enumerate(train_loader):
            data, target= Variable(data), Variable(target)
            # print('opt')
            optimizer.zero_grad()
            output = model(data)
            loss= criterion(output,target)
            bat_loss += loss.data
            # print(epoch,loss.data)

            loss.backward()
            optimizer.step()
            if batch_idx %int(len(train_loader.dataset)/BATCH_SIZE)==0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                batch_idx*len(data),len(train_loader.dataset)
                ,100*batch_idx/len(train_loader),bat_loss/(batch_idx+1)))
    torch.save(model.state_dict(),f"log detection/MODEL{bat_loss}")

if __name__ == "__main__":
    train()