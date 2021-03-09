import pandas as pd
import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.utils.data import dataloader,dataset
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import tqdm
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch.nn.functional as F
BATCH_SIZE = 32
HIDDEN_SIZE = 64
OUTPUT_SIZE = 1
INPUT_SIZE = 5

class Data_Set(dataset.Dataset):
    def __init__(self) -> None:
        super(Data_Set,self).__init__()
        train_x = np.load('log detection/dataset/req2log.npy')
        self.test_x = np.load('log detection/dataset/req2logTEST.npy')
        anomal_x = np.load('log detection/dataset/req2logANOMAL.npy')
        
        train_y = np.zeros((train_x.shape[0]))
        anomal_y = np.zeros((anomal_x.shape[0])) + 1

        self.train_data = np.concatenate([train_x,anomal_x],axis=0)


        scaler = StandardScaler()
        self.train_data = scaler.fit_transform(self.train_data)
        self.train_data = torch.from_numpy(self.train_data).float()
        self.train_label = np.concatenate([train_y,anomal_y]).reshape(-1,1)
    
        self.train_label = torch.from_numpy(self.train_label).float()
        self.len = len(self.train_data)

    def __getitem__(self, index):
        return self.train_data[index],self.train_label[index]

    def __len__(self):
        return self.len

def loader(batch_size):
    sets = Data_Set()
    return dataloader.DataLoader(sets,batch_size=batch_size,shuffle=True)

class Model(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(Model,self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.Dropout(0.2),
            # nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size*4),
            nn.Dropout(0.2),
            # nn.BatchNorm1d(hidden_size*4),
            nn.ReLU(),
            # nn.Linear(hidden_size*4,hidden_size*2),
            # nn.Dropout(0.2),
            # nn.ReLU(),
            nn.Linear(hidden_size*4,hidden_size*8),
            # nn.BatchNorm1d(hidden_size*8),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_size*8,output_size)
            )
    def forward(self,x):
        x = self.seq(x)
        return x

def train():
    train_loader = loader(BATCH_SIZE)
    model = Model(input_size=INPUT_SIZE,hidden_size=HIDDEN_SIZE,output_size=OUTPUT_SIZE).cuda()
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adamax(model.parameters())
    for epoch in tqdm.tqdm(range(100)):
        bat_loss=0
        for batch_idx, (data,target) in enumerate(train_loader):
            data, target= Variable(data).cuda(), Variable(target).cuda()
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

def test():
    model = Model(input_size=INPUT_SIZE,hidden_size=HIDDEN_SIZE,output_size=OUTPUT_SIZE).cuda()

    model.load_state_dict(torch.load('log detection/MODEL465.0867614746094'))
    model.eval()
    test_loss = 0
    correct = 0
    train_loader = loader(batch_size=BATCH_SIZE)
    for data, target in train_loader:
        data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda()
        output = model(data)
        # sum up batch loss
        test_loss += F.binary_cross_entropy_with_logits(output, target, size_average=False).data
        # get the index of the max log-probability
        pred = F.sigmoid(output.data).round() #q반올림
        correct += pred.eq(target.data.view_as(pred)).cuda().sum() #eq => 같은지 비교 // view_as(pred) => pred처럼 봐라(shape) 

    test_loss /= len(train_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))

if __name__ == '__main__':
    # train()
    test()

