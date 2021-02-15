import torch
import torch.nn as nn
from torch._C import dtype
from torch.utils import data
from torch.utils.data.dataloader import Dataset,DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as Datasets

torch.cuda.device(0)

train_loader =DataLoader(Datasets.MNIST('data/',train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307),(0.3081))])),batch_size=256,shuffle=True)
test_loader = DataLoader(Datasets.MNIST('data/',train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307),(0.3081))])),batch_size=256,shuffle=True)



class Model(torch.nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        linear = torch.nn.Linear(784,1024)#on in one out
        linear2 = torch.nn.Linear(1024,256)
        linear3 = torch.nn.Linear(256,10)

        self.fc_module = nn.Sequential(
            linear,
            nn.ReLU(),
            linear2,
            nn.ReLU(),
            linear3
        )
        self.fc_module = self.fc_module.cuda()

    def forward(self,x):
        x = x.view(-1,784)
        y_pred = self.fc_module(x)
        return y_pred

model = Model()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
def train():
    for epoch in range(2):
        for batch_idx, (data,target) in enumerate(train_loader):
            data, target= Variable(data).cuda(), Variable(target).cuda()
            
            optimizer.zero_grad()
            output = model(data)

            loss= criterion(output,target)
            # print(epoch,loss.data)

            loss.backward()
            optimizer.step()
            if batch_idx %10 ==0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                batch_idx*len(data),len(train_loader.dataset)
                ,100*batch_idx/len(train_loader),loss.data))

def test():
    model.eval()
    test_loss=0
    correct = 0
    for data, target in test_loader:
        data,target = Variable(data,volatile=True).cuda(), Variable(target).cuda()
        output = model(data)
        test_loss += criterion(output,target).data
        pred = torch.max(output.data,1)[1]
        correct += pred.eq(target.data.view_as(pred).cuda().sum())
    
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f},Accuracy:{}/{} ({:.0f}%)\n'.format(
        test_loss,correct,len(test_loader.dataset),
        100.*correct/len(test_loader.dataset)
    ))
# hour_var = Variable(torch.Tensor([[4.0,0.2]]))
# print("predict (after training)", 4, model.forward(hour_var).data[0][0])
train()
test()
