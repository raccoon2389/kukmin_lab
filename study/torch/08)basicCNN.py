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
        conv1 = nn.Conv2d(1,16,kernel_size=2)
        pool = nn.MaxPool2d(2)
        conv2 = nn.Conv2d(16,8,2)
        conv3 = nn.Conv2d(8,4,2)
        fc = nn.Linear(288,32)
        fc2= nn.Linear(32,10)


        self.conv_module = nn.Sequential(
            conv1,
            nn.ReLU(),
            pool,
            conv2,
            nn.ReLU(),
            pool,
            
        )
        self.fc_module = nn.Sequential(
            fc,
            nn.ReLU(),
            fc2
        )
        self.conv_module = self.conv_module.cuda()
        self.fc_module = self.fc_module.cuda()
    def forward(self,x):
        in_size= x.size(0)
        x = self.conv_module(x)
        x = x.view(in_size,288)
        y_pred = self.fc_module(x)
        return F.log_softmax(y_pred)

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
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda()
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1] #max(최대값)
        correct += pred.eq(target.data.view_as(pred)).cuda().sum() #eq => 같은지 비교 // view_as(pred) => pred처럼 봐라(shape) 

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# hour_var = Variable(torch.Tensor([[4.0,0.2]]))
# print("predict (after training)", 4, model.forward(hour_var).data[0][0])
train()
test()
