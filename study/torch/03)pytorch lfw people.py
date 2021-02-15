import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
# import pickle
# from sklearn.datasets import fetch_lfw_people

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) ])
trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True,num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True,transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False,num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super(Model,self).__init__()
        self.conv1 = nn.Conv2d(1,8,3,1,0)
        self.maxpool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(8,4,2)
        self.fc1 = nn.Linear(29*21*8,128)
        self.fc2 = nn.Linear(128,5749)


    
    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1,29*21*8)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

for epoch in range(2):
    running_loss =0.0
    # for i,data in enumerate(trainloader)
