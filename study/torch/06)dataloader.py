import torch
from torch._C import dtype
from torch.utils import data
from torch.utils.data.dataloader import Dataset,DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class DiabetesDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        xy = np.loadtxt('data/diabetes.csv',delimiter=',',dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:,0:-1])
        self.y_data = torch.from_numpy(xy[:,[-1]])

    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

dataset = DiabetesDataset()
train_loader = DataLoader(dataset = dataset,
                        batch_size=32,
                        shuffle=True,
                        num_workers=0)

class Model(torch.nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(8,16)#on in one out
        self.linear2 = torch.nn.Linear(16,8)
        self.linear3 = torch.nn.Linear(8,1)
    def forward(self,x):
        x = F.relu(self.linear(x))
        x = F.relu(self.linear2(x))
        
        y_pred = F.sigmoid(self.linear3(x))
        return y_pred

model = Model()

criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

for epoch in range(2):
    for i, data in enumerate(train_loader,0):
        inputs, labels = data
        inputs,labels = Variable(inputs), Variable(labels)
        y_pred = model(inputs)

        loss= criterion(y_pred,labels)
        print(epoch,loss.data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# hour_var = Variable(torch.Tensor([[4.0,0.2]]))
# print("predict (after training)", 4, model.forward(hour_var).data[0][0])
