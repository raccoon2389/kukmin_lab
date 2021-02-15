import torch
from torch._C import dtype
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
xy = np.loadtxt('data/diabetes.csv',delimiter=',',dtype=np.float32)
x_data = Variable(torch.from_numpy(xy[:,0:-1])) 
y_data = Variable(torch.from_numpy(xy[:,[-1]]))

 
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

for epoch in range(500):
    y_pred = model(x_data)

    loss= criterion(y_pred,y_data)
    print(epoch,loss.data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# hour_var = Variable(torch.Tensor([[4.0,0.2]]))
# print("predict (after training)", 4, model.forward(hour_var).data[0][0])
