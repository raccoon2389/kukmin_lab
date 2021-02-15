import torch
import torch.nn.functional as F
from torch.autograd import Variable

x_data = Variable(torch.tensor([[1.0],[2.0],[3.0],[4.0]])) 
y_data = Variable(torch.tensor([[0.],[1.],[0.],[1.]]))

class Model(torch.nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1,1)#on in one out

    def forward(self,x):
        y_pred = F.relu(self.linear(x))
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

hour_var = Variable(torch.Tensor([[4.0]]))
print("predict (after training)", 4, model.forward(hour_var).data[0][0])
