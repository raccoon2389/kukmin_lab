import torch
from torch.autograd import Variable

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w = Variable(torch.Tensor([1.0]),require_grad=True)

def forward(x):
    return x*w

def loss(x,y):
    y_pred=forward(x)
    return (y_pred-y)*(y_pred-y)

print("predict (before training)", 4 , forward(4).data[0])

for epoch in range(10):
    for x_val, y_val in zip(x_data,y_data):
        l = loss(x_val,y_val)
        l.backward()
        print("")