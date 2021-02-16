import torch
from torch.autograd.variable import Variable
import torch.nn as nn
import torch.autograd as autograd
import torch.tensor
import sys

HIDDEN_SIZE= 100
N_CHARS = 128
N_CLASSES=18


class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.rnn = nn.RNN(input_size = input_size,
                            hidden_size=hidden_size, batch_first=True)
    
    def forward(self, hidden,x):
        x = x.view(batch_size,seq_len,input_size)
        # print(hidden)
        out,hidden = self.rnn(x,hidden)
        out = out.view(-1,num_classes)
        return hidden,out

    def init_hidden(self):
        return Variable(torch.zeros(num_layers,batch_size,hidden_size))


def str2ascii_arr(msg):
    arr = [ord(c) for c in msg]
    return arr,len(arr)

def pad_sequence(vectorized_seqs, seq_lengths):
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq,seq_len) in enumerate(zip(vectorized_seqs,seq_lengths)):
        seq_tensor[idx,:seq_len] = torch.LongTensor(seq)
    return seq_tensor

def make_variable(names):
    sequence_and_length = [str2ascii_arr(name) for name in names]
    vectorized_seq = [sl[0] for sl in sequence_and_length]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length])
    return pad_sequence(vectorized_seq,seq_lengths)

if __name__ == '__main__':
    classifier = Model(N_CHARS,
                            HIDDEN_SIZE, N_CLASSES)
    arr,_ = str2ascii_arr("adylov")
    inp = Variable(torch.LongTensor([arr]))


indx2char=['h','i','e','l','o']

x_data= [0,1,0,2,3,3]
x_one_hot_look = [[1,0,0,0,0],
            [0,1,0,0,0],
            [0,0,1,0,0],
            [0,0,0,1,0],
            [0,0,0,0,1]]

y_data = [1,0,2,3,3,4]#ihello
x_one_hot = [x_one_hot_look[x] for x in x_data]



inputs = autograd.Variable(torch.Tensor(x_one_hot))
print("input size", inputs.size())
labels = Variable(torch.LongTensor(y_data))


num_classes = 5
input_size = 5
hidden_size = 5
batch_size =1
seq_len = 1
num_layers =1

model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.1)

for epoch in range(100):
    optimizer.zero_grad()
    loss = 0
    hidden = model.init_hidden()

    sys.stdout.write("predicted string: ")
    for input, label in zip(inputs, labels):
        # print(input.size(), label.size())
        hidden, output = model(hidden, input)
        val, idx = output.max(1)
        sys.stdout.write(indx2char[idx.data[0]])
        loss += criterion(output, torch.LongTensor([label]))

    print(", epoch: %d, loss: %1.3f" % (epoch + 1, loss))

    loss.backward()
    optimizer.step()