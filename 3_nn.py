import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import readline


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Create convolution layer with input image and square kernel of size 5
        self.conv1 = nn.Conv2d(1, 6, 5)
        # Create convolution layer with input 6 and output 16,
        # and square kernel size of 5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Make a fully connected layer of input size 400 and output of 120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # Make a fully connected layer of input size 120 and output size of 84
        self.fc2 = nn.Linear(120, 84)
        # Make a fully connected layer of input size 84 and output size 10
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Create 2x2 max pooling layer after each convolution layer
        # with Relu layer
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))

        # Create relu layers after fully connected layer 1 and 2
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # output is vector of length 10
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
params = list(net.parameters())
print(len(params))
print(params[0].size())
print(params[1].size())
print(params[2].size())
print(params[3].size())
print(params[4].size())
print(params[5].size())
print(params[6].size())
print(params[7].size())
print(params[8].size())
print(params[9].size())
input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)
net.zero_grad()
out.backward(torch.randn(1, 10))
exit()
params = list(net.parameters())
print(len(params))
print(params[0].size())
input = Variable(torch.randnn(1, 1, 32, 32))
input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)
print(out)
net.zero_grad()
out.backward(torch.randn(1, 10))
output = net(input)
target = Variable(torch.arange(1, 11))
target = target.view(1, -1)

criteron = nn.MSELoss()

loss = criteron(output, target)
print(loss)
loss.grad_fn
print(loss.grad_fn)
print(loss.grad_fn.next_functions[0][0])
net.zero_grad()
print(net.conv1.bias.grad)
loss.backward()
print(net.conv1.bias.grad)
learning_rate = .01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

optimizier = optim.SGD(net.parameters(), lr=.01)
optimizier.zero_grad()
output = net(input)
loss = criteron(output, target)
loss.backward()
optimizier.step()

readline.write_history_file('./3_autograd.py')
