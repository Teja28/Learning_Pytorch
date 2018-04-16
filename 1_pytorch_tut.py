
from __init__ import *
from __future__ import print_function
import torch

## Torch tensor definition
x = torch.Tensor(5,3)
print(x)

## Random Torch Tensor of size 5x3
x = torch.rand(5,3)
print(x)
print(x.size())

## instantiate another random tensor of size 5x3
y = torch.rand(5,3)

## Add tensors syntax 1
print(x+y)

## Add tensors syntax 2
print(torch.add(x,y))

## Store result of addition in a new tensor
result = torch.Tensor(5,3)
torch.add(x,y,out=result)
print(result)

## Add in place - store result in y
y.add_(x)
print(y)

## numpy index slicing supported
print(x[:,1])

## Different ways of reshaping sizes of tensors
x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1,8) # -1 inferes size of x dimension given y dimension
print(x.size(), y.size(), z.size())

## More numpy like functionality
a = torch.ones(5)
print(a)

## Turn torch tensor into ndarray
b = a.numpy()
print(b)
## Add 1 in place to each element of a
a.add_(1)
print(a)
print(b) ## b prints ndarray of 2 (dynamic graph computation)

## Convert numpy array to torch tensor
import numpy as np
a = np.oness(5)
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a,1,out=a)
print(a)
print(b)


import readline
readline.write_history_file('/home/ahj/history')
readline.write_history_file('.')
readline.write_history_file('./history')
