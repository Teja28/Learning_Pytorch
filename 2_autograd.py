# import torch, autograd and variable
import torch
from torch.autograd import Variable

# Declare a 2x2 matrix of ones
x = Variable(torch.ones(2,2), requires_grad=True)
print(x)

# Make second leave on computation graph
y = x + 2
print(y)
print(y.grad_fn)

# Make 3rd node on computation graph
z = y*y*3

# Find the value of each element in the matrix
out = z.mean()
print(z,out)

# Backprop
out.backward()
# Declare a tensor and make it a Variable
x = torch.randn(3)
x = Variable(x, requires_grad=True)
y = x*2
# Idk what this does
while y.data.norm() < 1000:
	y = y * 2
print(y)
# Tensor for initial gradient value
gradients = torch.FloatTensor([.1,1,.0001])
y.backward(gradients)
print(x.grad)
import readfile
import readline
readline.write_history_file('./2_autograd.py')
