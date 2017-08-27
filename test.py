"""
This is a basic pytorch tutorial script
source1: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/pytorch_basics/main.py
source2:https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/202_variable.py

#Qingui Liu
#26/08/2017
"""
from __future__ import print_function

import torch
import torchvision
import torch.nn as nn
import torch.cuda as cuda
from torch.autograd import Variable
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets


#basic examples 1
#tensors
x = Variable(torch.Tensor([1]),requires_grad=True)
w = Variable(torch.Tensor([2]),requires_grad=True)
b = Variable(torch.Tensor([3]),requires_grad=True)

y = w * x + b

# gradients.
y.backward()

# print(x.grad)
# print(w.grad)
# print(b.grad)
#convert numpy to tensor, vise versa
np_data = np.arange(6).reshape(2,3)
torch_data = torch.from_numpy(np_data)
tesnsor2array = torch_data.numpy()
# print ('\n numpy array:',np_data,
#        '\n troch tensor:', torch_data,
#        '\n tensor to arr:',tesnsor2array )

data = [-1, -2, -3, 3]
tensor = torch.FloatTensor(data)
#sin, mean
# print (np.sin(data), '\n', torch.sin(tensor))
# print (np.mean(data), '\n', torch.mean(tensor))

a = [[1,2],[3,2]]
b = torch.FloatTensor(a)
# print (np.matmul(a,a), torch.mm(b,b), a,b)

x= torch.Tensor(5,3)
y = torch.rand(5,3)
z= torch.Tensor(5,3)

torch.add(x,y,out=z)

a = z[:,1]
b = a.numpy()
c = torch.from_numpy(b)

if cuda.is_available():
    y = y.cuda()
    x = x.cuda()
    x+y
