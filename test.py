import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from resnet import resnet50

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.basenet = torchvision.models.resnet50(pretrained=True)
        self.stride = 1
        self.a = nn.Conv2d(1,3,3)
        self.b = nn.Conv2d(3,3,3)
        
    def forward(self,input):
        pass
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        #self.basenet = torchvision.models.resnet50(pretrained=True)
        self.stride = 1
        self.a = nn.Conv2d(1,3,3)
        self.b = nn.Conv2d(3,3,3)
        self.c = Net()
        self.d = nn.ReLU()
        
    def forward(self,input):
        pass
cc = Net2()

SGD = optim.SGD(cc.parameters(),lr=0.01,momentum=0.9)
for m in cc.modules():
    print(m)
print("====================")
print(type(cc.state_dict()['a.weight']))
print(type(list(cc.parameters())[0]))
b = cc.state_dict()
print("a.weight" in b)
print(b['a.weight'])
print(b)

d = torch.ones([3,4,4])
d = Variable(d)
print(type(d.size()[0]))
print(d.size()[0])
print(d[1,:,:])
e = d.sum(-1).unsqueeze(-1)
print(e)
print(e.size())
f = d.sum(-2).unsqueeze(-2)
print(f)
print(f.size())
print(f.is_contiguous())

g = torch.cat([d,d],0)
print(g.is_contiguous())

h = torch.transpose(f, -1, -2)
print(h)
print(h.is_contiguous())

k = d[0:2,:,:]
print(k.size())
print(type(b))

for item in cc.modules():
    print(item)