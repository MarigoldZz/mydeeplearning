import torch
from torch import nn

print(torch.device('cpu'))
print(torch.cuda.device('cuda'))
print(torch.cuda.device('cuda:0'))
print(torch.cuda.device_count())


def try_gpu(i=0):
    if torch.cuda.device_count() > i:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


print(try_gpu())
print(try_gpu(10))
print(try_all_gpus())

x = torch.tensor([1, 2, 3])
print(x.device)

X = torch.ones(2, 3, device=try_gpu())
print(X)

Y = torch.rand(2, 3)
print(Y)
print(Y.device)

Z = Y.cuda(0)
print(Z)

print(X + Z)
print(Z.cuda(0) is Z)

net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
print(net(X))
print(net[0].weight.data.device)
