import torch

a = torch.ones(2, 5, 4)
print(a.shape)
print(a.sum().shape)
print(a.sum(axis=1).shape)
print(a.sum(axis=[0, 2]).shape)
print(a.sum(axis=1, keepdims=True).shape)
print(a.sum(axis=[0, 2], keepdims=True).shape)
