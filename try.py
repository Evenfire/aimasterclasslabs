import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
import models
import datasets


# model = models.Net()

# print(model)

# # print([p for p in model.parameters()])

# perf = 0.75344

# torch.save({'net' : model, 
# 			'perf' : perf},
# 			'try.pth')

load = torch.load('try.pth')
model = load['net']
print(model)
print([p for p in model.parameters()])
print(load['perf'])