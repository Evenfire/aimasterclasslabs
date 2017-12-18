import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
import models
import datasets


import argparse

parser = argparse.ArgumentParser(description='first tryyeeeee')
parser.add_argument('--filename', type=str, default='',
					help='model to be tested')
args = parser.parse_args()

# model = models.Net()

# print(model)

# # print([p for p in model.parameters()])

# perf = 0.75344

# torch.save({'net' : model, 
# 			'perf' : perf},
# 			'try.pth')

load = torch.load(args.filename, map_location=lambda storage, loc: storage)
model = load['net']
print(model)
# print([p for p in model.parameters()])
# print(load['perf'])

# for p in model.parameters():
# 	p.requires_grad = False
# for p in model.layer3.parameters():
# 	p.requires_grad = True
# for p in model.layer3_1.parameters():
# 	p.requires_grad = True
# for p in model.layer3_2.parameters():#
# 	p.requires_grad = True#

print([p.requires_grad for p in model.parameters()])