from __future__ import print_function

import torch
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

import argparse
import datasets
import models


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--filename', type=str, default='',
					help='model to be converted')
args = parser.parse_args()

load = torch.load(args.filename, map_location=lambda storage, loc: storage)
torch.save(load, "{}_CPU.pth".format(args.filename.split('.')[0]))

