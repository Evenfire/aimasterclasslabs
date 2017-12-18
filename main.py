from __future__ import print_function
from sys import exit
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import models
import datasets
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
					help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
					help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
					help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
					help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
					help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
					help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
					help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
					help='how many batches to wait before logging training status')
parser.add_argument('--data-path', type=str, default='data/EMNIST', metavar='DPATH',
					help='dataset path (e.g. use to load dataset on Floydhub)')
parser.add_argument('--data-transfer', type=str, default='data/agir', metavar='DPATH',
					help='dataset path (e.g. use to load dataset on Floydhub)')
parser.add_argument('--server', action='store_true', default=False,
					help='specify if program is on Floydhub')
parser.add_argument('--short', action='store_true', default=False,
					help='shorten')
args = parser.parse_args()

#set Cuda
args.cuda = not args.no_cuda and torch.cuda.is_available()
print("Cuda on: {}".format(args.cuda))
torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

def load_data():
	print("Loading data...")
	PIL_imgs = []
	kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
	train_loader = torch.utils.data.DataLoader(
		datasets.EMNIST(args.data_path, 'letters', train=True, download=False,
						transform=transforms.Compose([
							transforms.ToTensor(),
							transforms.Normalize((0.1722,), (0.3309,))#CHECK
						])),
		batch_size=args.batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(
		datasets.EMNIST(args.data_path, 'letters', train=False, transform=transforms.Compose([
							transforms.ToTensor(),
							transforms.Normalize((0.1722,), (0.3309,))
						])),
		batch_size=args.test_batch_size, shuffle=True, **kwargs)
	train_transfer_loader = torch.utils.data.DataLoader(
		datasets.AgirEcole(args.data_transfer, 'train', train=True, download=False,
						transform=transforms.Compose([
							transforms.ToTensor(),
							transforms.Normalize((0.1722,), (0.3309,))#CHECK
						])),
		batch_size=4, shuffle=True)
	test_transfer_loader = torch.utils.data.DataLoader(
		datasets.AgirEcole(args.data_transfer, 'dev', train=False,
						transform=transforms.Compose([
							transforms.ToTensor(),
							transforms.Normalize((0.1722,), (0.3309,))
						])),
		batch_size=4, shuffle=True)
	print("Done")
	return train_loader, test_loader, train_transfer_loader, test_transfer_loader

model = models.Net()
print(model)

if args.cuda:
	print("Preparing model for GPU")#loading model on GPU
	model.cuda()
	print("Done")

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
# optimizer = optim.Adam(model.parameters(), lr=0.001)#, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001)

# def train(epoch, optimizer):
Losses = []
Absc = []
Lr = []
Lr_absc = []
def train(epoch, optimizer, scheduler, train_set, useSchedule):
	model.train()
	model.set_sa(True)
	i = 0
	for batch_idx, (data, target) in enumerate(train_set):
		if args.cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data), Variable(target)
		optimizer.zero_grad()
		output = model(data)
		# loss = F.nll_loss(output, target)
		loss = model.ceriation(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % args.log_interval == 0:
			# if useSchedule:
			# 	scheduler.step(loss.data[0])
			perc = 100. * batch_idx / len(train_set)
			Losses.append(loss.data[0])
			Absc.append(epoch + perc / 10)
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_set.dataset),
				perc, loss.data[0]))
			if args.short and batch_idx == args.log_interval * 3:
				return

def test(test_set, set_name):
	model.eval()
	model.set_sa(False)
	test_loss = 0
	correct = 0
	print("\nTesting on {}...".format(set_name))#progress bar
	for data, target in test_set:
		if args.cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data, volatile=True), Variable(target)
		output = model(data)
		test_loss += model.ceriation(output, target).data[0]
		# test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
		pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
		correct += pred.eq(target.data.view_as(pred)).cpu().sum()
		if args.short:
				break
	test_loss /= len(test_set.dataset)
	print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
			test_loss, correct, len(test_set.dataset),
			100. * correct / len(test_set.dataset)))
	return float(correct) / len(test_set.dataset)

# print([l for l in Losses])
# print([a for a in Absc])
# Absc = [a * 10 for a in Absc]
def plot():
	prefix = '/output/' if args.server else ""
	
	plt.figure(1)
	plt.figure(figsize=(100, 5))
	plt.plot(Absc, Losses, 'ro')
	plt.axis([0, args.epochs + 1, 0, int(max(Losses) + 0.1 * max(Losses) + 1)])
	plt.savefig('{}loss.png'.format(prefix), bbox_inches='tight')

	plt.figure(2)
	plt.figure(figsize=(100, 5))
	plt.plot(Lr, 'bs')
	plt.savefig('{}lr.png'.format(prefix), bbox_inches='tight')

def get_time():
	return time.strftime("%d-%m-%Y_%H:%M:%S_%Z")


def phase1(train_set, test_set, agir_test):
	print("\nPHASE 1")
	new_lr = 0.001
	# Lr.append(new_lr)
	# best_perf = 0.0
	optimizer = optim.Adam(model.parameters(), lr=new_lr)#, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001)
	# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
	scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.7, last_epoch=-1)
	for epoch in range(1, args.epochs + 1):
		train(epoch, optimizer, scheduler, train_set, True)
		perf_emnist = test(test_set, 'emnist')
		perf_agir = test(agir_test, 'agir')
		# new_lr = 0.001 / (1 + 10*epoch)
		# Lr.append(optimizer.state_dict())
		#graph
		Lr.append(optimizer.param_groups[0]['lr'])
		scheduler.step()
		if epoch == args.epochs:
			print("Saving model")
			prefix = "/output/" if args.server else ""
			torch.save({'net' : model, 'perf' : (perf_emnist, perf_agir)}, "{}model_emnist_{}.pth".format(prefix, get_time()))
			print("Done")
	return Lr[-1]

#PHASE2 changes: datasets, optimizer paramters, requires_grad (no scheduler)
def phase2(train_set, test_set, lr):
	print("\nPHASE 2")
	new_lr = lr
	# Lr.append(new_lr)
	# best_perf = 0.0

	for p in model.parameters():
		p.requires_grad = False
	for p in model.layer3.parameters():
		p.requires_grad = True
	for p in model.layer3_1.parameters():
		p.requires_grad = True
	for p in model.layer3_2.parameters():#
		p.requires_grad = True#
	# model.l3d = 0

	optimizer = optim.Adam([
					{'params': model.layer3.parameters()},
					{'params': model.layer3_1.parameters()},
					{'params': model.layer3_2.parameters()}
							], lr=new_lr)#, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001)
	# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
	max_epoch = 10
	for epoch in range(1, max_epoch + 1):
		train(epoch, optimizer, None, train_set, False)
		perf_agir = test(test_set, 'agir')
		#graph
		Lr.append(optimizer.param_groups[0]['lr'])
		if epoch == max_epoch:
			print("Saving model")
			prefix = "/output/" if args.server else ""
			torch.save({'net' : model, 'perf' : (None, perf_agir)}, "{}model_agir_{}.pth".format(prefix, get_time()))
			print("Done")


def new_lr1(lr0, epoch, decay_rate):
	return lr0 / (1 + decay_rate * (epoch - 1))
	
def new_lr2(lr0, epoch, decay_rate):
	return (0.95**(epoch - 1)) * lr0

#break function or use globals for model, etc.

p1_train, p1_test, p2_train, p2_test = load_data()
lr = phase1(p1_train, p1_test, p2_test)
phase2(p2_train, p2_test, lr)
plot()
