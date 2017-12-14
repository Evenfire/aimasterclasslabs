from __future__ import print_function
from sys import exit
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
parser.add_argument('--data-path', type=str, default='data', metavar='DPATH',
					help='dataset path (e.g. use to load dataset on Floydhub)')
parser.add_argument('--server', action='store_true', default=False,
					help='specify if program is on Floydhub')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
print("Cuda on: {}".format(args.cuda))

torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)


PIL_imgs = []
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
	datasets.EMNIST(args.data_path, 'letters', train=True, download=True,
					transform=transforms.Compose([
						transforms.ToTensor(),
						transforms.Normalize((0.1722,), (0.3309,))
					])),
	batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
	datasets.EMNIST(args.data_path, 'letters', train=False, transform=transforms.Compose([
						transforms.ToTensor(),
						transforms.Normalize((0.1722,), (0.3309,))
					])),
	batch_size=args.test_batch_size, shuffle=True, **kwargs)

#load model
model = models.Net()
print(model)

if args.cuda:
	print("Loading model on GPU")
	model.cuda()
	print("Done")

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
# optimizer = optim.Adam(model.parameters(), lr=0.001)#, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001)

# def train(epoch, optimizer):
Losses = []
Absc = []
Lr = []
Lr_absc = []
def train(epoch, optimizer, scheduler):
	model.train()
	i = 0
	for batch_idx, (data, target) in enumerate(train_loader):
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
			scheduler.step(loss.data[0])
			perc = 100. * batch_idx / len(train_loader)
			Losses.append(loss.data[0])
			Absc.append(epoch + perc / 10)
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				perc, loss.data[0]))

def test():
	model.eval()
	model.set_sa(False)
	test_loss = 0
	correct = 0
	for data, target in test_loader:
		if args.cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data, volatile=True), Variable(target)
		output = model(data)
		test_loss += model.ceriation(output, target).data[0]
		# test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
		pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
		correct += pred.eq(target.data.view_as(pred)).cpu().sum()
	test_loss /= len(test_loader.dataset)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))
	return correct / len(test_loader.dataset)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, 'model_best.pth.tar')

new_lr = 0.001
Lr.append(new_lr)
# best_perf = 0.0
optimizer = optim.Adam(model.parameters(), lr=new_lr)#, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
for epoch in range(1, args.epochs + 1):
	train(epoch, optimizer, scheduler)
	perf = test()
	# new_lr = 0.001 / (1 + 10*epoch)
	# Lr.append(new_lr)
	############
	# torch.save(
	# 	{'net': model,
	# 	 'test': perf},
	# 	"model.pth"
	# )
	# save_checkpoint({
		# 	'epoch': epoch + 1,
		# 	'arch': 'test_',#args.arch,
		# 	'state_dict': model.state_dict(),
		# 	'best_prec1': 'test_',
		# 	'optimizer' : optimizer.state_dict(),
		# 	# 'params' : model.parameters(),
		# 	'model' : model
		# }, 0)
	if epoch == args.epochs:
		print("saving model")
		torch.save(
			{'net': model,
			'test': perf},
			"/output/model.pth"
		)

def new_lr1(lr0, epoch, decay_rate):
	return lr0 / (1 + decay_rate * (epoch - 1))
	
def new_lr2(lr0, epoch, decay_rate):
	return (0.95**(epoch - 1)) * lr0

# print([l for l in Losses])
# print([a for a in Absc])
# Absc = [a * 10 for a in Absc]
# plt.figure(1)
# plt.figure(figsize=(100, 5))
# plt.plot(Absc, Losses, 'ro')
# plt.axis([0, args.epochs + 1, 0, int(max(Losses) + 0.1 * max(Losses) + 1)])
# plt.savefig('/output/loss.png', bbox_inches='tight')

# plt.figure(2)
# plt.figure(figsize=(100, 5))
# plt.plot(Lr, 'bs')
# plt.savefig('/output/lr.png', bbox_inches='tight')

