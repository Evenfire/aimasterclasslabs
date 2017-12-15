
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as M


# self.fc0 = nn.Linear(28*28, 27)
# x = self.c0(x.view(x.size(0), -1))
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.c0 = nn.Conv2d(1, 16, 1*1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.fc0 = nn.ReLU(inplace=False)

	def forward(self, x):
		print(x.size())
		x = self.c0(x)
		x = self.fc0(x)
		
		return F.log_softmax(x)

		# self.layer1 = nn.Sequential()
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(1, 20, 5),
			# nn.BatchNorm2d(20),
			nn.ReLU(),
			nn.MaxPool2d(2)
			)
		self.layer2 = nn.Sequential(
			nn.Conv2d(20, 50, 5),
			# nn.BatchNorm2d(50),
			nn.ReLU(),
			nn.MaxPool2d(2)
			)
		self.fc1 = nn.Linear(4*4*50, 500)
		self.fc2 = nn.Linear(500, 27)
		self.ceriation = nn.CrossEntropyLoss()
		
	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = x.view(-1, 4*4*50)
		x = self.fc1(x)
		x = self.fc2(x)
		return x

class Net(nn.Module):
	def __init__(self, init_values=[.3, .3]):
		super(Net, self).__init__()
		self.sa = True
		self.l3d = 0#0.4
		self.l31d = 0#0.4
		self.layer1 = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=(5, 5), stride=1, padding=2),
			# nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
			nn.ReLU(),
			nn.MaxPool2d((2,2), stride=2, padding=0)
			)
		self.layer2 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size=(5, 5), stride=1, padding=2),
			nn.ReLU(),
			nn.MaxPool2d((2,2), stride=2, padding=0)
			)
		#+conv?
		self.layer3 = nn.Sequential(
			nn.Linear(7*7*64, 2048),#(7*7?)
			nn.ReLU(),
			nn.Dropout(p=self.l3d)
			)
		self.layer3_1 = nn.Sequential(
			nn.Linear(2048, 27),
			# nn.ReLU(),
			# nn.Dropout(p=self.l31d)
			)
		#one more ?
		# self.layer4 = nn.Sequential(
		# 	nn.Linear(512, 27),#add input data for C: None?
		# 	# nn.ReLU()
		# 	)
		self.ceriation = nn.CrossEntropyLoss(weight=None, size_average=self.sa)
		#size_average ?
		#use weight if set is unbalanced(agirecole?)
		# nn.CrossEntropyLoss(weight=None, size_average=True)
		# nn.Dropout2d(p=0.5)

	def set_sa(self, size_average):
		self.sa = size_average

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = x.view(-1, 7*7*64)
		x = self.layer3(x)
		x = self.layer3_1(x)
		# x = self.layer4(x)
		return x
