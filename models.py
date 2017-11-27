
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
		self.conv1 = nn.Conv2d(1, 20, 5, 1)
		self.bn1 = nn.BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True)
		self.conv2 = nn.Conv2d(20, 50, 5, 1)
		self.bn2 = nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True)
		self.fc1 = nn.Linear(4*4*50, 500)
		self.fc2 = nn.Linear(500, 27)
		self.ceriation = nn.CrossEntropyLoss()
	    
	def forward(self, x):
	    x = self.conv1(x)
	    x = F.max_pool2d(x, 2, 2)
	    x = self.bn1(x)
	    x = F.relu(x)
	    x = self.conv2(x)
	    x = self.bn2(x)
	    x = F.max_pool2d(x, 2, 2)
	    x = F.relu(x)
	    x = x.view(-1, 4*4*50)
	    x = self.fc1(x)
	    x = self.fc2(x)
	    return x
