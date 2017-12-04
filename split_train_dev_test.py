import argparse
from torchvision import transforms
import datasets
import torch
from sys import exit
import PIL
from PIL import Image
import os
import os.path
import numpy as np
from sys import exit
from random import shuffle

def load_set():
	root='agirecole'
	name='val_new'
	print("Loading dataset {} {}".format(root, name))
	loaded_set = datasets.AgirEcole(root, name, train=True, download=True,
						transform=transforms.Compose([
							transforms.ToTensor()
						]))

	train_loader = torch.utils.data.DataLoader(
			datasets.AgirEcole(root, name, train=True, download=False,
					transform=transforms.Compose([
						transforms.ToTensor()
					])),
			shuffle=True)
	print("Done loading dataset")
	return loaded_set, train_loader

def save_pt(tensors, labels, name):
	root = 'agirecole'
	file_name = "data-{}.pt".format(name)
	print("Save - PT to {}".format(os.path.join(root, 'processed', file_name)))
	with open(os.path.join(root, 'processed', file_name), 'wb') as f:
		torch.save((tensors, labels), f)

def label_to_tensor(label):
	parsed = np.full((1, 1), label, dtype=np.uint8)
	lab_tens = torch.from_numpy(parsed).view(1).long()
	return lab_tens

def prep_tensors_labels(lst):
	tensors_lst = [torch.mul(tens, 255.).byte() for tens, lab in lst]
	labels_lst = [label_to_tensor(lab) for tens, lab in lst]
	tensors = torch.cat(tensors_lst)
	labels = torch.cat(labels_lst)
	return tensors, labels

load, tr = load_set()

print(len(load))

lst = []
for tl in load:
	lst.append(tl)
shuffle(lst)
print(len(lst))

size = len(lst) * 0.8
train_lst = []
for i in range(0, int(size)):
	train_lst.append(lst[i])
print(len(train_lst))

size2 = len(lst) * 0.1
dev_lst = []
for i in range(int(size), int(size) + int(size2) + 1):
	dev_lst.append(lst[i])
print(len(dev_lst))

test_lst = []
for i in range(int(size) + int(size2) + 1, int(size) + (int(size2) + 1)*2):
	test_lst.append(lst[i])
print(len(test_lst))

tens, labs = prep_tensors_labels(train_lst)
save_pt(tens, labs, 'train')

tens, labs = prep_tensors_labels(dev_lst)
save_pt(tens, labs, 'dev')

tens, labs = prep_tensors_labels(test_lst)
save_pt(tens, labs, 'test')

