import argparse
from torchvision import transforms
import datasets
import torch
from sys import exit
import PIL
import os
import os.path
import numpy as np
from sys import exit

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=0, metavar='N',
					help='starting index')
parser.add_argument('--size', type=int, default=10, metavar='N',
					help='number of images to generate')
parser.add_argument('--type', type=str, default='letters',
					choices=['letters', 'digits'],
					help='number of images to generate')
parser.add_argument('--set', type=str, default='train',
					choices=['train', 'eval'],
					help='number of images to generate')
parser.add_argument('--no-save', type=int, default=0,
					help='no save')
args = parser.parse_args()

if args.set == 'train':
	set_loader = datasets.EMNIST('data', args.type, train=True, download=True,
					transform=transforms.Compose([
						transforms.ToTensor(),
						transforms.ToPILImage()
					]))
else:
	set_loader = datasets.AgirEcole('agirecole', 'val', train=False,
					transform=transforms.Compose([
						transforms.ToTensor(),
						transforms.ToPILImage()
					]))

# for tl in set_loader:
# 	if tl[1] == 27:
# 		print(tl)
# 		img = tl[0].transpose(PIL.Image.FLIP_TOP_BOTTOM).rotate(-90)
# 		img.save("{}_{}.png".format('_', str(unichr(tl[1] + 96)),"PNG"))
# 		exit()

# for tl in set_loader:
# 		print(tl)

y = args.start
end = y + args.size
if end > len(set_loader):
	exit("BAD SIZE")
tensors_lst = []
labels_lst = []
for i in range(y, end):
	# img = set_loader[i][0].transpose(PIL.Image.FLIP_TOP_BOTTOM).rotate(-90)
	img = set_loader[i][0]
	tens = transforms.ToTensor()(img)
	parsed = np.full((1, 1), set_loader[i][1], dtype=np.uint8)
	lab_tens = torch.from_numpy(parsed).view(1).long()
	labels_lst.append(lab_tens)
	tensors_lst.append(tens)
	print(tens.size())
	# img = img.resize((20, 20), PIL.Image.BILINEAR)
	# container = Image.new('RGB', (28, 28))
	if args.type == 'letters':
		img.save("{}_{}_{}.png".format(i, str(unichr(set_loader[i][1] + 96)), str(args.set),"PNG"))
	else:
		img.save("{}_{}_{}.png".format(i, set_loader[i][1]), str(args.set), "PNG")
print(len(tensors_lst))
tensors = torch.cat(tensors_lst)
labels = torch.cat(labels_lst)
print(tensors.size())
print(labels.size())
# print(labels)
if args.no_save:
	exit("No save")
with open(os.path.join('data', 'processed', 'training_letters.pt'), 'wb') as f:
	torch.save((tensors, labels), f)