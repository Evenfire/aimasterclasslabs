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

def main():
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
	parser.add_argument('--no-pt', type=int, default=0,
						help='no save pt')
	parser.add_argument('--no-img', type=int, default=0,
						help='no save img')
	parser.add_argument('--rectify', type=int, default=0,
						help='rectify images to appear as normal letters')
	parser.add_argument('--no-mod', type=int, default=0,
						help='no image modification')
	parser.add_argument('--visu', type=int, default=0,
						help='no image modification')
	args = parser.parse_args()

	#### LOAD DATA ####
	set_loader = load_set(args)

	#### HANDLE ARGS ####
	if args.start + args.size > len(set_loader):
		exit("BAD SIZE")

	tensors_lst = []
	labels_lst = []
	for i in range(args.start, args.start + args.size):
		img = set_loader[i][0]
		if args.rectify:
			img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM).rotate(-90)
		if not args.no_mod:
			tlst, llst = gen_smaller_translate(img, 20, set_loader[i][1], i, args)
			tensors_lst = tensors_lst + tlst
			labels_lst = labels_lst + llst

		if args.visu:
			save_img(img, args, i, set_loader[i][1], "_")

		labels_lst.append(label_to_tensor(set_loader[i][1]))
		tensors_lst.append(img_to_tensor(img))

	tensors = torch.cat(tensors_lst)
	labels = torch.cat(labels_lst)
	print("tensors: "),
	print(tensors.size())
	print("labels: "),
	print(labels.size())
	save_pt(tensors, labels, args)

	if args.no_img:
		exit("No Save - IMG")
	print("Save - IMG")
	set_loader = load_set(args)
	for i in range(args.start, args.start + args.size):
		save_img(set_loader[i][0], args, i, set_loader[i][1], "_")

def gen_smaller_translate(img, new_size, label, index, args):
	tensors_lst = []
	labels_lst = []

	for x in range(3):
		for y in range(3):
			_x = x * 4
			_y = y * 4
			new_img = img.resize((new_size, new_size), PIL.Image.BILINEAR)
			container = Image.new('L', (28, 28))
			container.paste(new_img, (_x, _y, new_size + _x, new_size + _y))
			new_img = container
			labels_lst.append(label_to_tensor(label))
			tensors_lst.append(img_to_tensor(new_img))
			if args.visu:
				save_img(new_img, args, index, label, "{}{}{}".format(_x, _y, index))
	return tensors_lst, labels_lst

def label_to_tensor(label):
	parsed = np.full((1, 1), label, dtype=np.uint8)
	lab_tens = torch.from_numpy(parsed).view(1).long()
	return lab_tens

def img_to_tensor(img):
	tens = transforms.ToTensor()(img)
	tens = torch.mul(tens, 255.)
	tens = tens.byte()
	return tens

def load_set(args):
	print("Loading dataset {} {}".format(args.set, args.type))
	if args.set == 'train':
		loaded_set = datasets.EMNIST('data', args.type, train=True, download=True,
						transform=transforms.Compose([
							transforms.ToTensor(),
							transforms.ToPILImage()
						]))
	else:
		loaded_set = datasets.AgirEcole('agirecole', 'val', train=False,
						transform=transforms.Compose([
							transforms.ToTensor(),
							transforms.ToPILImage()
						]))
	print("Done loading dataset")
	return loaded_set

def save_pt(tensors, labels, args):
	if args.no_pt:
		print("No save - PT")
		return
	print("Save - PT")
	with open(os.path.join('data', 'processed', 'training_letters.pt'), 'wb') as f:
		torch.save((tensors, labels), f)

def save_img(img, args, index, label, supp):
	if args.type == 'letters':
		img.save("{}_{}_{}_{}.png".format(index, str(unichr(label + 96)), args.set, supp),"PNG")
	else:
		img.save("{}_{}_{}_{}.png".format(index, label, args.set, supp), "PNG")

if __name__=="__main__":
	main()



