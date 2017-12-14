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
# import matplotlib.pyplot as plt

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
	parser.add_argument('--array', type=int, default=0,
						help='see img array')
	# parser.add_argument('--no-pt', action='store_true', default=False,
	# 					help='no save pt')
	# parser.add_argument('--no-img', action='store_true', default=False,
	# 					help='no save img')
	# parser.add_argument('--rectify', action='store_true', default=False,
	# 					help='rectify images to appear as normal letters')
	# parser.add_argument('--no-mod', action='store_true', default=False,
	# 					help='no image modification')
	# parser.add_argument('--visu', action='store_true', default=False,
	# 					help='no image modification')
	args = parser.parse_args()

	#### LOAD DATA ####
	set_loader = load_set(args)
	print("set_loader {}".format(len(set_loader)))

	#### HANDLE ARGS ####
	if args.start + args.size > len(set_loader):
		args.size = len(set_loader) - args.start
		# exit("BAD SIZE set_loader {}".format(len(set_loader)))


	### origin 170 clean 157 ### new 1468
	remove_from_agirecole = [135, 118, 13, 146, 154, 162, 166, 25, 54, 6, 80, 88, 9]

	tens_labels = []
	for i in range(args.start, args.start + args.size):
		# if i in remove_from_agirecole:
		# 	continue
		img = set_loader[i][0]
		if args.rectify:
			img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM).rotate(-90)
		if not args.no_mod:
			# img = img.resize((25, 25), PIL.Image.BILINEAR)
			# img = img.rotate(-10, PIL.Image.BILINEAR)
			# container = Image.new('L', (28, 28))
			# # container.paste(img, (0, 4, 24, 28))
			# container.paste(img)
			# img = container
			# toto = np.array(img)
			tens_labels.extend(gen_smaller_translate(img, 20, set_loader[i][1], i, args))
			tens_labels.extend(gen_smaller_rotate(img, set_loader[i][1], i, args, 15))
			tens_labels.extend(gen_smaller_rotate(img, set_loader[i][1], i, args, -15))
			tens_labels.extend(gen_inclination(img, set_loader[i][1], i, args, 'anti'))
			tens_labels.extend(gen_inclination(img, set_loader[i][1], i, args, 'trigo'))
			tens_labels.extend(gen_zoom_out(img, 17, set_loader[i][1], i, args))

		if args.visu:
			save_img(img, args, i, set_loader[i][1], "")
		tens_labels.extend([(img_to_tensor(img), label_to_tensor(set_loader[i][1]))])

	tensors_lst = [tens for tens, lab in tens_labels]
	labels_lst = [lab for tens, lab in tens_labels]
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
	for i in range(args.start, args.start + len(labels_lst)):
		save_img(set_loader[i][0], args, i, set_loader[i][1], "")

# tmp = np.array(img_array[:,k])
# img_array[:,k] = np.roll(img_array[:,k], int(shift(k)))
def gen_inclination(img, label, index, args, direction='anti'):
	tup_ten_lab_lst = []

	img = img.resize((24, 24), PIL.Image.BILINEAR)
	container = Image.new('L', (28, 28))
	if direction == 'anti':
		container.paste(img)
		shift = lambda x: 3 * x / 7
	else:
		container.paste(img, (0, 4, 24, 28))#check
		shift = lambda x: -1 * x / 10
	img = container
	img_array = np.array(img)
	for k in range(img_array.shape[0]):
		if args.rectify:
			img_array[k,:], error = ft_roll(img_array[k,:], int(shift(k)))
		else:
			img_array[:,k], error = ft_roll(img_array[:,k], int(shift(k)))
		if error:
			return []
	img = Image.fromarray(img_array, 'L')
	img = img.resize((26, 26), PIL.Image.LANCZOS)#.BILINEAR
	img = img.resize((28, 28), PIL.Image.LANCZOS)#.BILINEAR
	tup_ten_lab_lst.append((img_to_tensor(img), label_to_tensor(label)))
	if args.array:
		display_img_array(img_array)
	if args.visu:
				save_img(img, args, index, label, "incl_{}".format(direction))
	return tup_ten_lab_lst


def ft_roll(a, shift):#check on rectify
	sum_a = np.sum(a)
	if shift == 0:
		out = np.array(a)
	elif shift < 0:
		out = np.pad(a, ((0, abs(shift))), mode='constant')[abs(shift):]
	else:
		out = np.pad(a, ((shift,0)), mode='constant')[:-shift]
	sum_o = np.sum(out)
	if sum_o == sum_a:
		return out, False
	return out, True

def gen_zoom_out(img, new_size, label, index, args):
	tup_ten_lab_lst = []

	offset = 5
	img = img.resize((new_size, new_size), PIL.Image.BILINEAR)
	container = Image.new('L', (28, 28))
	container.paste(img, (offset, offset, new_size + offset, new_size + offset))
	tup_ten_lab_lst.append((img_to_tensor(container), label_to_tensor(label)))
	if args.visu:
			save_img(container, args, index, label, "out_")
	return tup_ten_lab_lst

def gen_smaller_rotate(img, label, index, args, angle):
	tup_ten_lab_lst = []

	new_img = img.resize((25, 25), PIL.Image.BILINEAR)
	new_img = new_img.rotate(angle, PIL.Image.BILINEAR)
	container = Image.new('L', (28, 28))
	container.paste(new_img)
	tup_ten_lab_lst.append((img_to_tensor(container), label_to_tensor(label)))
	if args.visu:
			save_img(container, args, index, label, "rot_{}".format(angle))
	return tup_ten_lab_lst


def gen_smaller_translate(img, new_size, label, index, args):
	tup_ten_lab_lst = []

	# nb = 2
	for x in range(2):
		for y in range(2):
			_x = x * 4 + 2
			_y = y * 4 + 2
			new_img = img.resize((new_size, new_size), PIL.Image.BILINEAR)
			container = Image.new('L', (28, 28))
			container.paste(new_img, (_x, _y, new_size + _x, new_size + _y))
			new_img = container
			tup_ten_lab_lst.append((img_to_tensor(new_img), label_to_tensor(label)))
			if args.visu:
				save_img(new_img, args, index, label, "trans_{}_{}_{}".format(_x, _y, index))
	return tup_ten_lab_lst

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
		loaded_set = datasets.EMNIST('data/EMNIST', args.type, train=True, download=True,
						transform=transforms.Compose([
							transforms.ToTensor(),
							transforms.ToPILImage()
						]))
	else:
		loaded_set = datasets.AgirEcole('data/agir', 'train', train=False,#val
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
	if args.set == 'train':
		root = 'data/EMNIST'
		file_name = 'training_letters.pt'
	else:
		root = 'data/agir'
		file_name = 'data-val.pt'
	print("Save - PT to {}".format(os.path.join(root, 'processed', file_name)))
	with open(os.path.join(root, 'processed', file_name), 'wb') as f:
		torch.save((tensors, labels), f)

def save_img(img, args, index, label, supp):
	if args.type == 'letters':
		img.save("{}_{}_{}_{}.png".format(index, str(unichr(label + 96)), args.set, supp),"PNG")
	else:
		img.save("{}_{}_{}_{}.png".format(index, label, args.set, supp), "PNG")

def display_img_array(array):
	for t in range(array.shape[0]):
		for tt in range(array.shape[1]):
			print("{:3}".format(array[t][tt])),
		print("")
	print("")

if __name__=="__main__":
	main()



