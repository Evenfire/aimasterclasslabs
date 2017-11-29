import argparse
from torchvision import transforms
import datasets
from sys import exit
import PIL

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
	end = len(set_loader)
for i in range(y, end):
	img = set_loader[i][0].transpose(PIL.Image.FLIP_TOP_BOTTOM).rotate(-90)
	# img = img.resize((20, 20), PIL.Image.BILINEAR)
	# container = Image.new('RGB', (28, 28))
	if args.type == 'letters':
		img.save("{}_{}_{}.png".format(i, str(unichr(set_loader[i][1] + 96)), str(args.set),"PNG"))
	else:
		img.save("{}_{}_{}.png".format(i, set_loader[i][1]), str(args.set), "PNG")
