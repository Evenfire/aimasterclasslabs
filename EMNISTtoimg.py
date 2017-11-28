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
args = parser.parse_args()

train_loader = datasets.EMNIST('data', args.type, train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.ToPILImage()
                    ]))

# for tl in train_loader:
# 	if tl[1] == 27:
# 		print(tl)
# 		img = tl[0].transpose(PIL.Image.FLIP_TOP_BOTTOM).rotate(-90)
# 		img.save("{}_{}.png".format('_', str(unichr(tl[1] + 96)),"PNG"))
# 		exit()

# for tl in train_loader:
# 		print(tl)

y = args.start
for i in range(y, y + args.size):
	img = train_loader[i][0].transpose(PIL.Image.FLIP_TOP_BOTTOM).rotate(-90)
	if args.type == 'letters':
		img.save("{}_{}.png".format(i, str(unichr(train_loader[i][1] + 96)),"PNG"))
	else:
		img.save("{}_{}.png".format(i, train_loader[i][1]),"PNG")
