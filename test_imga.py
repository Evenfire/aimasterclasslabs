from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('0_w_train.png')
trans = transforms.ToPILImage()
trans1 = transforms.ToTensor()
img1 = trans(trans1(img))
img1.save('test.png', "PNG")
