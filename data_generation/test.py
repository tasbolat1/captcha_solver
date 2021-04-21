from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
from captcha.image import ImageCaptcha

# im = Image.open("../captchas/some6.png")
# grey_im = im.convert('L')


# image = ImageCaptcha(width=140, height=30, font_sizes=[25], fonts=['calibri.ttf'])
# #data = image.generate('91J1R3KV')
# data = image.generate('DXKTVHDS')


# fig, ax = plt.subplots(2, figsize=(16,27))
# ax[0].imshow(grey_im, cmap='Greys_r')
# ax[1].imshow(Image.open(data), cmap='Greys_r')

# plt.show()
# import pickle


a = np.load('../data/train/100.npy')

plt.imshow(a, cmap='Greys_r')
plt.show()

b = np.genfromtxt('../data/train/all', delimiter=' ', dtype='str')

print(b)

# plt.imshow(a, cmap='Greys_r')
# plt.show()