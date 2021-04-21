from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
from captcha.image import ImageCaptcha

im = Image.open("captchas/some6.png")
grey_im = im.convert('L')

# find arcs
a = np.array(grey_im)
b = np.ones_like(a)*255

# image = ImageCaptcha(width=140, height=30, font_sizes=[14,14,14])
# data = image.generate('1234AF')

# print(data)

fig, ax = plt.subplots(3)
ax[0].imshow(im)
ax[1].imshow(grey_im, cmap='Greys_r')
ax[2].imshow(b, cmap='Greys_r')
plt.show()



