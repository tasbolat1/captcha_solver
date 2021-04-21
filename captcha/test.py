from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
from captcha.image import ImageCaptcha

im = Image.open("../captchas/some6.png")
grey_im = im.convert('L')


image = ImageCaptcha(width=140, height=30, font_sizes=[24,24,24], fonts=['calibri.ttf'])
#data = image.generate('91J1R3KV')
data = image.generate('91J1R3KV')





fig, ax = plt.subplots(2, figsize=(16,27))
ax[0].imshow(grey_im, cmap='Greys_r')
ax[1].imshow(Image.open(data).convert('L'), cmap='Greys_r')

plt.show()


import pickle
#xys = pickle.load(open('info1.pkl', 'rb'))

# xys = []
# def onclick(event):
#     print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
#           ('double' if event.dblclick else 'single', event.button,
#            event.x, event.y, int(round(event.xdata)), int(round(event.ydata))))

#     xys.append([int(round(event.ydata)), int(round(event.xdata))])
    
# fig, ax = plt.subplots()
# cid = fig.canvas.mpl_connect('button_press_event', onclick)
# ax.imshow(grey_im, cmap='Greys_r')
# plt.show()



# pickle.dump(xys, open('info3.pkl', 'wb'))
# b = np.ones_like(a)*255
# for i,j in xys:
# 	b[i,j] = V

# c = a.copy()
# c[b == V] = 255

# fig, ax = plt.subplots(3, figsize=(16,27))
# #ax[0].imshow(im)
# ax[0].imshow(grey_im, cmap='Greys_r')
# ax[1].imshow(b, cmap='Greys_r')
# ax[2].imshow(c, cmap='Greys_r')
# plt.show()

# xys1 = pickle.load(open('info1.pkl', 'rb'))
# xys2 = pickle.load(open('info2.pkl', 'rb'))
# xys3 = pickle.load(open('info3.pkl', 'rb'))

# xys = xys1 + xys2 + xys3
# b = np.ones_like(a)*255
# for i,j in xys:
# 	b[i,j] = V

# np.save('circle_noise', b)

# c = a.copy()
# c[b == V] = 255

# fig, ax = plt.subplots(3, figsize=(16,27))
# #ax[0].imshow(im)
# ax[0].imshow(grey_im, cmap='Greys_r')
# ax[1].imshow(b, cmap='Greys_r')
# ax[2].imshow(c, cmap='Greys_r')
# plt.show()

