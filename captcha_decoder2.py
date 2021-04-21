from PIL import Image, ImageFilter, ImageMath
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import erosion, dilation, flood, closing, opening
from skimage.filters import median


original_im = Image.open("captchas/some3.png")
original_im = original_im.convert('L')
# apply median filter

kernel = np.array([ [1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1] ])

#im1 = np.array(original_im)
#im1 = median(im1, kernel)
im1 = original_im.filter(ImageFilter.MedianFilter(5))



a = np.array(original_im)
b = np.array(im1)

mask = ( (a<255) & (b<255) )
c = a.copy()
c[~mask] = 255


d = dilation(c)
e = c-d
e[e < 30] = 0
e = 255 - e


kernel = np.array([ [0, 1, 0],
					[0, 1, 0],
					[0, 1, 0] ])
f = erosion(e, kernel)

g = a.copy()
g[f == 255] = 255

fig, ax = plt.subplots(8)
ax[0].imshow(original_im, cmap='Greys')
ax[1].imshow(im1, cmap='Greys')
ax[2].imshow(c, cmap='Greys')
ax[3].imshow(d, cmap='Greys')
ax[4].imshow(e, cmap='Greys')
ax[5].imshow(f, cmap='Greys')
ax[6].imshow(g, cmap='Greys')
ax[7].imshow(Image.fromarray(g).filter(ImageFilter.MedianFilter(3)), cmap='Greys')

plt.show()