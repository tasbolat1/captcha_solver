from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import erosion, dilation, flood
def get_crop_sizes(x):
	top, bottom, right, left = 0,0,0,0
	height, width = x.shape
	print(height, width)

	for i in range(height):
		if np.sum(x[i, :]) != 255*width:
			top = i
			break

	for i in reversed(range(height)):
		if np.sum(x[i, :]) != 255*width:
			bottom = i
			break

	for i in range(width):
		print(i)
		if np.sum(x[:,i]) != 255*height:
			left = i
			break
	for i in reversed( range(width) ):
		if np.sum(x[:, i]) != 255*height:
			right = i
			break


	return top, bottom, right, left



def get_char_locs(im2):
	letters = []

	

	inletter = False
	foundletter=False
	start = 0
	end = 0


	for y in range(im2.size[0]): # slice across
	  for x in range(im2.size[1]): # slice down
	    pix = im2.getpixel((y,x))
	    if pix != 255:
	      inletter = True
	  if foundletter == False and inletter == True:
	    foundletter = True
	    start = y

	  if foundletter == True and inletter == False:
	    foundletter = False
	    end = y
	    letters.append((start,end))

	  inletter=False
	return letters

def segment_letters(im2, letters):
	count = 0
	characters = []
	for letter in letters:
	  characters.append( im2.crop(( letter[0] , 0, letter[1], im2.size[1] )) )

	return characters

THRESHOLD = 230


im1 = Image.open("captchas/some6.png")
a = np.array(im1)
a[a == 28] = 210
im1 = Image.fromarray(a)

im = im1.convert('L') # convert to grey

a = np.array(im) # get an array
height, width = a.shape

#a[a < THRESHOLD] = 0
#a[a >= THRESHOLD] = 255


# flood_mask = flood(a,(25, 25))
# a[flood_mask] = 150

b = dilation(a)
b = erosion(b)

letters = get_char_locs(Image.fromarray(b))

b[b < 170] = 0
#c = dilation(b)

# top, bottom, right, left = get_crop_sizes(b.copy())
# print(top, bottom, left, right)
# b = b[top:bottom, left:right]

# NOW we have cleaned outside

# let segment them
im2 = Image.fromarray(b)
characters = segment_letters(im2, letters)

fig, ax = plt.subplots(4+len(characters))
ax[0].imshow(im1)
ax[1].imshow(im)
ax[2].imshow(a)
ax[3].imshow(b)	

for i in range(len(characters)):
	ax[3+i].imshow(characters[i])
# ax[3].imshow(raw_characters[0])
# ax[4].imshow(raw_characters[1])
# ax[5].imshow(raw_characters[2])
# ax[6].imshow(raw_characters[3])
# ax[7].imshow(raw_characters[4])
# ax[8].imshow(raw_characters[5])

plt.show()