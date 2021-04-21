from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
from captcha.image import ImageCaptcha

import tqdm

import string
import random
import time

def seq_generator(size=8, chars=string.ascii_uppercase + string.digits):
	return ''.join(random.choice(chars) for _ in range(size))

captcha_generator = ImageCaptcha(width=140, height=30, font_sizes=[25], fonts=['calibri.ttf'])

save_dir = '../data3/'
N = 10

sequences = []

start = time.time()
for i in tqdm.tqdm( range(N) ):
	
	decider = np.random.gamma(2, 2.0)
	if decider >= 1.3:
		random_size = 8
	elif decider >= 0.8:
		random_size = 7
	else:
		random_size = 6

	seq = seq_generator(random_size)

	data = captcha_generator.generate(seq)
	captcha =  Image.open(data)
	captcha.save(f'../data3/{i}.jpg', "JPEG")
	#np.save(f'{save_dir}/{i}', captcha)
	sequences.append([i, seq])
	

sequences = np.array(sequences)

np.savetxt(f'{save_dir}/all.txt', sequences, delimiter=" ", fmt='%s')

print(f'Total {N} captchas are generated in {time.time()-start} seconds')


