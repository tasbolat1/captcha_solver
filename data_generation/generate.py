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

save_dir = '../data2/test'
N = 10000

sequences = []

start = time.time()
for i in tqdm.tqdm( range(N) ):
	
	seq = seq_generator()

	data = captcha_generator.generate(seq)
	
	captcha = np.array( Image.open(data), dtype=np.uint8 )
	np.save(f'{save_dir}/{i}', captcha)
	sequences.append([i, seq])
	

sequences = np.array(sequences)

np.savetxt(f'{save_dir}/all.txt', sequences, delimiter=" ", fmt='%s')

print(f'Total {N} captchas are generated in {time.time()-start} seconds')


