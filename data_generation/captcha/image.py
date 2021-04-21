# coding: utf-8
"""
    captcha.image
    ~~~~~~~~~~~~~

    Generate Image CAPTCHAs, just the normal image CAPTCHAs you are using.
"""

import os
import random
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter
from PIL.ImageDraw import Draw
from PIL.ImageFont import truetype
try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO


from PIL import Image, ImageOps


DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
DEFAULT_FONTS = [os.path.join(DATA_DIR, 'DroidSansMono.ttf')]

__all__ = ['ImageCaptcha']

table  =  []
for  i  in  range( 256 ):
    table.append( i * 1.97 )


class _Captcha(object):
    def generate(self, chars, format='png'):
        """Generate an Image Captcha of the given characters.

        :param chars: text to be generated.
        :param format: image file format
        """
        im = self.generate_image(chars)
        out = BytesIO()
        im.save(out, format=format)
        out.seek(0)
        return out

    def write(self, chars, output, format='png'):
        """Generate and write an image CAPTCHA data to the output.

        :param chars: text to be generated.
        :param output: output destination.
        :param format: image file format
        """
        im = self.generate_image(chars)
        return im.save(output, format=format)


class ImageCaptcha(_Captcha):
    """Create an image CAPTCHA.

    Many of the codes are borrowed from wheezy.captcha, with a modification
    for memory and developer friendly.

    ImageCaptcha has one built-in font, DroidSansMono, which is licensed under
    Apache License 2. You should always use your own fonts::

        captcha = ImageCaptcha(fonts=['/path/to/A.ttf', '/path/to/B.ttf'])

    You can put as many fonts as you like. But be aware of your memory, all of
    the fonts are loaded into your memory, so keep them a lot, but not too
    many.

    :param width: The width of the CAPTCHA image.
    :param height: The height of the CAPTCHA image.
    :param fonts: Fonts to be used to generate CAPTCHA images.
    :param font_sizes: Random choose a font size from this parameters.
    """
    def __init__(self, width=160, height=60, fonts=None, font_sizes=None):
        self._width = width
        self._height = height
        self._fonts = fonts or DEFAULT_FONTS
        self._font_sizes = font_sizes or (42, 50, 56)
        self._truefonts = []

    @property
    def truefonts(self):
        if self._truefonts:
            return self._truefonts
        self._truefonts = tuple([
            truetype(n, s)
            for n in self._fonts
            for s in self._font_sizes
        ])
        return self._truefonts

    @staticmethod
    def create_noise_curve(image):
        noisy_curve =  np.load('circle_noise.npy')

        a = np.array(image)
        a[np.where(noisy_curve == 28)] = 28


        return Image.fromarray(a, 'L')

    @staticmethod
    def create_noise_dots(image, color, width=3, number=30):
        draw = Draw(image)
        w, h = image.size
        while number:
            x1 = random.randint(0, w)
            y1 = random.randint(0, h)
            draw.line(((x1, y1), (x1, y1)), fill=color, width=width)
            number -= 1
        return image

    def create_captcha_image(self, chars, color, background):
        """Create the CAPTCHA image itself.

        :param chars: text to be generated.
        :param color: color of the text.
        :param background: color of the background.

        The color should be a tuple of 3 numbers, such as (0, 255, 255).
        """
        image = Image.new('L', (self._width, self._height), background)
        draw = Draw(image)


        color = (28)
        def _draw_character(c):
            font = self.truefonts[0]
            w, h = draw.textsize(c, font=font)
            im = Image.new('L', (w , h ), background)
            Draw(im).text((0, 0), c, font=font, fill=color)
            im = ImageOps.invert(im)
            im = im.rotate(random.uniform(-30, 30), Image.BICUBIC, expand=True)
            im = im.crop(im.getbbox())
            im = ImageOps.invert(im)
            im = make_28(im)

            #plt.imshow(im.convert('L'), cmap='Greys_r')
            #plt.show()
            return im

        images = []
        for c in chars:
            images.append(_draw_character(c))



        text_width = sum([im.size[0] for im in images])
        offset_init = random.randint(0, 10)
        left_over = self._width-text_width - 2*offset_init

        if left_over < 0:
            offset_init = random.randint(0, 2)
            left_over = self._width-text_width - 2*offset_init

        if left_over < 0:
            offset_init = 0
            left_over = self._width-text_width - 2*offset_init

        if left_over < 0:
            offset_init = 0
            left_over = 0        


        width = max(text_width, self._width)

        image = image.resize((width, self._height))
        offsets = randNums(len(chars), 0, int(left_over/len(chars))+1, left_over)

        offsets.insert(0, offset_init)
        
        offset = 0
        zz = 0
        height_random_offset=random.randint(-2,2)
        for im in images:
            w, h = im.size
            offset = offset+offsets[zz]
            image.paste(im, (offset, int((self._height - h) / 2 -  height_random_offset)))
            offset = offset + w 
            zz += 1


        return image

    def generate_image(self, chars):
        """Generate the image of the given characters.

        :param chars: text to be generated.
        """
        background = (255)
        color = (28)
        im = self.create_captcha_image(chars, color, background)

        dot_noise_size = random.randint(200,600)
        self.create_noise_dots(im, color, width=1, number=dot_noise_size)
        im = self.create_noise_curve(im)
        
        return im


def make_28(im):
    a = np.array(im)

    a[a <= 28] = 28

    a = a.astype( np.uint8)

    im = Image.fromarray(a, 'L')

    return im

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

# def randNums(n,a,b,s):
#     #finds n random ints in [a,b] with sum of s
#     hit = False
#     while not hit:
#         total, count = 0,0
#         nums = []
#         while total < s and count < n:
#             r = random.randint(a,b)
#             total += r
#             count += 1
#             nums.append(r)
#         if total == s and count == n: hit = True
#     return nums

def randNums(n,a,b,s):
    #finds n random ints in [a,b] with sum of s

    if s==0:
        nums = []
        for k in range(n):
            nums.append(k)
        return nums

    hit = False
    while not hit:
        total, count = 0,0
        nums = []
        while total < s and count < n:
            r = random.randint(a,b)
            total += r
            count += 1
            nums.append(r)
        if total == s and count == n: hit = True
    return nums
