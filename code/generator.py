# -*- coding: UTF-8 -*-
from PIL import Image, ImageDraw, ImageFont
from imgaug import augmenters as iaa
from boundingDetect import fit_contours
from scipy import ndimage
from otsu_thresh import otsu
from scipy.misc import imresize
import ttfquery.findsystem 
import ntpath
import numpy as np
import os
import glob
import sys
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-textfile", required = True, type = str, help = "Text file")
ap.add_argument("-fonts", required = True, type = str, help = "Fonts folder")
ap.add_argument("-saveto", required = False, type = str, help = "Save to")
args = ap.parse_args()

def reshape(im):
	if im.shape[0] != 32:
		if len(im.shape) >= 3:
			thresh = otsu(im)
			im = togray(im)
		else:
			thresh = im
		b, l, t, r = fit_contours(thresh)
		im = im[t:b+1, l:r+1]
		scale = (float)(32 )/ im.shape[0]
		im = imresize(im, (32, int(im.shape[1] * scale)))
	if len(im.shape) == 3:
		im = togray(im)
	im = im.reshape(im.shape[0], im.shape[1], 1)
	# im = np.transpose(im, (1,0,2)) 

	return im

def togray(rgb):
	# r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
	# gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
	# return gray
	return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2GRAY)

def paint_words(words_file, folder, font_fol):


	if not os.path.isdir(folder):
		os.mkdir(folder)

	position = (0,0)

	# contrast1 = iaa.ContrastNormalization((0.45, 0.5), per_channel=0.5)
	# contrast2 = iaa.ContrastNormalization((0.65, 0.85), per_channel=0.5)
	# add = iaa.Add(-40, per_channel=0.5)
	# drop = iaa.CoarseDropout(0.1, size_percent=0.5, per_channel=0.2)

	words_list = [ str(l).rstrip('\r\n') for l in open(words_file, 'r').readlines()]
	all_fonts = glob.glob(font_fol + "/*")
	# print all_fonts
	total = 0
	for w in words_list: 
		print len(unicode(w, 'utf-8'))
		sc = len(unicode(w, 'utf-8'))/7 + 1
		save = folder + "/"+ w
		if os.path.exists(save):
			print "Skiping ", w
			continue
		os.makedirs(save)
		print "Start processing %s: " % w
		for sys_font in all_fonts:

			fontSize = np.random.choice([10,20,30])
			background_color = np.random.choice([255, 200])
			text_color = np.random.randint(10)*10
			size = (int(150 * fontSize / 20.0), int(50* fontSize / 20.0))
			imgSize = (int(size[0]*sc), size[1])

			font_file = ntpath.basename(sys_font)
			font_file = font_file.rsplit('.')
			font_file = font_file[0]
			#weck desired font
			path = sys_font
			font = ImageFont.truetype(path, fontSize)
			image = Image.new("RGB", imgSize, (background_color,background_color,background_color))
			draw = ImageDraw.Draw(image)
			pos_x = 0
			pos_y = 10
			position = (pos_x,pos_y)
			draw.text(position, w.decode('utf-8'), (text_color,text_color,text_color), font=font)
			file_name = font_file +  '_' + str(0) + '_' + str(0) + '.png'
			file_name = os.path.join(save,file_name)
			# print file_name
			image.save(file_name)
			im = ndimage.imread(file_name)
			gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
			_, gray = cv2.threshold(gray,175,255,cv2.THRESH_BINARY)
			b, l, t, r = fit_contours(gray)
			im_1 = im[t:b+1, l:r+1]
			scale = 32.0/ im_1.shape[0]
			im_1 = cv2.resize(im_1, (int(im_1.shape[1] * scale), 32) )
			cv2.imwrite(save + "/" + font_file + str(1) + ".png", togray(im_1))

			for i in range(3):
				noise = iaa.AdditiveGaussianNoise(loc=0, scale=i*10+10, per_channel=i/10.0)
				c1 = noise.augment_image(im_1)
				
				cv2.imwrite(save + "/" + font_file + "_" + str(i) + '_' + str(1) + ".png", togray(c1))
				# cv2.imwrite(save + "/" + font_file + "_" + str(i) + '_' + str(2) + ".png", togray(c2))

			os.remove(file_name)
			total += 3
			
	print "Done with %d samples!" % total

curDir = os.getcwd()
if args.saveto == "cur" :
	paint_words(curDir + "/" + args.textfile, curDir, curDir + "/" + args.fonts )
else:
	paint_words(curDir + "/" + args.textfile, curDir + "/" + args.saveto, curDir + "/" + args.fonts )
