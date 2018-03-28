from otsu_thresh import otsu
from scipy import ndimage
from filter import Filter
from dfs import get_CCs
import cv2
import numpy as np
import collections
import glob
import os
import sys
import matplotlib.pyplot as plt 
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-t", required = False, type = int, help = "Save blocks")
ap.add_argument("-i", required = False, type = str, help = "Save blocks")

args = ap.parse_args()
INF = 10000000
SPACE_THRESHOLD = 2
LINE_THRESHOLD = 5


def project(im, orientation=0):
	''' orientation = 0 for lines 1 for columns '''
	im = 255-im
	if orientation == 1:
		x = [sum(im[:,i]) for i in range(im.shape[1])]
	else:
		x = [sum(im[i,:]) for i in range(im.shape[0])]
	return x

def extract_lines(im):
	''' extract lines from a plain paragraph '''
	image = otsu(im)
	lines = dict()
	projection = project(image, 0)
	prev = -1
	for i, p in enumerate(projection):
		if projection[i] > 0:
			if prev == -1:
				lines[i] = [i]
				prev = i
			else:
				lines[prev].append(i)
		else:
			prev = -1

	lines = collections.OrderedDict(sorted(lines.items()))
	lines = [v for (k, v) in lines.items()]

	# for i, v in enumerate(lines):
	# 	print v
	
	# All gaps that are smaller than a threshold should be merged to the following line
	if len(lines) == 0:
		yield None
	else:
		for i, v in enumerate(lines):
			if i >  0 and v[0] - lines[i-1][-1] < SPACE_THRESHOLD:
				lines[i] = lines[i-1] + v
				lines[i-1] = []	
	lines = [l for l in lines if len(l) > 0]

	# All lines that are smaller than a threshold should be merged to the nearest line
	candidates = [ 1 if len(v) >= LINE_THRESHOLD else 0 for v in lines ]
	for i, v in enumerate(lines):
		v.sort()
		if len(v) < LINE_THRESHOLD:
			upper, lower = INF, INF
			if i == 0:
				upper = INF
			else:
				for j in range(i-1, -1, -1):
					if candidates[j] == 1:
						upper = j
						break
			for j in range(i+1, len(lines)):
				if candidates[j] == 1:
					lower = j
					break

			try:
				dist_upper = v[0] - lines[upper][-1]
				# print "upper: ", upper, lines[upper], lines[upper][1]
			except IndexError as e:
				dist_upper = INF
			try:
				dist_lower = -v[-1] + lines[lower][0]
				# print "lower: ", lower, lines[lower], lines[lower][1]
			except IndexError as e:
				dist_lower = INF
			try:
				if dist_upper < dist_lower:
					if dist_upper < lines[upper][-1] - lines[upper][0] + 1:
						# print "to upper"
						lines[upper].extend(v)
				else:
					if dist_lower < lines[lower][-1] - lines[lower][0] + 1:
						# print "to lower"
						lines[lower] = v + lines[lower]
			except IndexError as e:
				yield 0, im.shape[0]-1
	
	# for i, v in enumerate(lines):
		# print v

	for i, v in enumerate(lines):
		if len(v) >= LINE_THRESHOLD:
			yield v[0], v[-1]


def divide(im, orientation, block_threshold = 10):
	''' orientation = 0 for horizontal, 1 for vertical '''
	image = otsu(im)
	lines = dict()
	projection = project(image, orientation)
	# # show(im)
	# plt.bar(np.arange(image.shape[orientation]), projection)
	# plt.show()
	prev = -1
	for i, p in enumerate(projection):
		if projection[i] == 0:
			if prev == -1:
				lines[i] = [i]
				prev = i
			else:
				lines[prev].append(i)
		else:
			prev = -1
	lines = collections.OrderedDict(sorted(lines.items()))
	lines = [v for (k, v) in lines.items() if len(v) > block_threshold]
	candidates = [ 1 if len(v) >= 10 else 0 for v in lines ]

	# for i, v in enumerate(lines):
	# 	print v, len(v)

	if len(lines) == 0:
		return [im]

	result = []
	if lines[0][0] > 0:
		result.append([0, lines[0][0]])

	for i, v in enumerate(lines):
		if i == 0:
			continue
		result.append([lines[i-1][-1], v[0]])

	if lines[-1][-1] < im.shape[orientation]-1:
		result.append([lines[-1][-1], im.shape[orientation]-1])	

	# print result
	if orientation == 0:
		result = [im[r[0]:r[1]+1, :] for r in result if r[1]+1-r[0] > 5]
	else:
		result = [im[:, r[0]:r[1]+1] for r in result if r[1]+1-r[0] > 5]
	return result

def divider(im, debug = False, block_threshold = 10):
	thresh = otsu(im)
	remove = []
	for left, top, right, bot, nodes in get_CCs(thresh):
		height = bot - top + 1
		width = right - left + 1
		if (float)(height) / width > 100 or (float)(width) / height > 100 \
						or (float)(height * width)/len(nodes) > 20:
			remove.extend(nodes)
	for pixel in remove:
		im[pixel[0]][pixel[1]] = 255

	# f = Filter()
	# im = f.filtering(im)
	blocks = divide(im, 1, block_threshold)
	prev_len = 0
	order = 0
	loop = 0
	while len(blocks) > prev_len:
		prev_len = len(blocks)
		temp = []
		for i, l in enumerate(blocks):
			temp.extend(divide(blocks[i], order, block_threshold))
		blocks = temp
		order = (order+1)%2
		loop +=1
		
	if debug:
		files = glob.glob('./ccs/*')
		for f in files:
			os.remove(f)
		for i, l in enumerate(blocks):
			cv2.imwrite('./ccs/' + str(i) + ".png", l)
		
	return blocks

def show(im):
	cv2.imshow("", im)
	cv2.waitKey()

if __name__ == "__main__":
	im = ndimage.imread(args.i)
	section = divider(im, True, args.t)
	# show(im)
	for i, sec in enumerate(section):
			# show(sec)
			cv2.imwrite('./sections/' + str(i) + ".png", sec)
	# for sec in section:
		# for line in extract_lines(sec):
			# cv2.imshow(" ", sec[line[0]:line[1]+1, :])
			# cv2.waitKey()