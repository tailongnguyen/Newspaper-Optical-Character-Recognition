# -*- coding: UTF-8 -*-
from scipy import ndimage
from scipy.misc import imresize
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model, Model
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, Lambda, Input
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Bidirectional
from keras import backend as K
from otsu_thresh import otsu

from boundingDetect import fit_contours
import codecs
import numpy as np 
import os
import cv2

np.random.seed(50)
s = """0123456789abcdefghijklmnopqrstuvwxyzàáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹABCDEFGHIJKLMNOPQRSTUVWXYZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ\%/.~,:-+()=><; "!?@*'[]{}^$#"""
forward_mapping = {}
backward_mapping = {}
for i, c in enumerate(s):
	forward_mapping[c] = i+1
	backward_mapping[i+1] = c

def get_dataset(root):
	words = os.listdir(root)
	words_dir = [os.path.join(root, f) for f in words]
	samples = []
	for w in words_dir:
		samples.extend([os.path.join(w, f) for f in os.listdir(w)])

	labels = []
	for i, w in enumerate(words):
		if "slash" in w:
			idx = w.index("slash")
			temp = list(w)
			del temp[idx+1:idx+5] 
			temp[idx] = '/'
			w = ''.join(temp)
		labels.extend([w]* len(os.listdir(words_dir[i])))
	labels = [text_to_labels(l, forward_mapping) for l in labels]
	dataset = zip(samples, labels)
	print ("%d training samples" % len(samples))
	return dataset

def text_to_labels(text, mapping=forward_mapping):
    return [mapping[char] for char in text]

def labels_to_text(labels, mapping=backward_mapping):
	ret = [mapping[l] for l in labels if l != 0]
	return ''.join(ret)

def calculate_mean(l):
	if len(l) == 0:
		return 0
	return (float)(sum(l))/(len(l))

def reshape(im):
	if im.shape[0] != 32:
		thresh = otsu(im)
		b, l, t, r = fit_contours(thresh)
		im = im[t:b+1, l:r+1]
		scale = 32.0/ im.shape[0]
		try:
			im = imresize(im, (32, int(im.shape[1] * scale)))
		except ValueError:
			return None
		
	if len(im.shape) == 3:
		im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY)

	im = im.reshape(im.shape[0], im.shape[1], 1)
	im = np.transpose(im, (1,0,2)) 
	return im

def argmax_decode(prediction):
	tokens = []
	c_prev = -1
	for c in prediction:
		if c == c_prev:
			continue
		if c != 0:  # Blank
			tokens.append(c)
		c_prev = c
	return tokens
	
def pred(pad_lines, model, text_file, print_screen = False, return_text = False):
	# print pad_lines.shape
	p = model.predict_step([pad_lines, pad_lines.shape[1], True])[0]
	p = ' '.join([labels_to_text(argmax_decode(p_)) for p_ in p])
	if print_screen:
		print (p)
	if text_file != None:
		text_file.write("%s " % p)
	if return_text:
		return p
