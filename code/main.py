from lines_extraction import *
from models_multi import *
from utils import pred, reshape
from filter import Filter
from keras.preprocessing.sequence import pad_sequences
import cv2
import sys
import codecs
import time
import datetime
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-debug", required = False, type = bool, help = "Save blocks")
ap.add_argument("-filter", required = False, type = bool, help = "Filter blocks")

args = ap.parse_args()
DEBUG = args.debug
FILTER = args.filter

# import decoder

M = CRNN(output_dim=226)
# M.model.load_weights('new_crnn_multi.h5')
M.model.load_weights('../weights/crnn_multi.h5')
print "Loaded model!"

# dcd = decoder.BeamLMDecoder()
# dcd.load_chars('char_map.txt')
# dcd.load_lm('kenlm-model.binary')

while 1:
    im_link = raw_input("Images: ")
    if not os.path.isfile(im_link):
        print "Oops. It looks like your input is incorrect. Please try again!"
        continue
    im = cv2.imread(im_link)
    if FILTER:
        f = Filter()
        im = f.filtering(im)

    now = time.time()
    section = divider(im, DEBUG)

    with codecs.open(im_link[:-4].replace('/', '_') + "_" + str(datetime.datetime.now()).split(' ')[-1] + ".txt", "a", "utf-8") as text_file:

        for sec in section:
            batch = [reshape(sec[line[0]:line[1]+1, :]) for line in extract_lines(sec)]
            pad_lines = pad_sequences(batch, padding='post', value=255.0)
            pred(pad_lines, M, text_file, print_screen=True, return_text =False)

        text_file.close()
    print "DONE IN %fs." % (time.time() - now)