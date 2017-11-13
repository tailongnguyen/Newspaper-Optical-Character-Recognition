# -*- coding: UTF-8 -*-
from PIL import Image, ImageDraw, ImageFont
from imgaug import augmenters as iaa
from boundingDetect import fit_contours
from scipy import ndimage
#from models import *
from utils import *
from new_multi_gpu import *
from models_multi import CRNN
from keras.utils import normalize

import ntpath
import glob
import itertools
import random
import sys
import time
import cv2
import numpy
import codecs
import random
import pickle
import argparse
import numpy as np  
import os.path
from concurrent.futures import ThreadPoolExecutor, wait


np.random.seed(50)

def read_batches(words):
    # all_fonts = glob.glob("/home/long/aeh16/unicodeFonts/*")
    all_fonts = glob.glob("/data/voice/mocr/unicodeFonts/*")

    # print "Start training on %s: " % word
    gen_ims = []
    labels = []
    label_lengths=[]
    for word in words:
        word = unicode(word, 'utf-8')
        sc = len(word)/7 + 1
        for sys_font in all_fonts:
            fontSize = np.random.choice([10,20,30,40,50])
            background_color = np.random.choice([255, 230])
            text_color = np.random.randint(10)*10
            size = (int(150 * fontSize / 20.0), int(50* fontSize / 20.0))
            imgSize = (int(size[0] * sc), size[1]) if len(word) > 7 else size

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
            draw.text(position, word, (text_color,text_color,text_color), font=font)
            file_name = './temp.png'
            # print file_name
            image.save(file_name)
            im = ndimage.imread(file_name)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            _, gray = cv2.threshold(gray,175,255,cv2.THRESH_BINARY)
            b, l, t, r = fit_contours(gray)
            im_1 = im[t:b+1, l:r+1]
            scale = 32.0/ im_1.shape[0]
            im_1 = cv2.resize(im_1, (int(im_1.shape[1] * scale), 32) )
            gen_ims.append(im_1)

            i = np.random.randint(3)
            noise = iaa.AdditiveGaussianNoise(loc=0, scale=i*10+10, per_channel=i/10.0)
            noisy = noise.augment_image(im_1)
            gen_ims.append(noisy)
            labels.extend(text_to_labels(word) * 2)
            try:
                label_lengths.extend([len(text_to_labels(word))]*2)
            except KeyError:
                yield None

    patches = [reshape(img) for img in gen_ims]
    patches = pad_sequences(patches, padding='post', value=255.0)
    assert patches.shape[0] == len(label_lengths)
    try:
        yield [ np.asarray(patches), \
                patches[0].shape[0],
                np.asarray(labels), \
                np.asarray([patches[0].shape[0]//4 for p in patches]), \
                np.asarray(label_lengths),
                ]
    except KeyError as e:
        yield None

def new_read_batch(word_list):
    pool = ThreadPoolExecutor(1)  # Run a single I/O thread in parallel
    future = pool.submit(read_batches, word_list[0:5])
    for i in range(1, len(word_list)/5):
        wait([future])
        minibatch = future.result()
#         print minibatch
        # While the current minibatch is being consumed, prepare the next
        future = pool.submit(read_batches, word_list[i*5:(i+1)*5])
        yield minibatch
        
    # Wait on the last minibatch
    wait([future])
    minibatch = future.result()
    yield minibatch

def train(n_epoch, learn_rate, report_steps, words_file, initial_weights=None):

    M = CRNN(learn_rate, 226, int(sys.argv[3]))
    M.compile_model()
    
    if os.path.isfile('crnn_multi.h5') and initial_weights != None:
        M.model.load_weights(initial_weights)
        print "Loaded weights!"

    # M.model.summary()

    words_list = [ str(l).rstrip('\r\n') for l in open(words_file, 'r').readlines()]
    np.random.shuffle(words_list)
    number_of_batches = len(words_list)/5

    with open("logging.txt", "a") as text_file:
        for ep in range(n_epoch):
            t = time.time()
            batch_loss = []
            for num_w, big_batch in enumerate(new_read_batch(words_list)):
                for batch in big_batch:
                    try:
                        if batch != None:
                            loss, _  = M.train_step([batch[0], batch[1], batch[2], batch[3], batch[4], True])
                            batch_loss.append(loss)

                    except KeyboardInterrupt:
                        M.model.save_weights('crnn_multi.h5')
                        return 
                        
                if num_w % report_steps == 0:
                    print("%d/%d: %f\n"  % (num_w, number_of_batches, calculate_mean(batch_loss)))
                    print "ETA %d/%d : %f seconds\n" % (num_w, number_of_batches, time.time() - t)
                    text_file.write("%d/%d: %f\n"  % (num_w, number_of_batches, calculate_mean(batch_loss)))
                    text_file.write("ETA %d/%d : %f seconds\n" % (num_w, number_of_batches, time.time() - t))
                    t = time.time()
                    batch_loss = []

                if num_w % 1000 == 0:
                    M.model.save_weights('crnn_multi.h5')
                            
            np.random.shuffle(words_list)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        initial_weights = sys.argv[2]
    else:
        initial_weights = None

    train(n_epoch = 2,
          learn_rate=0.0001,
          report_steps=50,
          words_file = sys.argv[1],
          initial_weights=initial_weights)

