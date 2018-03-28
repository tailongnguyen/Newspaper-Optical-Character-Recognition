# -*- coding: UTF-8 -*-
from PIL import Image, ImageDraw, ImageFont
from imgaug import augmenters as iaa
from boundingDetect import fit_contours
from scipy import ndimage
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

def read_batches(texts):
    all_fonts = glob.glob("/data/voice/mocr/unicodeFonts/*")

    # print "Start training on %s: " % text
    gen_ims = []
    labels = []
    label_lengths=[]
    for text in texts:
        text = unicode(text, 'utf-8')
        sc = len(text)/7 + 1
        for sys_font in all_fonts:

            # Randomly choose a fontSize
            fontSize = np.random.choice([10, 20, 30, 40, 50])
            background_color = np.random.choice([255, 230])
            text_color = np.random.randint(10)*10
            size = (int(150 * fontSize / 20.0), int(50* fontSize / 20.0))
            imgSize = (int(size[0] * sc), size[1]) if len(text) > 7 else size

            # Load font
            font_file = ntpath.basename(sys_font)
            font_file = font_file.rsplit('.')
            font_file = font_file[0]

            # Draw the text to image
            path = sys_font
            font = ImageFont.truetype(path, fontSize)
            image = Image.new("RGB", imgSize, (background_color, background_color, background_color))
            draw = ImageDraw.Draw(image)
            pos_x = 0
            pos_y = 10
            position = (pos_x, pos_y)
            draw.text(position, text, (text_color, text_color, text_color), font=font)

            # Convert image to numpy array and then transform to get height of 32
            im = np.array(image)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            _, gray = cv2.threshold(gray,175,255,cv2.THRESH_BINARY)
            b, l, t, r = fit_contours(gray)
            im_1 = im[t:b+1, l:r+1]
            scale = 32.0/ im_1.shape[0]
            im_1 = cv2.resize(im_1, (int(im_1.shape[1] * scale), 32) )
            gen_ims.append(im_1)

            # Randomly adding noise
            i = np.random.randint(3)
            noise = iaa.AdditiveGaussianNoise(loc=0, scale=i*10+10, per_channel=i/10.0)
            noisy = noise.augment_image(im_1)
            gen_ims.append(noisy)
            labels.extend(text_to_labels(text) * 2)
            try:
                label_lengths.extend([len(text_to_labels(text))]*2)
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

def new_read_batch(text_list, batch_size = 5):
    pool = ThreadPoolExecutor(1)  # Run a single I/O thread in parallel
    future = pool.submit(read_batches, text_list[0:batch_size])
    for i in range(1, len(text_list)/batch_size + 1):
        wait([future])
        minibatch = future.result()
        # While the current minibatch is being consumed, prepare the next
        future = pool.submit(read_batches, text_list[i*batch_size:(i+1)*batch_size])
        yield minibatch
        
    # Wait for the last minibatch
    wait([future])
    minibatch = future.result()
    yield minibatch

def train(n_epoch, learn_rate, report_steps, texts_file, batch_size, initial_weights=None):

    M = CRNN(learn_rate, 226, int(sys.argv[3]))
    M.compile_model()
    
    if os.path.isfile('crnn_multi.h5') and initial_weights != None:
        M.model.load_weights(initial_weights)
        print "Loaded weights!"

    # M.model.summary()

    texts_list = [str(l).rstrip('\r\n') for l in open(texts_file, 'r').readlines()]
    np.random.shuffle(texts_list)
    number_of_batches = len(texts_list)/batch_size

    for ep in range(n_epoch):
        batch_loss = []
        for num_w, big_batch in enumerate(new_read_batch(texts_list, batch_size)):
            for batch in big_batch:
                try:
                    if batch != None:
                        loss, _  = M.train_step([batch[0], batch[1], batch[2], batch[3], batch[4], True])
                        batch_loss.append(loss)

                except KeyboardInterrupt:
                    M.model.save_weights('crnn_multi.h5')
                    return 
                    
            if num_w % report_steps == 0:
                sys.stdout.write("\r%d/%d: %f\n"  % (num_w, number_of_batches, calculate_mean(batch_loss)))
                t = time.time()
                batch_loss = []

            if num_w % 1000 == 0:
                M.model.save_weights('crnn_multi.h5')
                        
        np.random.shuffle(texts_list)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        initial_weights = sys.argv[2]
    else:
        initial_weights = None

    train(n_epoch = 2,
          learn_rate=0.0001,
          report_steps=50,
          texts_file = sys.argv[1],
          batch_size = 5,
          initial_weights=initial_weights)

