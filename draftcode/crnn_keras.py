from keras.models import load_model, Model
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, Lambda, Input
from keras.layers import LSTM, TimeDistributed, Bidirectional
from models import *
from keras import backend as K
from utils import *
import codecs
import random
import keras
import pickle
import argparse
import numpy as np  
import os.path

ap = argparse.ArgumentParser()
ap.add_argument("-ba", required = True, type = int, help = "Batch Size")
ap.add_argument("-ep", required = True, type = int, help = "Number of epochs")
ap.add_argument("-lw", required = True, type = int, help = "Load weights")
ap.add_argument("-lr", required = True, type = float, help = "Learning rate")
ap.add_argument("-ro", required = True, type = str, help = "Train folder")

args = ap.parse_args()

root = args.ro
dataset = get_dataset(root)
size = len(dataset)

batch_size = args.ba
epochs = args.ep
learning_rate = args.lr


M = CRNN(learning_rate, 213)
M.model.summary()
if os.path.isfile('crnn.h5') and args.lw == 1:
    M.model.load_weights('crnn.h5')
    print "Loaded weights!"
with codecs.open("logging.txt", "a", "utf-8") as text_file:
    for ep in range(epochs):
        loss_batch = []
        idx = 0
        for batch in get_batches_crnn(dataset, batch_size = batch_size):
            loss, _  = M.train_step([batch[0], batch[1], batch[2], batch[3], batch[4], True])
            # print batch[0].shape, batch[1], batch[2], batch[3], batch[4], loss
            loss_batch.append(loss)
            text_file.write("%d/%d: %f\n" % (idx, size, loss))
            print "%d/%d %f" %(idx, size, loss)
            idx += batch[0].shape[0]
            if idx % 10000 == 0:
                predict(10, num_classes, M, dataset = dataset)
                M.model.save_weights('crnn.h5')

        text_file.write("Epoch %d: %f\n" % (ep, calculate_mean(loss_batch)))
        print "Epoch %d: %f" % (ep, calculate_mean(loss_batch))
        M.model.save_weights('crnn.h5')



