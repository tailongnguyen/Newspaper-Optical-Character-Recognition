import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os
import keras 
from keras.models import Sequential
from keras.layers import Dense, Activation
from projection import project
from otsu_thresh import otsu

def get_data(folder):
    data = os.listdir(folder)
    data = [folder + "/" + d for d in data]
    return data

# def get_features(im):
#     hist = cv2.calcHist([im],[0],None,[255],[0,255])
#     norm_hist = hist/np.max(hist)
#     var = sum(norm_hist * (np.arange(len(norm_hist)).reshape(-1, 1)))[0]

#     if len(im.shape) > 2:
# 		im = otsu(im)
#     projection = project(im, 0)
#     num = 0 if projection[0] == 0 else 1
#     for i, p in enumerate(projection):
#         if p > 0 and projection[i-1] == 0:
#             num += 1
#     return var, num


# imgs = get_data('./newspaper/images')
# txts = get_data('./newspaper/text')
# data = imgs + txts
# X = np.asarray([get_features(cv2.imread(im)) for im in data])
# X = X.reshape((X.shape[0], X.shape[1]))
# Y = np.asarray([0 for i in range(len(imgs))] + [1 for i in range(len(txts))])
# Y = keras.utils.to_categorical(Y)

# # plt.figure(1)
# # plt.subplot(221)
# # plt.imshow(cv2.imread(data[0]), cmap='gray')
# # plt.subplot(222)
# # plt.bar(np.arange(255), X[0], width=1.0)
# # plt.text(112, 0.5, Y[0],fontdict=None, withdash=True)
# # plt.subplot(223)
# # plt.imshow(cv2.imread(data[1]), cmap='gray')
# # plt.subplot(224)
# # plt.bar(np.arange(255), X[1], width=1.0)
# # plt.text(112, 0.5, Y[1],fontdict=None, withdash=True)
# # plt.show()
# print X.shape, Y.shape

# model = Sequential()

# model.add(Dense(32, input_shape=(2,)))
# model.add(Activation('relu'))
# model.add(Dense(2))
# model.add(Activation('sigmoid'))
# model.summary()
# model.compile(optimizer = keras.optimizers.RMSprop(),
#              loss = keras.losses.binary_crossentropy, metrics=['accuracy'])

# model.fit(X[:-100], Y[:-100], batch_size=1, epochs=5, validation_data=(X[-100:], Y[-100:]))

# model.save_weights('block_filter.h5')

def get_features_1(im):
    hist = cv2.calcHist([im],[0],None,[255],[0,255])
    norm_hist = hist/np.max(hist)
    var = sum((norm_hist - 0.5)**2) 
    return var

# def get_features_2(im):
#     if len(im.shape) > 2:
# 		im = otsu(im)
#     projection = project(im, 0)
#     num = 0 if projection[0] == 0 else 1
#     for i, p in enumerate(projection):
#         if p > 0 and projection[i-1] == 0:
#             num += 1
#     return num
# def get_features_3(im):
#     return im.shape[0]*im.shape[1]


imgs = get_data('./newspaper/images')
txts = get_data('./newspaper/text')
img_fts_1 = [get_features_1(cv2.imread(i, 0))  for i in imgs]
# img_fts_2 = [get_features_3(cv2.imread(i, 0)) for i in imgs]
txt_fts_1 = [get_features_1(cv2.imread(i, 0))  for i in txts]
# txt_fts_2 = [get_features_3(cv2.imread(i, 0)) for i in txts]


plt.scatter(np.arange(len(img_fts_1)), img_fts_1, s=20, c='red', marker='o')
plt.scatter(np.arange(len(txt_fts_1)), txt_fts_1, s=20, c='green', marker='o')

plt.show()




# link1 = sys.argv[1]
# link2 = sys.argv[2]
# img = cv2.imread(link1, 0)
# img1 = cv2.imread(link2, 0)

# # img = cv2.resize(img, SHAPE)
# # img1 = cv2.resize(img1, SHAPE)
# hist = cv2.calcHist([img],[0],None,[255],[0,255])
# hist1 = cv2.calcHist([img1],[0],None,[255],[0,255])
# norm_hist = hist/np.max(hist)
# norm_hist1 = hist1/np.max(hist1)

# var = sum((norm_hist - 0.5)**2) 
# var1 = sum((norm_hist1 - 0.5)**2) 


# plt.figure(1)
# plt.subplot(221)
# plt.imshow(img, cmap='gray')
# plt.subplot(222)
# plt.bar(np.arange(255), norm_hist, width=1.0)
# plt.text(112, 0.5, var,fontdict=None, withdash=True)
# plt.annotate(var, xy=(0.5, 112), xytext=(0, 0),
#     arrowprops=dict(arrowstyle="->"))
# plt.subplot(223)
# plt.imshow(img1, cmap='gray')
# plt.subplot(224)
# plt.bar(np.arange(255), norm_hist1, width=1.0)
# plt.text(112, 0.5, var1,fontdict=None, withdash=True)
# plt.show()


