from keras.models import Sequential
from keras.layers import Dense, Activation
from otsu_thresh import otsu
from projection import project
import cv2
import sys
import matplotlib.pyplot as plt 
import numpy as np 
import keras 

class Filter():
    def __init__(self):
        self.model = self.get_filter()

    def classify(self, im):
        ft = self.get_features(im)
        ft = np.asarray(ft)
        ft = ft.reshape(1, ft.shape[0])
        pred = self.model.predict(ft)[0]
        if np.argmax(pred) == 1:
            return "Text"
        else:
            return "Image"

    def get_features(self, im):
        hist = cv2.calcHist([im],[0],None,[255],[0,255])
        norm_hist = hist/np.max(hist)
        var = sum(norm_hist * (np.arange(len(norm_hist)).reshape(-1, 1)))[0]

        if len(im.shape) > 2:
            im = otsu(im)
        projection = project(im, 0)
        num = 0 if projection[0] == 0 else 1
        for i, p in enumerate(projection):
            if p > 0 and projection[i-1] == 0:
                num += 1
        return var/1000, num

    def get_filter(self):
        model = Sequential()
        model.add(Dense(32, input_shape=(2,)))
        model.add(Activation('relu'))
        model.add(Dense(2))
        model.add(Activation('sigmoid'))
        model.load_weights('block_filter.h5')
        return model

    def filtering(self, im):
        thresh = 255 - otsu(im)
        kernel = np.ones((10,10),np.uint8)
        dilation = cv2.dilate(thresh,kernel,iterations = 1)
        _, cnts, hierachy = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.1 * peri, True)
            x,y,w,h = cv2.boundingRect(approx)
            if w*h > 40000:
                temp = im[y:y+h, x:x+w]
                if self.classify(temp) == "Image":
                    im[y:y+h,x:x+w] = 255
        return im

if __name__ == "__main__":
    f = Filter()
    im = cv2.imread(sys.argv[1])
    thresh = otsu(im)
    thresh = 255 - thresh

    kernel = np.ones((10,10),np.uint8)
    dilation = cv2.dilate(thresh,kernel,iterations = 1)
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # dilation = 255 - dilation
    plt.imshow(dilation, cmap='gray')
    plt.show()
    img = im.copy()
    _, cnts, hierachy = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * peri, True)
        x,y,w,h = cv2.boundingRect(approx)
        if w*h > 40000:
            im = cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            # temp = img[y:y+h, x:x+w]
            # cv2.imshow("", temp)
            # cv2.waitKey()
            # print f.get_features(temp)
            # if f.classify(temp) == "Image":
            #     im[y:y+h,x:x+w] = 255
                
    plt.figure(1)
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(otsu(im))
    plt.show()
        