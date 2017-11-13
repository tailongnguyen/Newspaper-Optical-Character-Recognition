from dfs import get_CCs
from scipy import ndimage
from otsu_thresh import otsu
import matplotlib.pyplot as plt
import cv2
import sys
import numpy as np

def show(im):
    cv2.imshow('', im)
    cv2.waitKey()

im = ndimage.imread(sys.argv[1])
thresh = otsu(im)
# show(im)
remove = []
l1, l2, l3 = [], [] ,[]
for left, top, right, bot, nodes in get_CCs(thresh):
    height = bot - top + 1
    width = right - left + 1
    # l1.append((float)(height) / width)
    # l2.append((float)(width) / height)
    # l3.append((float)(height * width)/len(nodes))
    if (float)(height) / width > 100 or (float)(width) / height > 100 \
    or (float)(height * width)/len(nodes) > 20:
        # thresh[top:bot+1, left:right+1] = 255
        print left, top, right, bot
        remove.extend(nodes)
# print max(l1), max(l2), max(l3)
print len(remove)        
for pixel in remove:
    thresh[pixel[0]][pixel[1]] = 255
# show(im)
cv2.imwrite('output.png', thresh)
plt.imshow(thresh, cmap='gray')
plt.show()



