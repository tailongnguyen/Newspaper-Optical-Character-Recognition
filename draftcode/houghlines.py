import cv2
import numpy as np
import sys 
from otsu_thresh import otsu
import matplotlib.pyplot as plt
img = cv2.imread(sys.argv[1])
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray,50,150,apertureSize = 3)
edges = otsu(img)
edges = 255 - edges
# plt.imshow(edges, cmap='gray')
# plt.show()
lines = cv2.HoughLines(edges,1,np.pi/180, 1000)
print lines.shape

vertical = [0.0, 0.2]
horizontal = [np.pi/2 - 0.1, np.pi/2 + 0.1]
for t in lines:
    # print t
    rho, theta = t[0][0], t[0][1]
    if not ( (theta >= vertical[0] and theta <= vertical[1]) or (theta >= horizontal[0] and theta <= horizontal[1])):
        continue
    print rho, theta

    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 5000*(-b))
    y1 = int(y0 + 5000*(a))
    x2 = int(x0 - 5000*(-b))
    y2 = int(y0 - 5000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),5)
# cv2.imwrite('houghlines3.jpg',img)
plt.imshow(img)
plt.show()
