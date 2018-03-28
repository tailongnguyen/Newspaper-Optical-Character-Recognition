import numpy as np 

def project(im, orientation=0):
	''' orientation = 0 for lines 1 for columns '''
	im = 255-im
	if orientation == 1:
		x = [sum(im[:,i]) for i in range(im.shape[1])]
	else:
		x = [sum(im[i,:]) for i in range(im.shape[0])]
	return x
