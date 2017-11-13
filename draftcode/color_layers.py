import color_quantization as CQ 
import numpy as np 
import cv2

def textLayers(quant, nontext):
	clrs = []
	for i in range(quant.shape[0]):
		for j in range(quant.shape[1]):
			if not np.any(clrs == quant[i][j]):
				clrs.append(quant[i][j])
	# return len(clrs)
	idx = 0
	new_img = quant.copy()
	final_quant = None
	max_number_pix = 0
	for clr in clrs:
		if np.any(nontext == clr):
			continue
		for i in range(new_img.shape[0]):
			for j in range(new_img.shape[1]):
				new_img[i][j] = np.array([255]*3 if (quant[i][j] == clr).all() else [0]*3)
		# idx += 1
		# print(idx)
		img = new_img[:,:,0].copy()

		kernel	=	np.ones((1,new_img.shape[0]),	np.uint8) * 255
		img	=	cv2.dilate(img,	kernel,	iterations=1)
		# img	=	cv2.erode(img,	kernel,	iterations=1)

		cnt = 0
		for i in range(new_img.shape[0]):
			for j in range(new_img.shape[1]):
				if img[i][j] == 255:
					cnt += 1
		if cnt > max_number_pix:
			max_number_pix = cnt
			final_quant = clr
		# cv2.imshow("xxx", np.hstack([255-new_img[:,:,0], 255-img]))
		# cv2.waitKey()
	return final_quant