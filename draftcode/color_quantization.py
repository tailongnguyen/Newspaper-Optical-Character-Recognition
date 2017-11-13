from sklearn.cluster import MiniBatchKMeans
import cv2
import numpy as np 

	
def quantize(n_clusters, image):
	"""
		Input: initial image
		Output: image with # (= # of clusters) colors 
	"""
	(h, w) = image.shape[:2]
	 
	image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
	 
	image = image.reshape((image.shape[0] * image.shape[1], 3))
	 
	clt = MiniBatchKMeans(n_clusters)
	labels = clt.fit_predict(image)
	quant = clt.cluster_centers_.astype("uint8")[labels]
	 
	quant = quant.reshape((h, w, 3))
	image = image.reshape((h, w, 3))
	 
	quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
	image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)


	return quant