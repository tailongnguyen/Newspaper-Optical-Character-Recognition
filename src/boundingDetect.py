
def fit_contours(im):
	"Find the contour fit to the character in a binary image"
	def top(x, y, im):
		for i in range(im.shape[0]):
			for j in range(x, y+1):
				if im[i][j] == 0:
					return i
		return 0

	def bottom(x, y, im):
		for i in range(im.shape[0]):
			for j in range(x, y+1):
				if im[im.shape[0]-1-i][j] == 0:
					return im.shape[0]-1-i
		return im.shape[0]-1

	def leftMost(im):
		for i in range(im.shape[1]):
			for j in range(im.shape[0]):
				if im[j][i] == 0:
					return i
		return 0

	def rightMost(im):
		for i in range(im.shape[1]-1, -1, -1):
			for j in range(0, im.shape[0]):
				if im[j][i] == 0:
					return i
		return im.shape[1]-1
	l, r = leftMost(im), rightMost(im)
	t, b = top(l, r, im), bottom(l, r, im)
	if t > 0:
		return b, l, t-1, r 
	return b, l, t, r