import numpy as np
from scipy.interpolate import interp1d

def calc_cost(img):
	m, n = img.shape
	arr = np.sum(img, axis=1)
	idx = np.linspace(0, m, m, endpoint=False)
	f = interp1d(idx, arr, kind='cubic')
	new_idx = np.linspace(0, m, m/8, endpoint=False)
	new_arr = f(new_idx)
	n_arr = new_arr.shape[0]
	local_max = 0.0
	local_min = 0.0
	for i in range(1, n_arr - 1):
		if (new_arr[i] > new_arr[i-1] and new_arr[i] > new_arr[i+1]):
			local_max = local_max + new_arr[i]
		if (new_arr[i] < new_arr[i-1] and new_arr[i] < new_arr[i+1]):
			local_min = local_min + new_arr[i]

	return local_max - local_min