
import keras.layers as L
from keras.layers.core import Lambda
from keras import backend as K
from keras.layers import merge, Concatenate,Flatten

import tensorflow as tf
from keras.models import Model,Sequential
	
def make_parallel(model, gpu_count):
	""" Allows a model to run on multiple GPUs.
	"""
	def slice_batch(x, n_gpus, part):
		""" Divide the input batch into [n_gpus] slices,
			and obtain slice no. [part].
			i.e. if len(x) = 10, then slice_batch(x, 2, 1)
			will return x[5:].
		"""
		sh = K.shape(x)
		L = sh[0] // n_gpus # sh[0] = batch_size

		if part == n_gpus - 1:
			return x[part*L:]

		return x[part*L:(part+1)*L]

	all_outputs = []

	# Empty list for each output in the model
	for i in range(len(model.outputs)):
		all_outputs.append([])

	# Place a copy of the model on each GPU, 
	# each getting a slice of the batch
	for i in range(gpu_count):
    		
		with tf.device('/cpu:0'):
			slices = []  # multi-input case
			for x in model.inputs:
				input_shape = (None,) + tuple(x.get_shape().as_list())[1:]
				slice_g = Lambda(
					slice_batch,  # lambda shape: shape,
					lambda shape: input_shape,arguments={'n_gpus': gpu_count, 'part': i})(x)
				slices.append(slice_g)
			
		with tf.device('/gpu:%d' % i):
			with tf.name_scope('tower_%d' % i) as scope:

				outputs = model(slices)
				tf.get_variable_scope().reuse_variables()

				if not isinstance(outputs, list):
					outputs = [outputs]

				# Save all the outputs for 
				# merging back together later
				for l in range(len(outputs)):
					all_outputs[l].append(outputs[l])

	# Merge outputs on CPU
	with tf.device('/cpu:0'):
		merged = []

		func = lambda x: L.concatenate(x, axis=0)

		for outputs in all_outputs:
			merged.append(func(outputs))

		return Model(inputs=model.inputs, outputs=merged)
