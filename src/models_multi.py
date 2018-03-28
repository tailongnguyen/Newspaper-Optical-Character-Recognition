from keras.models import load_model, Model
from keras.layers import Dense, Flatten, MaxPooling2D
from keras.layers import Conv2D, Lambda, Input, Activation
from keras.layers import LSTM, TimeDistributed, Bidirectional, GRU
from keras.layers.merge import add, concatenate
from keras.optimizers import SGD
from keras import backend as K
from multi_gpu import *

import tensorflow as tf
import random 
import keras
import numpy as np 

class CNN(object): 
	"""docstring for CNN"""
	def __init__(self, input_shape = (32,32,3), num_classes = 62):
		super(CNN, self).__init__()
		self.input_shape = input_shape
		self.num_classes = num_classes
		self.inp = Input(shape = self.input_shape, name = "Input")
		self.label = K.placeholder(shape= (None, self.num_classes), dtype = 'float32')

		self.conv1 = Conv2D(96, kernel_size=9, data_format='channels_last', activation='relu', input_shape=self.input_shape)(self.inp)
		self.conv1 = Lambda(self.Maxout, name ='act1', arguments={"num_unit": 2})(self.conv1)
		self.conv2 = Conv2D(128, 9, activation='relu', data_format='channels_last')(self.conv1)
		self.conv2 = Lambda(self.Maxout, name ='act2', arguments={"num_unit": 2})(self.conv2) 
		self.conv3 = Conv2D(256, 9, activation='relu', data_format='channels_last')(self.conv2)
		self.conv3 = Lambda(self.Maxout, name ='act3', arguments={"num_unit": 2})(self.conv3)
		self.conv4 = Conv2D(512, 8, activation='relu', data_format='channels_last')(self.conv3)
		self.conv4 = Lambda(self.Maxout, name ='act4', arguments={"num_unit": 4})(self.conv4)
		self.conv5 = Conv2D(144, 1, activation='relu', data_format='channels_last')(self.conv4)
		self.conv5 = Lambda(self.Maxout, name ='act5', arguments={"num_unit": 4})(self.conv5)

		self.out = Dense(num_classes, activation='softmax')(Flatten()(self.conv5))

		self.model = Model(inputs = self.inp, outputs = self.out)
		self.cross_entropy = K.categorical_crossentropy(self.out, self.label)
		self.cross_entropy = K.mean(self.cross_entropy)
		self.optimizer = keras.optimizers.Adam(lr = 0.001)
		self.update = self.optimizer.get_updates(self.model.trainable_weights, [], loss = self.cross_entropy)
		self.train_step = K.function([self.inp, self.label, K.learning_phase()], [self.out, self.cross_entropy], updates = self.update)
		self.accuracy = K.function([self.inp, self.label, K.learning_phase()], \
							[K.mean(K.cast(K.equal(K.argmax(self.out, 1), K.argmax(self.label, 1)), dtype ='float32'))])
		self.features_vec = K.function([self.inp, K.learning_phase()], [K.reshape(self.conv4, (-1,128))])

	def Maxout(self, x, num_unit = None):
	    """
	    Maxout as in the paper `Maxout Networks <http://arxiv.org/abs/1302.4389>`_.
	    Args:
	        x (tf.Tensor): a NHWC or NC tensor. Channel has to be known.
	        num_unit (int): a int. Must be divisible by C.
	    Returns:
	        tf.Tensor: of shape NHW(C/num_unit) named ``output``.
	    """
	    input_shape = x.get_shape().as_list()
	    ndim = len(input_shape)
	    # print "Input shape: ", input_shape
	    assert ndim == 4 or ndim == 2
	    ch = input_shape[-1]
	    assert ch is not None and ch % num_unit == 0
	    if ndim == 4:
	        x = K.reshape(x, [-1, input_shape[1], input_shape[2], ch / num_unit, num_unit])
	    else:
	        x = K.reshape(x, [-1, ch / num_unit, num_unit])
	    return K.max(x, ndim)	


class CRNN(object):
	"""docstring for RNN"""
	def __init__(self, learning_rate = 0.001, output_dim = 63, gpu_count=2):
		conv_filters = 16
		kernel_size = (3, 3)
		pool_size = 2
		time_dense_size = 32
		rnn_size = 512
		img_h = 32
		act = 'relu'

		self.learning_rate = learning_rate
		self.width = K.placeholder(name= 'width', ndim =0, dtype='int32')
		self.input_data = Input(name='the_input', shape=(None, img_h, 1), dtype='float32')
		self.inner = Conv2D(conv_filters, kernel_size, padding='same',
					activation=act, kernel_initializer='he_normal',
					name='conv1')(self.input_data)
		self.inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(self.inner)
		self.inner = Conv2D(conv_filters, kernel_size, padding='same',
					activation=act, kernel_initializer='he_normal',
					name='conv2')(self.inner)
		self.inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(self.inner)

		self.inner = Lambda(self.res, arguments={"last_dim": (img_h // (pool_size ** 2)) * conv_filters \
                                                , "width": self.width // 4})(self.inner)

		# cuts down input size going into RNN:
		self.inp = Dense(time_dense_size, activation=act, name='dense1')(self.inner)
		self.batch_norm = keras.layers.normalization.BatchNormalization()(self.inp)

		self.gru_1 = Bidirectional(GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal',\
									 name='gru1'),merge_mode="sum")(self.batch_norm)
		self.gru_2 = Bidirectional(GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal',\
									 name='gru2'),merge_mode="concat")(self.gru_1)
		self.gru_3 = Bidirectional(GRU(rnn_size, recurrent_dropout=0.5, return_sequences=True, \
								kernel_initializer='he_normal', name='gru3'),merge_mode="concat")(self.gru_2)
		self.gru_4 = Bidirectional(GRU(rnn_size, recurrent_dropout=0.5, return_sequences=True, \
								kernel_initializer='he_normal', name='gru4'),merge_mode="concat")(self.gru_3)

		self.y_pred = TimeDistributed(Dense(output_dim, kernel_initializer='he_normal', \
		              			name='dense2', activation='linear'))(self.gru_4)

		self.model = Model(inputs=self.input_data, outputs=self.y_pred)
		self.model = make_parallel(self.model, gpu_count)
		self.model.summary()
		
		self.predict_step = K.function([self.input_data, self.width, K.learning_phase()],\
										 [K.argmax(self.y_pred, axis=2)])
										 
		self.prob = K.function([self.input_data, self.width, K.learning_phase()], [K.softmax(self.y_pred)])

	def compile_model(self):
		import warpctc_tensorflow

		self.output_ctc = self.model.outputs[0]	
		self.y_true = K.placeholder(name='y_true', ndim=1, dtype='int32')
		self.input_length = K.placeholder(name='input_length', ndim=1, dtype='int32')
		self.label_length = K.placeholder(name='label_length', ndim=1, dtype='int32')
		self.loss_out = K.mean(warpctc_tensorflow.ctc(tf.transpose(self.output_ctc, perm=[1,0,2]),\
												 self.y_true, self.label_length, self.input_length))
		self.optimizer = keras.optimizers.SGD(lr=self.learning_rate, decay=1e-6, \
												momentum=0.9, nesterov=True, clipnorm = 200)
		self.update = self.optimizer.get_updates(self.model.trainable_weights, [], loss = self.loss_out)

		self.train_step = K.function([self.input_data, self.width, self.y_true, self.input_length, \
									  self.label_length, K.learning_phase()], \
										[self.loss_out, self.output_ctc], updates = self.update)


	def res (self, x, width, last_dim):
		return K.reshape(x, (-1, width, last_dim))

