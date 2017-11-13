import keras.backend as K

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