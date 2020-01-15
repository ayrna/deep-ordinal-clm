import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout, Flatten, Dense, Lambda
from keras import regularizers
from keras import backend as K
from activations import SPP, SPPT, MPELU, RTReLU, RTPReLU, PairedReLU, EReLU, SQRTActivation, CLM, RReLu, PELU, SlopedReLU, PTELU, Antirectifier, CReLU, EPReLU
from layers import GeometricLayer, ScaleLayer
from resnet import Resnet_2x4

from inception_resnet_v2 import InceptionResNetV2 as Irnv2
from unimodal_extensions import _add_binom_m


class Net:
	def __init__(self, size, activation, final_activation, f_a_params={}, use_tau=True, prob_layer=None, num_channels=3,
				 num_classes=5, spp_alpha=0.2, dropout=0):
		self.size = size
		self.activation = activation
		self.final_activation = final_activation
		self.f_a_params = f_a_params
		self.use_tau = use_tau
		self.prob_layer = prob_layer
		self.num_channels = num_channels
		self.num_classes = num_classes
		self.spp_alpha = spp_alpha
		self.dropout = dropout

	def build(self, net_model):
		if hasattr(self, net_model):
			return getattr(self, net_model)()
		else:
			raise Exception('Invalid network model.')

	def vgg19(self):
		model = keras.models.Sequential([
			# Block 1
			keras.layers.Conv2D(64, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',
								input_shape=(self.size, self.size, self.num_channels), data_format='channels_last'),
			self.__activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(64, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', ),
			self.__activation(),
			keras.layers.BatchNormalization(),
			keras.layers.MaxPooling2D(),

			# Block 2
			keras.layers.Conv2D(128, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', ),
			self.__activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(128, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', ),
			self.__activation(),
			keras.layers.BatchNormalization(),
			keras.layers.MaxPooling2D(),

			# Block 3
			keras.layers.Conv2D(256, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', ),
			self.__activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(256, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', ),
			self.__activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(256, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', ),
			self.__activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(256, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', ),
			self.__activation(),
			keras.layers.BatchNormalization(),
			keras.layers.MaxPooling2D(),

			# Block 4
			keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', ),
			self.__activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', ),
			self.__activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', ),
			self.__activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', ),
			self.__activation(),
			keras.layers.BatchNormalization(),
			keras.layers.MaxPooling2D(),

			# Block 5
			keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', ),
			self.__activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', ),
			self.__activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', ),
			self.__activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', ),
			self.__activation(),
			keras.layers.BatchNormalization(),
			keras.layers.MaxPooling2D(),

			# Classification block
			keras.layers.Flatten(),
			keras.layers.Dropout(rate=self.dropout),
			keras.layers.Dense(4096),
			self.__activation(),
			keras.layers.Dense(4096),
			self.__activation(),
		])

		model = self.__final_activation(model)

		return model

	def vgg16(self):
		# Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
		model = Sequential()
		weight_decay = 0.0005

		model.add(Conv2D(64, (3, 3), padding='same',
						 input_shape=(self.size, self.size, self.num_channels), kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.3))

		model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.5))

		model.add(Flatten())
		model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())

		model.add(Dropout(0.5))
		model.add(Dense(self.num_classes))
		model.add(Activation('softmax'))

		return model

	def vgg16pu(self):
		# Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
		model = Sequential()
		weight_decay = 0.0005

		model.add(Conv2D(64, (3, 3), padding='same',
						 input_shape=(self.size, self.size, self.num_channels), kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.3))

		model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.5))

		model.add(Flatten())
		model.add(ScaleLayer())
		model.add(Lambda(lambda x: K.log(x)))
		model.add(Dense(16))
		model.add(Lambda(lambda x: K.exp(x)))
		model.add(self.__activation())
		model.add(BatchNormalization())

		model.add(Dropout(0.5))
		model.add(Dense(self.num_classes))
		model.add(Activation('softmax'))

		return model


	def conv128(self):

		feature_filter_size = 3
		classif_filter_size = 4

		input = keras.layers.Input(shape=(self.size, self.size, self.num_channels))

		x = keras.layers.Conv2D(32, feature_filter_size, strides=(1, 1), kernel_initializer='he_uniform')(input)
		x = self.__activation()(x)
		x = keras.layers.BatchNormalization()(x)
		x = keras.layers.Conv2D(32, feature_filter_size, strides=(1, 1), kernel_initializer='he_uniform')(x)
		x = self.__activation()(x)
		x = keras.layers.BatchNormalization()(x)
		x = keras.layers.MaxPooling2D()(x)

		x = keras.layers.Conv2D(64, feature_filter_size, strides=(1, 1), kernel_initializer='he_uniform')(x)
		x = self.__activation()(x)
		x = keras.layers.BatchNormalization()(x)
		x = keras.layers.Conv2D(64, feature_filter_size, strides=(1, 1), kernel_initializer='he_uniform')(x)
		x = self.__activation()(x)
		x = keras.layers.BatchNormalization()(x)
		x = keras.layers.MaxPooling2D()(x)

		x = keras.layers.Conv2D(128, feature_filter_size, strides=(1, 1), kernel_initializer='he_uniform')(x)
		x = self.__activation()(x)
		x = keras.layers.BatchNormalization()(x)
		x = keras.layers.Conv2D(128, feature_filter_size, strides=(1, 1), kernel_initializer='he_uniform')(x)
		x = self.__activation()(x)
		x = keras.layers.BatchNormalization()(x)
		x = keras.layers.MaxPooling2D()(x)

		x = keras.layers.Conv2D(128, feature_filter_size, strides=(1, 1), kernel_initializer='he_uniform')(x)
		x = self.__activation()(x)
		x = keras.layers.BatchNormalization()(x)
		x = keras.layers.Conv2D(128, feature_filter_size, strides=(1, 1), kernel_initializer='he_uniform')(x)
		x = self.__activation()(x)
		x = keras.layers.BatchNormalization()(x)
		x = keras.layers.MaxPooling2D()(x)

		x = keras.layers.Conv2D(128, classif_filter_size, strides=(1, 1), kernel_initializer='he_uniform')(x)
		x = self.__activation()(x)
		x = keras.layers.BatchNormalization()(x)

		x = keras.layers.Flatten()(x)

		x = keras.layers.Dense(96)(x)

		if self.dropout > 0:
			x = keras.layers.Dropout(rate=self.dropout)(x)

		x = self.__final_activation(x)

		model = keras.models.Model(input, x)

		return model

	def inceptionresnetv2(self):
		input = keras.layers.Input(shape=(self.size, self.size, self.num_channels))
		x = input
		# Required size >= 75 x 75
		size = self.size
		if size < 75:
			size = 75
			x = keras.layers.ZeroPadding2D(padding=(75 - self.size) // 2 + 1)(x)

		x = Irnv2(input_tensor=x, include_top=False, input_shape=(size, size, self.num_channels),
				  classes=self.num_classes, pooling='avg', activation=self.__activation())(x)

		x = keras.layers.Dense(512)(x)

		if self.dropout > 0:
			x = keras.layers.Dropout(rate=self.dropout)(x)

		x = self.__final_activation(x)

		model = keras.models.Model(input, x)

		return model

	def beckhamresnet(self):
		input = keras.layers.Input(shape=(self.size, self.size, self.num_channels))
		x = input

		resnet = Resnet_2x4((self.size, self.size, self.num_channels), activation=self.activation)
		x = resnet.get_net()(x)

		if self.dropout > 0:
			x = keras.layers.Dropout(rate=self.dropout)(x)

		x = self.__final_activation(x)

		model = keras.models.Model(input, x)

		return model

	def __activation(self):
		if self.activation == 'relu':
			return keras.layers.Activation('relu')
		elif self.activation == 'lrelu':
			return keras.layers.LeakyReLU()
		elif self.activation == 'prelu':
			return keras.layers.PReLU()
		elif self.activation == 'elu':
			return keras.layers.ELU()
		elif self.activation == 'softplus':
			return keras.layers.Activation('softplus')
		elif self.activation == 'spp':
			return SPP(self.spp_alpha)
		elif self.activation == 'sppt':
			return SPPT()
		elif self.activation == 'mpelu':
			return MPELU(channel_wise=True)
		elif self.activation == 'rtrelu':
			return RTReLU()
		elif self.activation == 'rtprelu':
			return RTPReLU()
		elif self.activation == 'pairedrelu':
			return PairedReLU()
		elif self.activation == 'erelu':
			return EReLU()
		elif self.activation == 'eprelu':
			return EPReLU()
		elif self.activation == 'sqrt':
			return SQRTActivation()
		elif self.activation == 'rrelu':
			return RReLu()
		elif self.activation == 'pelu':
			return PELU()
		elif self.activation == 'slopedrelu':
			return SlopedReLU()
		elif self.activation == 'ptelu':
			return PTELU()
		elif self.activation == 'antirectifier':
			return Antirectifier()
		elif self.activation == 'crelu':
			return CReLU()
		else:
			return keras.layers.Activation('relu')

	def __final_activation(self, x):
		if self.final_activation == 'poml':
			x = keras.layers.Dense(1)(x)
			x = keras.layers.BatchNormalization()(x)
			x = CLM(self.num_classes, 'logit', self.f_a_params, use_tau=self.use_tau)(x)
		elif self.final_activation == 'pomp':
			x = keras.layers.Dense(1)(x)
			x = keras.layers.BatchNormalization()(x)
			x = CLM(self.num_classes, 'probit', self.f_a_params, use_tau=self.use_tau)(x)
		elif self.final_activation == 'pomclog':
			x = keras.layers.Dense(1)(x)
			x = keras.layers.BatchNormalization()(x)
			x = CLM(self.num_classes, 'cloglog', self.f_a_params, use_tau=self.use_tau)(x)
		elif self.final_activation == 'pomglogit':
			x = keras.layers.Dense(1)(x)
			x = keras.layers.BatchNormalization()(x)
			x = CLM(self.num_classes, 'glogit', self.f_a_params, use_tau=self.use_tau)(x)
		elif self.final_activation == 'clmcauchit':
			x = keras.layers.Dense(1)(x)
			x = keras.layers.BatchNormalization()(x)
			x = CLM(self.num_classes, 'cauchit', self.f_a_params, use_tau=self.use_tau)(x)
		elif self.final_activation == 'clmggamma':
			x = keras.layers.Dense(1)(x)
			x = keras.layers.BatchNormalization()(x)
			x = CLM(self.num_classes, 'ggamma', self.f_a_params, use_tau=self.use_tau)(x)
		elif self.final_activation == 'clmgauss':
			x = keras.layers.Dense(1)(x)
			x = keras.layers.BatchNormalization()(x)
			x = CLM(self.num_classes, 'gauss', self.f_a_params, use_tau=self.use_tau)(x)
		elif self.final_activation == 'clmexpgauss':
			x = keras.layers.Dense(1)(x)
			x = keras.layers.BatchNormalization()(x)
			x = CLM(self.num_classes, 'expgauss', self.f_a_params, use_tau=self.use_tau)(x)
		elif self.final_activation == 'binomial':
			_add_binom_m(model, self.num_classes, 1.0, 'sigm_learnable')
		else:
			x = keras.layers.Dense(self.num_classes)(x)
			if self.prob_layer == 'geometric':
				x = GeometricLayer()(x)
			x = keras.layers.Activation(self.final_activation)(x)

		return x
