import keras
from activations import CLM
from resnet import Resnet_2x4



class Net:
	def __init__(self, size, activation, final_activation, f_a_params={}, use_tau=True, num_channels=3,
				 num_classes=5, spp_alpha=0.2, dropout=0):
		self.size = size
		self.activation = activation
		self.final_activation = final_activation
		self.f_a_params = f_a_params
		self.use_tau = use_tau
		self.num_channels = num_channels
		self.num_classes = num_classes
		self.spp_alpha = spp_alpha
		self.dropout = dropout

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

	def beckham_resnet(self):
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
		else:
			x = keras.layers.Dense(self.num_classes)(x)
			x = keras.layers.Activation(self.final_activation)(x)

		return x
