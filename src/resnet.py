# -*- coding: utf-8 -*-

from keras.regularizers import l2

from keras.models import Model, Sequential
from keras.initializers import Constant, he_normal
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, \
	Dense, Flatten, Activation, ZeroPadding2D, \
	Add, Activation, BatchNormalization, GlobalAveragePooling2D


def _residual_block(layer, n_out_channels, stride=1, nonlinearity='relu'):
	"""Crea un bloque residual de la red.

    :param layer: Capa de anterior al bloque residual a crear.
    :param n_out_channels: Número de filtros deseados para las convoluciones realizadas en el bloque.
    :param stride: Stride para la primera convolución realizada y en el caso de ser mayor a 1, el usado en un primer AveragePooling.
    :param nonlinearity: No linealidad aplicada a las salidas de las BatchNormalization aplicadas.
    :return: La última capa del bloque residual.
    """
	conv = layer
	if stride > 1:
		# padding: https://stackoverflow.com/a/47213171
		layer = AveragePooling2D(pool_size=1, strides=stride, padding="same")(layer)
	# Si no hay concordancia de dimensiones entre las capas se hace un padding con ceros
	if (n_out_channels != int(layer.get_shape()[3])):
		diff = n_out_channels - int(layer.get_shape()[3])
		diff_2 = int(diff / 2)
		if diff % 2 == 0:
			width_tp = ((0, 0), (diff_2, diff_2))
		else:
			width_tp = ((0, 0), ((diff_2) + 1, diff_2))
		# Para que el pad se haga en la dimension correcta, al no poder seleccionar
		# como en lasagne batch_ndim, se usa data_format='channels_last'
		layer = ZeroPadding2D(padding=(width_tp), data_format='channels_first')(layer)
	conv = Conv2D(filters=n_out_channels, kernel_size=(3, 3),
				  strides=(stride, stride), padding='same',
				  activation='linear',
				  kernel_initializer=he_normal(),
				  bias_initializer=Constant(0.),
				  kernel_regularizer=l2(1e-4),
				  bias_regularizer=l2(1e-4))(conv)
	conv = BatchNormalization(beta_initializer=Constant(0.),
							  gamma_initializer=Constant(1.),
							  beta_regularizer=l2(1e-4),
							  gamma_regularizer=l2(1e-4))(conv)
	conv = Activation(nonlinearity)(conv)
	conv = Conv2D(filters=n_out_channels, kernel_size=(3, 3),
				  strides=(1, 1), padding='same',
				  activation='linear',
				  kernel_initializer=he_normal(),
				  bias_initializer=Constant(0.),
				  kernel_regularizer=l2(1e-4),
				  bias_regularizer=l2(1e-4))(conv)
	conv = BatchNormalization(beta_initializer=Constant(0.),
							  gamma_initializer=Constant(1.),
							  beta_regularizer=l2(1e-4),
							  gamma_regularizer=l2(1e-4))(conv)
	sum_ = Add()([conv, layer])
	return Activation(nonlinearity)(sum_)


def _resnet_2x4(l_in, nf=(32, 64, 128, 256), N=2, activation='relu'):
	"""Función para crear de manera automática los múltiples bloques residuales de la red.

    :param l_in: Capa de entrada.
    :param nf: Lista con el distinto número de filtros a utilizar por capa.
    :param N: Profundidad de los bloques residuales intermedios y contiguos.
    :return: La última capa después de los bloques residuales.
    """
	assert len(nf) == 4  # this is a 4-block resnet
	layer = Conv2D(filters=nf[0], kernel_size=7,
				   strides=2, activation=activation,
				   bias_initializer=Constant(0),
				   padding='same', kernel_initializer=he_normal(),
				   kernel_regularizer=l2(1e-4),
				   bias_regularizer=l2(1e-4))(l_in)
	layer = MaxPooling2D(pool_size=3, strides=2)(layer)

	# Residual blocks go here
	#
	for i in range(N):
		layer = _residual_block(layer, nf[0], nonlinearity=activation)
	layer = _residual_block(layer, nf[1], stride=2, nonlinearity=activation)

	for i in range(N):
		layer = _residual_block(layer, nf[1], nonlinearity=activation)
	layer = _residual_block(layer, nf[2], stride=2, nonlinearity=activation)

	for i in range(N):
		layer = _residual_block(layer, nf[2], nonlinearity=activation)
	layer = _residual_block(layer, nf[3], stride=2, nonlinearity=activation)

	for i in range(N):
		layer = _residual_block(layer, nf[3], nonlinearity=activation)

	# layer = AveragePooling2D(pool_size=int(layer.get_shape()[-1]), strides=1, padding='valid')(layer)
	layer = GlobalAveragePooling2D(data_format='channels_last')(layer)

	# layer = Flatten()(layer)

	model = Model(l_in, layer, name='resnet2x4')

	return model


class Resnet_2x4():
	"""Construye la arquitectura base de la red residual usada.

    :param input_shape: Dimensiones de las imagenes usadas (NxWxH).
    """

	def __init__(self, input_shape=(3, 224, 224), activation='relu'):
		self.inputs = Input(shape=input_shape)
		self.net = _resnet_2x4(self.inputs, activation=activation)

	def get_net(self):
		"""Devuelve la salida de la red para los patrones dados como entrada.

        :return: Clasificación de la red para los patrones de entrada.
        """
		return self.net