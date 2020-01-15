import math
import keras
from tensorflow import distributions, matrix_band_part, igamma, lgamma
from keras import backend as K

def cons_greater_zero(value):
	epsilon = 1e-9
	return epsilon + K.pow(value, 2)

class SPP(keras.layers.Layer):
	"""
	Parametric softplus activation layer.
	"""

	def __init__(self, alpha, **kwargs):
		super(SPP, self).__init__(**kwargs)
		self.__name__ = 'SPP'
		self.alpha = alpha

	def build(self, input_shape):
		super(SPP, self).build(input_shape)

	def call(self, inputs, **kwargs):
		return K.softplus(inputs) - self.alpha

	def compute_output_shape(self, input_shape):
		return input_shape

class SPPT(keras.layers.Layer):
	"""
	Trainable Parametric softplus activation layer.
	"""

	def __init__(self, **kwargs):
		super(SPPT, self).__init__(**kwargs)
		self.__name__ = 'SPP'

	def build(self, input_shape):
		self.alpha = self.add_weight(name='alpha', shape=(1,), dtype=K.floatx(),
									 initializer=keras.initializers.RandomUniform(minval=0, maxval=1),
									 trainable=True)

		super(SPPT, self).build(input_shape)

	def call(self, inputs, **kwargs):
		return K.softplus(inputs) - self.alpha

	def compute_output_shape(self, input_shape):
		return input_shape


def parametric_softplus(spp_alpha):
	"""
	Compute parametric softplus function with given alpha.
	:param spp_alpha: alpha parameter for softplus function.
	:return: parametric softplus activation value.
	"""

	def spp(x):
		return K.log(1 + K.exp(x)) - spp_alpha

	return spp


class MPELU(keras.layers.Layer):
	def __init__(self, channel_wise=True, **kwargs):
		super(MPELU, self).__init__(**kwargs)
		self.channel_wise = channel_wise

	def build(self, input_shape):
		shape = [1]

		if self.channel_wise:
			shape = [int(input_shape[-1])]  # Number of channels

		self.alpha = self.add_weight(name='alpha', shape=shape, dtype=K.floatx(),
									 initializer=keras.initializers.RandomUniform(minval=-1, maxval=1),
									 trainable=True)
		self.beta = self.add_weight(name='beta', shape=shape, dtype=K.floatx(),
									initializer=keras.initializers.RandomUniform(minval=0.0, maxval=1),
									trainable=True)

		# Finish buildidng
		super(MPELU, self).build(input_shape)

	def call(self, inputs, **kwargs):
		positive = keras.activations.relu(inputs)
		negative = self.alpha * (K.exp(-keras.activations.relu(-inputs) * cons_greater_zero(self.beta)) - 1)

		return positive + negative

	def compute_output_shape(self, input_shape):
		return input_shape


class RTReLU(keras.layers.Layer):
	def __init__(self, **kwargs):
		super(RTReLU, self).__init__(**kwargs)

	def build(self, input_shape):
		shape = [int(input_shape[-1])]  # Number of channels

		self.a = self.add_weight(name='a', shape=shape, dtype=K.floatx(),
								 initializer=keras.initializers.RandomUniform(minval=-1, maxval=1),
								 trainable=False)

		# Finish building
		super(RTReLU, self).build(input_shape)

	def call(self, inputs, **kwargs):
		return keras.activations.relu(inputs + self.a)

	def compute_output_shape(self, input_shape):
		return input_shape


class RTPReLU(keras.layers.PReLU):
	def __init__(self, **kwargs):
		super(RTPReLU, self).__init__(**kwargs)

	def build(self, input_shape):
		shape = [int(input_shape[-1])]  # Number of channels

		self.a = self.add_weight(name='a', shape=shape, dtype=K.floatx(),
								 initializer=keras.initializers.RandomUniform(minval=-1, maxval=1),
								 trainable=False)

		# Call PReLU build method
		super(RTPReLU, self).build(input_shape)

	def call(self, inputs, **kwargs):
		pos = keras.activations.relu(inputs + self.a)
		neg = -self.alpha * keras.activations.relu(-(inputs * self.a))

		return pos + neg


class PairedReLU(keras.layers.Layer):
	def __init__(self, scale=0.5, **kwargs):
		super(PairedReLU, self).__init__(**kwargs)
		self.scale = scale

	def build(self, input_shape):
		self.theta = self.add_weight(name='theta', shape=[1], dtype=K.floatx(),
									 initializer=keras.initializers.RandomUniform(minval=-1, maxval=1),
									 trainable=True)
		self.theta_p = self.add_weight(name='theta_p', shape=[1], dtype=K.floatx(),
									   initializer=keras.initializers.RandomUniform(minval=-1, maxval=1),
									   trainable=True)

		# Finish building
		super(PairedReLU, self).build(input_shape)

	def call(self, inputs, **kwargs):
		return K.concatenate(
			(keras.activations.relu(self.scale * inputs - self.theta), keras.activations.relu(-self.scale * inputs - self.theta_p)),
			axis=len(inputs.get_shape()) - 1)

	def compute_output_shape(self, input_shape):
		shape = list(input_shape)
		shape[-1]  = shape[-1] * 2
		shape = tuple(shape)
		return shape


class EReLU(keras.layers.Layer):
	def __init__(self, alpha=0.5, **kwargs):
		super(EReLU, self).__init__(**kwargs)
		self.alpha = alpha

	def build(self, input_shape):
		# shape = input_shape[1:]

		# self.k = self.add_weight(name='k', shape=shape, dtype=K.floatx(),
		# 						 initializer=keras.initializers.RandomUniform(minval=1 - self.alpha, maxval=1 + self.alpha), trainable=False)

		# Finish building
		super(EReLU, self).build(input_shape)

	def call(self, inputs, **kwargs):
		# Generate random uniform tensor between [1-alpha, 1+alpha] for training and ones tensor for test (ReLU)
		k = K.in_train_phase(K.random_uniform(inputs.shape[1:], 1 - self.alpha, 1 + self.alpha), K.ones(inputs.shape[1:]))

		return keras.activations.relu(inputs * k)

	def compute_output_shape(self, input_shape):
		return input_shape


class EPReLU(keras.layers.Layer):
	def __init__(self, alpha=0.5, **kwargs):
		super(EPReLU, self).__init__(**kwargs)
		self.alpha = alpha

	def build(self, input_shape):
		# Trainable (PReLU) parameter
		self.a = self.add_weight(name='a', shape=input_shape[1:], dtype=K.floatx(), initializer=keras.initializers.RandomUniform(0.0, 1.0))

		# Finish building
		super(EPReLU, self).build(input_shape)

	def call(self, inputs, **kwargs):
		# Generate random uniform tensor between [1-alpha, 1+alpha] for training and ones tensor for test
		k = K.in_train_phase(K.random_uniform(inputs.shape[1:], 1 - self.alpha, 1 + self.alpha),
							 K.ones(inputs.shape[1:]))

		pos = keras.activations.relu(inputs) * k
		neg = -self.a * keras.activations.relu(-inputs)

		return pos + neg


class SQRTActivation(keras.layers.Layer):
	def __init__(self, **kwargs):
		super(SQRTActivation, self).__init__(**kwargs)

	def build(self, input_shape):
		super(SQRTActivation, self).build(input_shape)

	def call(self, inputs, **kwargs):
		pos = K.sqrt(keras.activations.relu(inputs))
		neg = - K.sqrt(keras.activations.relu(-inputs))

		return pos + neg


# Randomized Leaky Rectified Linear Unit
class RReLu(keras.layers.Layer):
	def __init__(self, **kwargs):
		super(RReLu, self).__init__(**kwargs)

	def build(self, input_shape):
		# self.alpha = self.add_weight(name='alpha', shape=input_shape[1:], dtype=K.floatx(),
		#							 initializer=keras.initializers.RandomUniform(minval=0.0, maxval=1.0))

		super(RReLu, self).build(input_shape)

	def call(self, inputs, **kwargs):
		# Generate random uniform alpha
		alpha = K.in_train_phase(K.random_uniform(inputs.shape[1:], 0.0, 1.0), K.constant((0.0+1.0)/2.0, shape=inputs.shape[1:]))

		pos = keras.activations.relu(inputs)
		neg = alpha * keras.activations.relu(-inputs)

		return pos + neg


class PELU(keras.layers.Layer):
	def __init__(self, **kwargs):
		super(PELU, self).__init__(**kwargs)

	def build(self, input_shape):
		self.alpha = self.add_weight(name='alpha', shape=(1,), dtype=K.floatx(),
									 initializer=keras.initializers.RandomUniform(minval=0.0, maxval=1))
		# self.alpha = K.clip(self.alpha, 0.0001, 10)

		self.beta = self.add_weight(name='beta', shape=(1,), dtype=K.floatx(),
									initializer=keras.initializers.RandomUniform(minval=0.0, maxval=1))
		# self.beta = K.clip(self.beta, 0.0001, 10)

		super(PELU, self).build(input_shape)

	def call(self, inputs, **kwargs):
		pos = (cons_greater_zero(self.alpha) / cons_greater_zero(self.beta)) * keras.activations.relu(inputs)
		neg = cons_greater_zero(self.alpha) * (K.exp((-keras.activations.relu(-inputs)) / cons_greater_zero(self.beta)) - 1)

		return pos + neg


class SlopedReLU(keras.layers.Layer):
	def __init__(self, **kwargs):
		super(SlopedReLU, self).__init__(**kwargs)

	def build(self, input_shape):
		self.alpha = self.add_weight(name='alpha', shape=(1,), dtype=K.floatx(),
									 initializer=keras.initializers.RandomUniform(minval=1.0, maxval=10.0))
		self.alpha = K.clip(self.alpha, 1.0, 10)

		super(SlopedReLU, self).build(input_shape)

	def call(self, inputs, **kwargs):
		return keras.activations.relu(self.alpha * inputs)


class PTELU(keras.layers.Layer):
	def __init__(self, **kwargs):
		super(PTELU, self).__init__(**kwargs)

	def build(self, input_shape):
		self.alpha = self.add_weight(name='alpha', shape=(1,), dtype=K.floatx(),
									 initializer=keras.initializers.RandomUniform(minval=0.01, maxval=1))
		self.alpha = K.clip(self.alpha, 0.0001, 100)

		self.beta = self.add_weight(name='beta', shape=(1,), dtype=K.floatx(),
									initializer=keras.initializers.RandomUniform(minval=0.01, maxval=1))
		self.beta = K.clip(self.beta, 0.0001, 100)

		super(PTELU, self).build(input_shape)

	def call(self, inputs, **kwargs):
		pos = keras.activations.relu(inputs)
		neg = self.alpha * K.tanh(- self.beta * keras.activations.relu(-inputs))

		return pos + neg


class Antirectifier(keras.layers.Layer):
	'''This is the combination of a sample-wise
	L2 normalization with the concatenation of the
	positive part of the input with the negative part
	of the input. The result is a tensor of samples that are
	twice as large as the input samples.
	It can be used in place of a ReLU.
	# Input shape
		2D tensor of shape (samples, n)
	# Output shape
		2D tensor of shape (samples, 2*n)
	# Theoretical justification
		When applying ReLU, assuming that the distribution
		of the previous output is approximately centered around 0.,
		you are discarding half of your input. This is inefficient.
		Antirectifier allows to return all-positive outputs like ReLU,
		without discarding any data.
		Tests on MNIST show that Antirectifier allows to train networks
		with twice less parameters yet with comparable
		classification accuracy as an equivalent ReLU-based network.
	'''

	def compute_output_shape(self, input_shape):
		shape = list(input_shape)
		assert len(shape) == 2  # only valid for 2D tensors
		shape[-1] *= 2
		return tuple(shape)

	def call(self, inputs, **kwargs):
		inputs -= K.mean(inputs, axis=1, keepdims=True)
		inputs = K.l2_normalize(inputs, axis=1)
		pos = K.relu(inputs)
		neg = K.relu(-inputs)
		return K.concatenate([pos, neg], axis=1)


class CReLU(keras.layers.Layer):
	def compute_output_shape(self, input_shape):
		shape = list(input_shape)
		shape[-1] *= 2
		return tuple(shape)

	def call(self, inputs, **kwargs):
		pos = K.relu(inputs)
		neg = K.relu(-inputs)
		return K.concatenate([pos, neg])

class CLM(keras.layers.Layer):
	"""
	Proportional Odds Model activation layer.
	"""

	def __init__(self, num_classes, link_function, p, use_tau, **kwargs):
		self.num_classes = num_classes
		self.dist = distributions.Normal(loc=0., scale=1.)
		self.link_function = link_function
		self.p = p.copy()
		self.use_tau = use_tau
		super(CLM, self).__init__(**kwargs)

	def _convert_thresholds(self, b, a):
		a = K.pow(a, 2)
		thresholds_param = K.concatenate([b, a], axis=0)
		th = K.sum(
			matrix_band_part(K.ones([self.num_classes - 1, self.num_classes - 1]), -1, 0) * K.reshape(
				K.tile(thresholds_param, [self.num_classes - 1]), shape=[self.num_classes - 1, self.num_classes - 1]),
			axis=1)
		return th

	def _nnpom(self, projected, thresholds):
		if self.use_tau == 1:
			projected = K.reshape(projected, shape=[-1]) / self.tau
		else:
			projected = K.reshape(projected, shape=[-1])

		# projected = K.Print(projected, data=[K.reduce_min(projected), K.reduce_max(projected), K.reduce_mean(projected)], message='projected min max mean')

		m = K.shape(projected)[0]
		a = K.reshape(K.tile(thresholds, [m]), shape=[m, -1])
		b = K.transpose(K.reshape(K.tile(projected, [self.num_classes - 1]), shape=[-1, m]))
		z3 = a - b

		# z3 = K.cond(K.reduce_min(K.abs(z3)) < 0.01, lambda: K.Print(z3, data=[K.reduce_min(K.abs(z3))], message='z3 abs min', summarize=100), lambda: z3)

		if self.link_function == 'probit':
			a3T = self.dist.cdf(z3)
		elif self.link_function == 'cloglog':
			a3T = 1 - K.exp(-K.exp(z3))
		elif self.link_function == 'glogit':
			a3T = 1.0 / K.pow(1.0 + K.exp(-self.lmbd * (z3 - self.mu)), self.alpha)
		elif self.link_function == 'cauchit':
			a3T = K.atan(z3 / math.pi) + 0.5
		elif self.link_function == 'lgamma':
			a3T = K.cond(self.q < 0, lambda: igammac(K.pow(self.q, -2), K.pow(self.q, -2) * K.exp(self.q * z3)),
						  lambda: K.cond(self.q > 0, lambda: igamma(K.pow(self.q, -2),
																		K.pow(self.q, -2) * K.exp(self.q * z3)),
										  lambda: self.dist.cdf(z3)))
		elif self.link_function == 'gauss':
			# a3T = 1.0 / 2.0 + K.sign(z3) * K.igamma(1.0 / self.alpha, K.pow(K.abs(z3) / self.r, self.alpha)) / (2 * K.exp(K.lgamma(1.0 / self.alpha)))
			# z3 = K.Print(z3, data=[K.reduce_max(K.abs(z3))], message='z3 abs max')
			# K.sigmoid(z3 - self.p['mu']) - 1)
			a3T = 1.0 / 2.0 + K.tanh(z3 - self.p['mu']) * igamma(1.0 / self.p['alpha'],
																	K.pow(K.pow((z3 - self.p['mu']) / self.p['r'], 2),
																		   self.p['alpha'])) / (
								  2 * K.exp(lgamma(1.0 / self.p['alpha'])))
		elif self.link_function == 'expgauss':
			u = self.lmbd * (z3 - self.mu)
			v = self.lmbd * self.sigma
			dist1 = distributions.Normal(loc=0., scale=v)
			dist2 = distributions.Normal(loc=v, scale=K.pow(v, 2))
			a3T = dist1.cdf(u) - K.exp(-u + K.pow(v, 2) / 2 + K.log(dist2.cdf(u)))
		elif self.link_function == 'ggamma':
			a3T = igamma(self.p['d'] / self.p['p'], K.pow((z3 / self.p['a']), self.p['p'])) / K.exp(lgamma(self.p['d'] / self.p['p']))
		else:
			a3T = 1.0 / (1.0 + K.exp(-z3))

		a3 = K.concatenate([a3T, K.ones([m, 1])], axis=1)
		a3 = K.concatenate([K.reshape(a3[:, 0], shape=[-1, 1]), a3[:, 1:] - a3[:, 0:-1]], axis=-1)

		return a3

	def build(self, input_shape):
		self.thresholds_b = self.add_weight('b_b_nnpom', shape=(1,),
											initializer=keras.initializers.RandomUniform(minval=0, maxval=0.1))
		self.thresholds_a = self.add_weight('b_a_nnpom', shape=(self.num_classes - 2,),
											initializer=keras.initializers.RandomUniform(
												minval=math.sqrt((1.0 / (self.num_classes - 2)) / 2),
												maxval=math.sqrt(1.0 / (self.num_classes - 2))))

		if self.use_tau == 1:
			print('Using tau')
			self.tau = self.add_weight('tau_nnpom', shape=(1,),
									   initializer=keras.initializers.RandomUniform(minval=1, maxval=10))
			self.tau = K.clip(self.tau, 1, 1000)

		if self.link_function == 'glogit':
			self.lmbd = self.add_weight('lambda_nnpom', shape=(1,),
										initializer=keras.initializers.RandomUniform(minval=1, maxval=1))
			self.alpha = self.add_weight('alpha_nnpom', shape=(1,),
										 initializer=keras.initializers.RandomUniform(minval=1, maxval=1))
			self.mu = self.add_weight('mu_nnpom', shape=(1,),
									  initializer=keras.initializers.RandomUniform(minval=0, maxval=0))
		elif self.link_function == 'lgamma':
			self.q = self.add_weight('q_nnpom', shape=(1,),
									 initializer=keras.initializers.RandomUniform(minval=-1, maxval=1))
		elif self.link_function == 'gauss':
			if not 'alpha' in self.p:
				self.p['alpha'] = self.add_weight('alpha_nnpom', shape=(1,), initializer=keras.initializers.Constant(0.5))
				self.p['alpha'] = K.clip(self.p['alpha'], 0.1, 1.0)

			if not 'r' in self.p:
				self.p['r'] = self.add_weight('r_nnpom', shape=(1,), initializer=keras.initializers.Constant(1.0))
				self.p['r'] = K.clip(self.p['r'], 0.05, 100)

			if not 'mu' in self.p:
				self.p['mu'] = self.add_weight('mu_nnpom', shape=(1,), initializer=keras.initializers.Constant(0.0))

			# self.alpha = self.add_weight('alpha_nnpom', shape=(1,), initializer=keras.initializers.Constant(0.3))
			# self.alpha = K.clip(self.alpha, 0.2, 0.6)
			# self.alpha = 0.5
			# self.r = self.add_weight('r_nnpom', shape=(1,), initializer=keras.initializers.Constant(1.0))
			# self.r = K.clip(self.r, 0.2, 100)
			# self.r = 0.3
			# self.mu = self.add_weight('mu_nnpom', shape=(1,), initializer=keras.initializers.Constant(0.0))
			# self.mu = 0.0
		elif self.link_function == 'expgauss':
			self.mu = self.add_weight('mu_nnpom', shape=(1,), initializer=keras.initializers.Constant(0.0))
			self.sigma = self.add_weight('sigma_nnpom', shape=(1,), initializer=keras.initializers.Constant(1.0))
			self.lmbd = self.add_weight('lambda_nnpom', shape=(1,), initializer=keras.initializers.Constant(1.0))
		elif self.link_function == 'ggamma':
			self.__set_default_param('a', self.add_weight('a_clm', shape=(1,), initializer=keras.initializers.Constant(0.5)))
			self.__set_default_param('d', self.add_weight('d_clm', shape=(1,), initializer=keras.initializers.Constant(0.5)))
			self.__set_default_param('p', self.add_weight('p_clm', shape=(1,), initializer=keras.initializers.Constant(0.5)))


	def call(self, x, **kwargs):
		thresholds = self._convert_thresholds(self.thresholds_b, self.thresholds_a)
		return self._nnpom(x, thresholds)

	def compute_output_shape(self, input_shape):
		return (input_shape[0], 1)

	def __set_default_param(self, param, value):
		if not param in self.p:
			self.p[param] = value