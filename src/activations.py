import math
import keras
from tensorflow import distributions, matrix_band_part, igamma, lgamma
from keras import backend as K

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


	def call(self, x):
		thresholds = self._convert_thresholds(self.thresholds_b, self.thresholds_a)
		return self._nnpom(x, thresholds)

	def compute_output_shape(self, input_shape):
		return (input_shape[0], 1)

	def __set_default_param(self, param, value):
		if not param in self.p:
			self.p[param] = value