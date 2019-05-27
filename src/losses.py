import keras
import numpy as np
from keras import backend as K

def make_cost_matrix(num_ratings):
	"""
	Create a quadratic cost matrix of num_ratings x num_ratings elements.

	:param thresholds_b: threshold b1.
	:param thresholds_a: thresholds alphas vector.
	:param num_labels: number of labels.
	:return: cost matrix.
	"""

	cost_matrix = np.reshape(np.tile(range(num_ratings), num_ratings), (num_ratings, num_ratings))
	cost_matrix = np.power(cost_matrix - np.transpose(cost_matrix), 2) / (num_ratings - 1) ** 2.0
	return np.float32(cost_matrix)


def qwk_loss(cost_matrix):
	"""
	Compute QWK loss function.

	:param pred_prob: predict probabilities tensor.
	:param true_prob: true probabilities tensor.
	:param cost_matrix: cost matrix.
	:return: QWK loss value.
	"""
	def _qwk_loss(true_prob, pred_prob):
		targets = K.argmax(true_prob, axis=1)
		costs = K.gather(cost_matrix, targets)


		# costs = K.Print(costs, data=[costs], summarize=100, message='costs')

#		pred_cls = K.argmax(pred_prob, axis=1)

# 		conf_mat = K.confusion_matrix(targets, pred_cls)

		numerator = costs * pred_prob
		numerator = K.sum(numerator)

		sum_prob = K.sum(pred_prob, axis=0)
		n = K.sum(true_prob, axis=0)

		a = K.reshape(K.dot(cost_matrix, K.reshape(sum_prob, shape=[-1, 1])), shape=[-1])
		b = K.reshape(n / K.sum(n), shape=[-1])

		epsilon = 10e-9

		denominator = a * b
		denominator = K.sum(denominator) + epsilon

		return numerator / denominator # + K.cast(K.sum(conf_mat) * 0, dtype=K.floatx())

	return _qwk_loss
