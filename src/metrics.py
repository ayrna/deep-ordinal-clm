import tensorflow as tf
import keras
import numpy as np
from keras import backend as K
from sklearn.metrics import confusion_matrix

def _confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
	"""
	Returns the confusion matrix between rater's ratings
	"""
	assert (len(rater_a) == len(rater_b))
	if min_rating is None:
		min_rating = min(rater_a + rater_b)
	if max_rating is None:
		max_rating = max(rater_a + rater_b)
	num_ratings = int(max_rating - min_rating + 1)
	conf_mat = [[0 for i in range(num_ratings)]
				for j in range(num_ratings)]
	for a, b in zip(rater_a, rater_b):
		conf_mat[a - min_rating][b - min_rating] += 1
	return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
	"""
	Returns the counts of each type of rating that a rater made
	"""
	if min_rating is None:
		min_rating = min(ratings)
	if max_rating is None:
		max_rating = max(ratings)
	num_ratings = int(max_rating - min_rating + 1)
	hist_ratings = [0 for x in range(num_ratings)]
	for r in ratings:
		hist_ratings[r - min_rating] += 1

	return hist_ratings

def np_quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):
	"""
	Calculates the quadratic weighted kappa
	quadratic_weighted_kappa calculates the quadratic weighted kappa
	value, which is a measure of inter-rater agreement between two raters
	that provide discrete numeric ratings.  Potential values range from -1
	(representing complete disagreement) to 1 (representing complete
	agreement).  A kappa value of 0 is expected if all agreement is due to
	chance.

	quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
	each correspond to a list of integer ratings.  These lists must have the
	same length.

	The ratings should be integers, and it is assumed that they contain
	the complete range of possible ratings.

	quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
	is the minimum possible rating, and max_rating is the maximum possible
	rating
	"""

	# Change values lower than min_rating to min_rating and values
	# higher than max_rating to max_rating.
	rater_a = np.clip(rater_a, min_rating, max_rating)
	rater_b = np.clip(rater_b, min_rating, max_rating)

	rater_a = np.round(rater_a).astype(int).ravel()
	rater_a[~np.isfinite(rater_a)] = 0
	rater_b = np.round(rater_b).astype(int).ravel()
	rater_b[~np.isfinite(rater_b)] = 0

	assert (len(rater_a) == len(rater_b))
	# Get min_rating and max_rating from raters if they are None.
	if min_rating is None:
		min_rating = min(min(rater_a), min(rater_b))
	if max_rating is None:
		max_rating = max(max(rater_a), max(rater_b))
	conf_mat = _confusion_matrix(rater_a, rater_b,
								min_rating, max_rating)

	num_ratings = len(conf_mat)
	num_scored_items = float(len(rater_a))

	hist_rater_a = histogram(rater_a, min_rating, max_rating)
	hist_rater_b = histogram(rater_b, min_rating, max_rating)

	numerator = 0.0
	denominator = 0.0

	for i in range(num_ratings):
		for j in range(num_ratings):
			expected_count = (hist_rater_a[i] * hist_rater_b[j]
							  / num_scored_items)
			d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
			numerator += d * conf_mat[i][j] / num_scored_items
			denominator += d * expected_count / num_scored_items

	return 1.0 - numerator / denominator


def top_2_accuracy(y_true, y_pred):
	return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)

def top_3_accuracy(y_true, y_pred):
	return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

def _compute_sensitivities(y_true, y_pred):
	if y_true.shape[1] > 1:
		y_true = np.argmax(y_true, axis=1)
	if y_pred.shape[1] > 1:
		y_pred = np.argmax(y_pred, axis=1)

	conf_mat = confusion_matrix(y_true, y_pred)

	sum = np.sum(conf_mat, axis=1)
	mask = np.eye(conf_mat.shape[0], conf_mat.shape[1])
	correct = np.sum(conf_mat * mask, axis=1)
	sensitivities = correct / sum

	return sensitivities

def minimum_sensitivity(y_true, y_pred):
	return np.min(_compute_sensitivities(y_true, y_pred))

def accuracy_off1(y_true, y_pred):
	if y_true.shape[1] > 1:
		y_true = np.argmax(y_true, axis=1)
	if y_pred.shape[1] > 1:
		y_pred = np.argmax(y_pred, axis=1)

	conf_mat = confusion_matrix(y_true, y_pred)
	n = conf_mat.shape[0]
	mask = np.eye(n, n) + np.eye(n, n, k=1), + np.eye(n, n, k=-1)
	correct = mask * conf_mat

	return 1.0 * np.sum(correct) / np.sum(conf_mat)