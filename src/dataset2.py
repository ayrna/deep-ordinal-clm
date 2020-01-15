import numpy as np
import os
import math
from sklearn.model_selection import train_test_split, StratifiedKFold
import keras
import cv2
import pandas as pd
from skimage.io import imread
from sklearn.utils.class_weight import compute_class_weight
from generators import SmallGenerator, BigGenerator
from multiprocessing import Pool
from functools import partial

DATASETS_DIR = '../datasets/'
# Change datasets directory through environ variable
if 'DATASETS_DIR' in os.environ:
	DATASETS_DIR = os.environ['DATASETS_DIR']

# Parallel sum of image pixels
def parallel_img_sum(base_path, path):
	img = imread(os.path.join(base_path, path))
	return img.sum()

# Parallel square sum of image pixels
def parallel_variance_sum(base_path, mean, n, path):
	img = imread(os.path.join(base_path, path))
	return np.sum(((img - mean) ** 2) / n)


class Dataset:
	"""
	Class that represents a dataset that is loaded from a file.
	"""
	def __init__(self, name, seed=1):
		# Name / path of the dataset
		self._name = name
		
		# Random seed
		self._seed = seed

		# Default holdout / kfold values
		self._n_folds = 1 # Holdout
		self._holdout = 0.2 # for validation
		self._folds_indices = None

		# Initialize current fold
		self._current_fold = 0

		# Load status
		self._loaded = False
		self._big_dataset = False
		self._splits_loaded = False

		# Numpy arrays for small datasets
		self._x_trainval = None
		self._y_trainval = None
		self._x_test = None
		self._y_test = None

		# Numpy arrays for splitted dataset (folds)
		self._x_train = None
		self._y_train = None
		self._x_val = None
		self._y_val = None

		# Dataframes for big datasets
		self._df_trainval = None
		self._df_test = None

		# Dataframes for splitted dataset (folds)
		self._df_train = None
		self._df_val = None

		# Set dataframes x and y columns
		self._x_col = 'path'
		self._y_col = 'y'

		# Base path for images of big datasets
		self._base_path = None

		# Generator for each dataset split
		self._train_generator = None
		self._val_generator = None
		self._test_generator = None

		# Store means and std to avoid multiple calculations
		self._mean_train = None
		self._mean_val = None
		self._mean_test = None
		self._std_train = None
		self._std_val = None
		self._std_test = None

		# Do not load dataset here. Better load it when we need it.
		# self.load(name)

	# Load dataset and splits if not loaded
	def load(self, name):
		if not self._loaded:
			if hasattr(self, "_load_" + name):
				getattr(self, "_load_" + name)()
			else:
				raise Exception('Invalid dataset.')

		# Data hasn't been splitted yet
		if self._loaded and not self._splits_loaded:
			if self._n_folds > 1:
				# K-Fold
				if self._folds_indices is None:
					self._folds_indices, _ = self._create_folds(self._n_folds)
				# Load current fold
				self._load_partition(self._folds_indices[self._current_fold])
			else:
				# Holdout
				self._load_holdout()

			self._splits_loaded = True

	@property
	def n_folds(self):
		return self._n_folds

	@n_folds.setter
	def n_folds(self, n_folds):
		# If folds == 1 -> hold out
		self._n_folds = n_folds

		self._clear_partitions()

	@property
	def holdout(self):
		return self._holdout

	@holdout.setter
	def holdout(self, holdout):
		# Define holdout portion for validation
		self._holdout = holdout

		self._clear_partitions()


	# Load next fold
	def next_fold(self):
		# Check if it is the last fold
		if self._current_fold + 1 < self._n_folds:
			self._current_fold += 1
		else:
			# Return to the first fold when the end is reached
			self._current_fold = 0

		# Mark the splits as not loaded in order to load them again (load next fold)
		self._splits_loaded = False

	# Set current fold
	def set_fold(self, fold):
		# Check if it is a valid fold number
		if fold < self._n_folds:
			self._current_fold = fold

			# Mark the splits as not loaded in order to load them again
			self._splits_loaded = False

	# Get indices of each fold for a given number of folds 
	def _create_folds(self, n_folds):
		skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self._seed)

		train_indices = []
		val_indices = []

		if self._big_dataset:
			for train, val in skf.split(self._df_trainval[self._x_col], self._df_trainval[self._y_col]):
				train_indices.append(train)
				val_indices.append(val)
		else:
			for train, val in skf.split(self._x_trainval, self._y_trainval):
				train_indices.append(train)
				val_indices.append(val)

		return train_indices, val_indices

	# Load train and val sets from trainval set using given indices
	def _load_partition(self, train_indices):
		n = self.size_trainval()
		train_mask = np.full(n, False, dtype=bool)
		train_mask[train_indices] = True
		val_mask = ~train_mask

		if self._big_dataset:
			self._df_train = self._df_trainval[train_mask]
			self._df_val = self._df_trainval[val_mask]
		else:
			self._x_train = self._x_trainval[train_mask]
			self._y_train = self._y_trainval[train_mask]
			self._x_val = self._x_trainval[val_mask]
			self._y_val = self._y_trainval[val_mask]


	# Load holdout splits
	def _load_holdout(self):
		if self._big_dataset:
			self._df_train, self._df_val = train_test_split(self._df_trainval, test_size=self._holdout, random_state=self._seed, stratify=self._df_trainval[self._y_col])
		else:
			self._x_train, self._x_val, self._y_train, self._y_val = train_test_split(self._x_trainval, self._y_trainval, test_size=self._holdout, random_state=self._seed, stratify=self._y_trainval)


	# Clear all the variables related to data partitions
	def _clear_partitions(self):
		self._folds_indices = None
		self._current_fold = 0

		self._df_train = None
		self._df_val = None
		self._x_train = None
		self._y_train = None
		self._x_val = None
		self._y_val = None

		self._splits_loaded = False


	def _load_cifar10(self):
		# Small dataset
		self._big_dataset = False

		# Set sample shape and number of classes
		self._sample_shape = (32, 32, 3)
		self._num_classes = 10

		# Load data
		(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

		# Save x and y
		self._x_trainval, self._y_trainval = x_train, y_train
		self._x_test, self._y_test = x_test, y_test

		# Mark dataset as loaded
		self._loaded = True
		
	def _load_cinic10(self):
		# Big dataset
		self._big_dataset = True

		# Load dataframes
		self._df_trainval = pd.read_csv(os.path.join(DATASETS_DIR, 'CINIC/data/trainval.csv'))
		self._df_test = pd.read_csv(os.path.join(DATASETS_DIR, 'CINIC/data/test.csv'))

		# Set x and y columns
		self._x_col = 'path'
		self._y_col = 'category'

		# Set base path for images
		self._base_path = os.path.join(DATASETS_DIR, 'CINIC/data/')

		# Set sample shape and number of classes
		self._sample_shape = (32, 32, 3)
		self._num_classes = 10

		# Check that images exist
		if self._check_dataframe_images(self._df_trainval, self._x_col, self._base_path) and \
        self._check_dataframe_images(self._df_test, self._x_col, self._base_path):
			# If everything is correct, mark dataset as loaded
			self._loaded = True

	def _load_mnist(self):
		# Small dataset
		self._big_dataset = False

		# Set sample shape and number of classes
		self._sample_shape = (32, 32, 1)
		self._num_classes = 10

		# Load data
		(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

		# Upscale
		x_train = self._resize_data(x_train, 32, 32, self.num_channels)
		x_test = self._resize_data(x_test, 32, 32, self.num_channels)

		# Save x and y
		self._x_trainval, self._y_trainval = x_train, y_train
		self._x_test, self._y_test = x_test, y_test

		# Mark dataset as loaded
		self._loaded = True

	def _load_wiki(self):
		# Small dataset
		self._big_dataset = False

		# Load dataframes
		df_trainval = pd.read_csv(os.path.join(DATASETS_DIR, 'wiki_crop/data_processed/trainval.csv'))
		df_test = pd.read_csv(os.path.join(DATASETS_DIR, 'wiki_crop/data_processed/test.csv'))
		
		# Base path for images
		base_path = os.path.join(DATASETS_DIR, 'wiki_crop/data_processed/')

		# Dataframe columns
		x_col = 'path'
		y_col = 'age_cat'

		# Set sample shape and number of classes
		self._sample_shape = (128, 128, 3)
		self._num_classes = 8

		# Load data from dataframe
		self._x_trainval, self._y_trainval = self._load_from_dataframe(df_trainval, x_col, y_col, base_path)
		self._x_test, self._y_test = self._load_from_dataframe(df_test, x_col, y_col, base_path)

		# Mark dataset as loaded
		self._loaded = True

	def _load_imdb(self):
		# Big dataset
		self._big_dataset = True

		# Load dataframes
		self._df_trainval = pd.read_csv(os.path.join(DATASETS_DIR, 'imdb_crop/data_processed/trainval.csv'))
		self._df_test = pd.read_csv(os.path.join(DATASETS_DIR, 'imdb_crop/data_processed/test.csv'))

		# Set x and y columns
		self._x_col = 'path'
		self._y_col = 'age_cat'

		# Set base path for images
		self._base_path = os.path.join(DATASETS_DIR, 'imdb_crop/data_processed/')

		# Set sample shape and number of classes
		self._sample_shape = (128, 128, 3)
		self._num_classes = 8

		# Check that images exist
		if self._check_dataframe_images(self._df_trainval, self._x_col, self._base_path) and \
        self._check_dataframe_images(self._df_test, self._x_col, self._base_path):
			# If everything is correct, mark dataset as loaded
			self._loaded = True

	def _load_retinopathy(self):
		# Big dataset
		self._big_dataset = True

		# Load dataframes
		self._df_trainval = pd.read_csv(os.path.join(DATASETS_DIR, 'retinopathy/data128/trainval.csv'))
		self._df_test = pd.read_csv(os.path.join(DATASETS_DIR, 'retinopathy/data128/test.csv'))

		# Set x and y columns
		self._x_col = 'path'
		self._y_col = 'category'

		# Set base path for images
		self._base_path = os.path.join(DATASETS_DIR, 'retinopathy/data128/')

		# Set sample shape and number of classes
		self._sample_shape = (128, 128, 3)
		self._num_classes = 5

		# Check that images exist
		if self._check_dataframe_images(self._df_trainval, self._x_col, self._base_path) and \
				self._check_dataframe_images(self._df_test, self._x_col, self._base_path):
			# If everything is correct, mark dataset as loaded
			self._loaded = True

	def _load_adience(self):
		# Big dataset
		self._big_dataset = True

		# Load dataframes
		self._df_trainval = pd.read_csv(os.path.join(DATASETS_DIR, 'adience/data256/trainval.csv'))
		self._df_test = pd.read_csv(os.path.join(DATASETS_DIR, 'adience/data256/test.csv'))

		# Set x and y columns
		self._x_col = 'path'
		self._y_col = 'category'

		# Set base path for images
		self._base_path = os.path.join(DATASETS_DIR, 'adience/data256/')

		# Set sample shape and number of classes
		self._sample_shape = (256, 256, 3)
		self._num_classes = 8

		# Check that images exist
		if self._check_dataframe_images(self._df_trainval, self._x_col, self._base_path) and \
				self._check_dataframe_images(self._df_test, self._x_col, self._base_path):
			# If everything is correct, mark dataset as loaded
			self._loaded = True

	def _load_historical(self):
		# Small dataset
		self._big_dataset = False

		# Load dataframes
		df_trainval = pd.read_csv(os.path.join(DATASETS_DIR, 'historical/data_processed/trainval.csv'))
		df_test = pd.read_csv(os.path.join(DATASETS_DIR, 'historical/data_processed/test.csv'))

		# Base path for images
		base_path = os.path.join(DATASETS_DIR, 'historical/data_processed/')

		# Dataframe columns
		x_col = 'path'
		y_col = 'category'

		# Set sample shape and number of classes
		self._sample_shape = (256, 256, 3)
		self._num_classes = 5

		# Load data from dataframe
		self._x_trainval, self._y_trainval = self._load_from_dataframe(df_trainval, x_col, y_col, base_path)
		self._x_test, self._y_test = self._load_from_dataframe(df_test, x_col, y_col, base_path)

		# Mark dataset as loaded
		self._loaded = True

	def _load_fgnet(self):
		# Small dataset
		self._big_dataset = False

		# Load dataframes
		df_trainval = pd.read_csv(os.path.join(DATASETS_DIR, 'fgnet/data_processed/trainval.csv'))
		df_test = pd.read_csv(os.path.join(DATASETS_DIR, 'fgnet/data_processed/test.csv'))

		# Base path for images
		base_path = os.path.join(DATASETS_DIR, 'fgnet/data_processed/')

		# Dataframe columns
		x_col = 'path'
		y_col = 'category'

		# Set sample shape and number of classes
		self._sample_shape = (128, 128, 3)
		self._num_classes = 6

		# Load data from dataframe
		self._x_trainval, self._y_trainval = self._load_from_dataframe(df_trainval, x_col, y_col, base_path)
		self._x_test, self._y_test = self._load_from_dataframe(df_test, x_col, y_col, base_path)

		# Mark dataset as loaded
		self._loaded = True

	# Fully load x and y from dataframe
	def _load_from_dataframe(self, df, x_col, y_col, base_path):
		x = []
		y = np.array(list(df[y_col]))

		for path in df['path']:
			img = imread(os.path.join(base_path, path))

			if len(img.shape) < 3:
				img = np.stack((img,)*3, axis=-1)

			x.append(img)
		
		x = np.concatenate([arr[np.newaxis] for arr in x])

		return x, y

	# Resize array of images
	def _resize_data(self, x, width, height, channels):
		x_resized = np.zeros((x.shape[0], width, height, channels))
		for i, img in enumerate(x):
			img_resized = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
			# cv2 returns 2 dims array when using non rgb images but we need 3 dims
			if len(img_resized.shape) < 3:
				img_resized = np.expand_dims(img_resized, axis=-1)
			x_resized[i] = img_resized

		return x_resized

	def generate_train(self, batch_size, augmentation):
		# Load dataset if not loaded
		self.load(self._name)

		if self._big_dataset:
			return BigGenerator(self._df_train, self._base_path, self._num_classes, self._x_col, self._y_col, mean=self.mean_train, std=self.std_train, batch_size=batch_size, augmentation=augmentation)
		else:
			return SmallGenerator(self._x_train, self._y_train, self._num_classes, mean=self.mean_train, std=self.std_train, batch_size=batch_size, augmentation=augmentation)

	def generate_val(self, batch_size):
		# Load dataset if not loaded
		self.load(self._name)

		if self._big_dataset:
			return BigGenerator(self._df_val, self._base_path, self._num_classes, self._x_col, self._y_col, mean=self.mean_train, std=self.std_train, batch_size=batch_size)
		else:
			return SmallGenerator(self._x_val, self._y_val, self._num_classes, mean=self.mean_train, std=self.std_train, batch_size=batch_size)

	def generate_test(self, batch_size):
		# Load dataset if not loaded
		self.load(self._name)

		if self._big_dataset:
			return BigGenerator(self._df_test, self._base_path, self._num_classes, self._x_col, self._y_col, mean=self.mean_train, std=self.std_train, batch_size=batch_size)
		else:
			return SmallGenerator(self._x_test, self._y_test, self._num_classes, mean=self.mean_train, std=self.std_train, batch_size=batch_size)


	def _check_dataframe_images(self, df, x_col, base_path):
		for path in df[x_col]:
			if not os.path.exists(os.path.join(base_path, path)):
				return False
		return True

	def _mean_small(self, x):
		return x.mean()

	def _mean_big(self, df):
		paths = df[self._x_col].values
		count = df.shape[0]

		with Pool(7) as p:
			func = partial(parallel_img_sum, self._base_path)
			summ = p.map(func, paths)

		return np.array(summ).sum() / (np.array(self._sample_shape).prod() * count)

	def _std_small(self, x):
		return x.std()

	def _std_big(self, df, mean):
		paths = df[self._x_col].values
		n = df.shape[0]

		with Pool(7) as p:
			func = partial(parallel_variance_sum, self._base_path, mean, n)
			sums = p.map(func, paths)

		std = np.sqrt(np.sum(sums) / np.prod(self._sample_shape))

		return std

	@property
	def mean_train(self):
		# Load dataset if not loaded
		# self.load(self._name)

		if not self._mean_train:
			self._mean_train = self._mean_big(self._df_train) if self._big_dataset else self._mean_small(self._x_train)

		return self._mean_train

	@property
	def mean_val(self):
		# Load dataset if not loaded
		# self.load(self._name)

		if not self._mean_val:
			self._mean_val = self._mean_big(self._df_val) if self._big_dataset else self._mean_small(self._x_val)
		return self._mean_val

	@property
	def mean_test(self):
		# Load dataset if not loaded
		# self.load(self._name)

		if not self._mean_test:
			self._mean_test = self._mean_big(self._df_test) if self._big_dataset else self._mean_small(self._x_test)
		return self._mean_test

	@property
	def std_train(self):
		# Load dataset if not loaded
		# self.load(self._name)

		if not self._std_train:
			self._std_train = self._std_big(self._df_train, self.mean_train) if self._big_dataset else self._std_small(self._x_train)

		return self._std_train

	@property
	def std_val(self):
		# Load dataset if not loaded
		# self.load(self._name)

		if not self._std_val:
			self._std_val = self._std_big(self._df_val, self.mean_val) if self._big_dataset else self._std_small(self._x_val)
		return self._std_val

	@property
	def std_test(self):
		# Load dataset if not loaded
		# self.load(self._name)

		if not self._std_test:
			self._std_test = self._std_big(self._df_test, self.mean_test) if self._big_dataset else self._std_small(self._x_test)
		return self._std_test

	@property
	def num_classes(self):
		return self._num_classes if self._num_classes is not None else 0

	@num_classes.setter
	def num_classes(self, num_classes):
		self._num_classes = num_classes

	@num_classes.deleter
	def num_classes(self):
		del self._num_classes

	@property
	def sample_shape(self):
		return self._sample_shape if self._sample_shape is not None else ()

	@sample_shape.setter
	def sample_shape(self, sample_shape):
		self._sample_shape = sample_shape

	@sample_shape.deleter
	def sample_shape(self):
		del self._sample_shape

	def size_trainval(self):
		"""
		Get dataset train size.
		:return: number of samples.
		"""
		# Load dataset if not loaded
		# self.load(self._name)
		# Disabled because of recursion problem		

		return 0 if not self._loaded else (self._df_trainval.shape[0] if self._big_dataset else self._y_trainval.shape[0])

	def size_train(self):
		"""
		Get dataset train size.
		:return: number of samples.
		"""
		# Load dataset if not loaded
		self.load(self._name)

		return 0 if not self._splits_loaded else (self._df_train.shape[0] if self._big_dataset else self._y_train.shape[0])

	def size_val(self):
		"""
		Get dataset val size.
		:return: number of samples.
		"""
		# Load dataset if not loaded
		self.load(self._name)

		return 0 if not self._splits_loaded else (self._df_val.shape[0] if self._big_dataset else self._y_val.shape[0])

	def size_test(self):
		"""
		Get dataset test size.
		:return: number of samples.
		"""
		# Load dataset if not loaded
		self.load(self._name)

		return 0 if not self._splits_loaded else (self._df_test.shape[0] if self._big_dataset else self._y_test.shape[0])

	def num_batches_train(self, batch_size):
		"""
		Get number of train batches for a given batch size.
		:param batch_size: batch size.
		:return: number of batches.
		"""
		return math.ceil(self.size_train() / batch_size)

	def num_batches_val(self, batch_size):
		"""
		Get number of val batches for a given batch size.
		:param batch_size: batch size.
		:return: number of batches.
		"""
		return math.ceil(self.size_val() / batch_size)

	def num_batches_test(self, batch_size):
		"""
		Get number of test batches for a given batch size.
		:param batch_size: batch size.
		:return: number of batches.
		"""
		return math.ceil(self.size_test() / batch_size)

	def get_class_weights(self):
		"""
		Get class weights that you can use to counter-act the dataset unbalance.
		Class weights are calculated based on the frequency of each class.
		:return: dictionary that contains the weight for each class.
		"""
		# Load dataset if not loaded
		self.load(self._name)

		# No weights if not loaded
		if not self._splits_loaded:
			return {}

		y_label = self._df_train[self._y_col] if self._big_dataset else self._y_train

		return compute_class_weight('balanced', np.unique(y_label), y_label.ravel())

	@property
	def num_channels(self):
		"""
		Get number of channels of the images.
		:return: number of channels.
		"""
		return len(self.sample_shape) == 3 and self.sample_shape[2] or 1

	@property
	def img_size(self):
		"""
		Get image size for squared images.
		:return: image size (integer).
		"""
		return self.sample_shape[0]

	def is_rgb(self):
		"""
		Check whether the images are RGB.
		:return:
		"""
		return self.num_channels == 3

	@property
	def y_train(self):
		return (self._df_train[self._y_col].values if self._big_dataset else self.y_train) if self._loaded else np.array([])

	@property
	def y_val(self):
		return (self._df_val[self._y_col].values if self._big_dataset else self.y_val) if self._loaded else np.array([])

	@property
	def y_test(self):
		return (self._df_test[self._y_col].values if self._big_dataset else self._y_test) if self._loaded else np.array([])
