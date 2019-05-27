import keras
import numpy as np
import resnet
from net_keras import Net
import os
import glob
import time
import click
import pickle
import h5py
from scipy import io as spio
from losses import qwk_loss, make_cost_matrix
from metrics import np_quadratic_weighted_kappa, top_2_accuracy, top_3_accuracy, \
	minimum_sensitivity, accuracy_off1
from dataset import Dataset
from sklearn.metrics import confusion_matrix
import math
import gc
from keras import backend as K

class Experiment:
	"""
	Class that represents a single experiment that can be run and evaluated.
	"""

	def __init__(self, name='unnamed', db='100', net_type='vgg19', batch_size=128, epochs=100,
				 checkpoint_dir='checkpoint', loss='categorical_crossentropy', activation='relu',
				 final_activation='softmax', f_a_params = {}, use_tau=True, spp_alpha=1.0, lr=0.1, momentum=0.9, dropout=0, task='both', workers=4,
				 queue_size=1024, val_metrics=['loss', 'acc'], rescale_factor=0, augmentation={}):
		self._name = name
		self._db = db
		self._net_type = net_type
		self._batch_size = batch_size
		self._epochs = epochs
		self._checkpoint_dir = checkpoint_dir
		self._loss = loss
		self._activation = activation
		self._use_tau = use_tau
		self._final_activation = final_activation
		self._f_a_params = f_a_params
		self._spp_alpha = spp_alpha
		self._lr = lr
		self._momentum = momentum
		self._dropout = dropout
		self._task = task
		self._finished = False
		self._workers = workers
		self._queue_size = queue_size
		self._val_metrics = val_metrics
		self._rescale_factor = rescale_factor
		self._augmentation = augmentation

		self._best_metric = None

		# Model and results file names
		self.model_file = 'model.h5'
		self.best_model_file = 'best_model.h5'
		self.model_file_extra = 'model.txt'
		self.csv_file = 'results.csv'
		self.evaluation_file = 'evaluation.pickle'

	def set_auto_name(self):
		"""
		Set experiment name based on experiment parameters.
		:return: None
		"""
		self.name = self.get_auto_name()

	def get_auto_name(self):
		"""
		Get experiment auto-generated name based on experiment parameters.
		:return: experiment auto-generated name.
		"""
		return "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(self.db, self.net_type, self.batch_size, self.activation,
														 self.loss,
														 self.final_activation,
														 self.spp_alpha, self.lr,
														 self.momentum, self.dropout)

	# PROPERTIES

	@property
	def name(self):
		return self._name

	@name.setter
	def name(self, name):
		self._name = name

	@name.deleter
	def name(self):
		del self._name

	@property
	def db(self):
		return self._db

	@db.setter
	def db(self, db):
		self._db = db

	@db.deleter
	def db(self):
		del self._db

	@property
	def net_type(self):
		return self._net_type

	@net_type.setter
	def net_type(self, net_type):
		self._net_type = net_type

	@net_type.deleter
	def net_type(self):
		del self._net_type

	@property
	def batch_size(self):
		return self._batch_size

	@batch_size.setter
	def batch_size(self, batch_size):
		self._batch_size = batch_size

	@batch_size.deleter
	def batch_size(self):
		del self._batch_size

	@property
	def epochs(self):
		return self._epochs

	@epochs.setter
	def epochs(self, epochs):
		self._epochs = epochs

	@epochs.deleter
	def epochs(self):
		del self._epochs

	@property
	def checkpoint_dir(self):
		return self._checkpoint_dir

	@checkpoint_dir.setter
	def checkpoint_dir(self, checkpoint_dir):
		self._checkpoint_dir = checkpoint_dir

	@checkpoint_dir.deleter
	def checkpoint_dir(self):
		del self._checkpoint_dir

	@property
	def loss(self):
		return self._loss

	@loss.setter
	def loss(self, loss):
		self._loss = loss

	@loss.deleter
	def loss(self):
		del self._loss

	@property
	def activation(self):
		return self._activation

	@activation.setter
	def activation(self, activation):
		self._activation = activation

	@activation.deleter
	def activation(self):
		del self._activation

	@property
	def final_activation(self):
		return self._final_activation

	@final_activation.setter
	def final_activation(self, final_activation):
		self._final_activation = final_activation

	@final_activation.deleter
	def final_activation(self):
		del self._final_activation

	@property
	def f_a_params(self):
		return self._f_a_params

	@f_a_params.setter
	def f_a_params(self, f_a_params):
		self._f_a_params = f_a_params

	@f_a_params.deleter
	def f_a_params(self):
		del self._f_a_params

	@property
	def use_tau(self):
		return self._use_tau

	@use_tau.setter
	def use_tau(self, use_tau):
		self._use_tau = use_tau

	@use_tau.deleter
	def use_tau(self):
		del self._use_tau

	@property
	def spp_alpha(self):
		return self._spp_alpha

	@spp_alpha.setter
	def spp_alpha(self, spp_alpha):
		self._spp_alpha = spp_alpha

	@spp_alpha.deleter
	def spp_alpha(self):
		del self._spp_alpha

	@property
	def lr(self):
		return self._lr

	@lr.setter
	def lr(self, lr):
		self._lr = lr

	@lr.deleter
	def lr(self):
		del self._lr

	@property
	def momentum(self):
		return self._momentum

	@momentum.setter
	def momentum(self, momentum):
		self._momentum = momentum

	@momentum.deleter
	def momentum(self):
		del self._momentum

	@property
	def dropout(self):
		return self._dropout

	@dropout.setter
	def dropout(self, dropout):
		self._dropout = dropout

	@dropout.deleter
	def dropout(self):
		del self._dropout

	@property
	def task(self):
		return self._task

	@task.setter
	def task(self, task):
		self._task = task

	@task.deleter
	def task(self):
		del self._task

	@property
	def finished(self):
		return self._finished

	@finished.setter
	def finished(self, finished):
		self._finished = finished

	@finished.deleter
	def finished(self):
		del self._finished

	@property
	def workers(self):
		return self._workers

	@workers.setter
	def workers(self, workers):
		self._workers = workers

	@workers.deleter
	def workers(self):
		del self._workers

	@property
	def queue_size(self):
		return self._workers

	@queue_size.setter
	def queue_size(self, queue_size):
		self._queue_size = queue_size

	@queue_size.deleter
	def queue_size(self):
		del self._queue_size

	@property
	def val_metrics(self):
		return self._val_metrics

	@val_metrics.setter
	def val_metrics(self, val_metrics):
		self._val_metrics = val_metrics

	@val_metrics.deleter
	def val_metrics(self):
		del self._val_metrics

	@property
	def rescale_factor(self):
		return self._rescale_factor

	@rescale_factor.setter
	def rescale_factor(self, rescale_factor):
		self._rescale_factor = rescale_factor

	@rescale_factor.deleter
	def rescale_factor(self):
		del self._rescale_factor

	@property
	def augmentation(self):
		return self._augmentation

	@augmentation.setter
	def augmentation(self, augmentation):
		self._augmentation = augmentation

	@augmentation.deleter
	def augmentation(self):
		del self._augmentation

	@property
	def best_metric(self):
		return self._best_metric

	def new_metric(self, metric, maximize=False):
		"""
		Updates best metric if metric provided is better than the best metric stored.
		:param metric: new metric.
		:param maximize: maximize metric instead of minimizing.
		:return: True if new metric is better than best metric or False otherwise.
		"""
		if self._best_metric is None or (
						maximize and metric > self._best_metric or not maximize and metric <= self._best_metric):
			self._best_metric = metric
			return True
		return False

	# # # # # # #

	def run(self):
		"""
		Run training process.
		:return: None
		"""

		print('=== RUNNING {} ==='.format(self.name))

		# Garbage collection
		gc.collect()

		# Initial epoch. 0 by default
		start_epoch = 0

		# Load training status
		if os.path.isfile(os.path.join(self.checkpoint_dir, self.model_file_extra)):
			# Continue from the epoch where we were and load the best metric
			with open(os.path.join(self.checkpoint_dir, self.model_file_extra), 'r') as f:
				start_epoch = int(f.readline())
				self.new_metric(float(f.readline()))

		if start_epoch >= self.epochs:
			print("Training already finished. Skipping...")
			return

		# Train data generator
		train_datagen = keras.preprocessing.image.ImageDataGenerator(
			rescale=self.rescale_factor,
			**self.augmentation
		)

		# Augmentation for validation / test
		eval_augmentation = {k: v for k, v in self.augmentation.items() if
							 k == 'featurewise_center' or k == 'featurewise_std_normalization'}

		# Validation data generator
		val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=self.rescale_factor, **eval_augmentation)

		# Get database paths
		train_path, val_path, _ = self.get_db_path(self.db)

		# Check that database exists and paths are correct
		if train_path == '' or val_path == '':
			raise Exception('Invalid database. Choose one of: Retinopathy or Adience.')

		# Load datasets
		ds_train = Dataset(train_path)
		# ds_train.standardize_data()
		ds_val = Dataset(val_path)
		# ds_val.standardize_data()

		# Get dataset details
		num_classes = ds_train.num_classes
		num_channels = ds_train.num_channels
		img_size = ds_train.img_size

		# Fit for zca_whitening, featurewise_center, featurewise_std_normalization
		if 'zca_whitening' in self.augmentation or 'featurewise_center' in self.augmentation or 'featurewise_std_normalization' in self.augmentation:
			train_datagen.fit(ds_train.x)
			val_datagen.mean = train_datagen.mean
			val_datagen.std = train_datagen.std

		# Train data generator used for training
		train_generator = train_datagen.flow(
			ds_train.x,
			ds_train.y,
			batch_size=self.batch_size
		)

		# Validation generator
		val_generator = val_datagen.flow(
			ds_val.x,
			ds_val.y,
			batch_size=self.batch_size,
			shuffle=True
		)

		# Calculate the number of steps per epoch
		steps = (len(ds_train.y) * 1) // self.batch_size
		steps_val = ds_val.num_batches(self.batch_size)

		# Get class weights based on frequency
		class_weight = ds_train.get_class_weights()

		# Free dataset object
		del ds_train
		del ds_val
		gc.collect()

		# Learning rate scheduler callback
		def learning_rate_scheduler(epoch):
			lr = self.lr * np.exp(-0.01 * epoch)

			return lr

		# Save epoch callback for training process
		def save_epoch(epoch, logs):
			# Check whether new metric is better than best metric
			if (self.new_metric(logs['val_loss'])):
				model.save(os.path.join(self.checkpoint_dir, self.best_model_file))
				print("Best model saved.")

			with open(os.path.join(self.checkpoint_dir, self.model_file_extra), 'w') as f:
				f.write(str(epoch + 1))
				f.write('\n' + str(self.best_metric))


		save_epoch_callback = keras.callbacks.LambdaCallback(
			on_epoch_end=save_epoch
		)

		# NNet object
		net_object = Net(img_size, self.activation, self.final_activation, self.f_a_params, self.use_tau, num_channels, num_classes,
						 self.spp_alpha,
						 self.dropout)

		model = self.get_model(net_object, self.net_type)

		# Create checkpoint dir if not exists
		if not os.path.isdir(self.checkpoint_dir):
			os.makedirs(self.checkpoint_dir)

		# Check whether a saved model exists
		if os.path.isfile(os.path.join(self.checkpoint_dir, self.model_file)):
			print("===== RESTORING SAVED MODEL =====")
			model.load_weights(os.path.join(self.checkpoint_dir, self.model_file))

		# Create the cost matrix that will be used to compute qwk
		cost_matrix = K.constant(make_cost_matrix(num_classes), dtype=K.floatx())

		# Cross-entropy loss by default
		loss = 'categorical_crossentropy'

		# Quadratic Weighted Kappa loss
		if self.loss == 'qwk':
			loss = qwk_loss(cost_matrix)

		# Only accuracy for training.
		# Computing QWK for training properly is too expensive
		metrics = ['accuracy']

		# Compile the keras model
		model.compile(
			optimizer=keras.optimizers.Adam(lr=self.lr),  # keras.optimizers.SGD(self.lr, 0.9),
			loss=loss,
			metrics=metrics
		)

		# Print model summary
		model.summary()

		# Run training
		model.fit_generator(train_generator, epochs=self.epochs,
							initial_epoch=start_epoch,
							steps_per_epoch=steps,
							callbacks=[keras.callbacks.LearningRateScheduler(learning_rate_scheduler),
									   keras.callbacks.ModelCheckpoint(
										   os.path.join(self.checkpoint_dir, self.model_file)),
									   save_epoch_callback,
									   keras.callbacks.CSVLogger(os.path.join(self.checkpoint_dir, self.csv_file),
																	append=True),
									   keras.callbacks.TensorBoard(log_dir=self.checkpoint_dir),
									   keras.callbacks.TerminateOnNaN(),
									   ],
							workers=self.workers,
							use_multiprocessing=False,
							max_queue_size=self.queue_size,
							class_weight=class_weight,
							validation_data=val_generator,
							validation_steps=steps_val,
							verbose=2
							)


		self.finished = True

		# Mark the training as finished in the checkpoint file
		with open(os.path.join(self.checkpoint_dir, self.model_file_extra), 'w') as f:
			f.write(str(self.epochs))
			f.write('\n' + str(self.best_metric))

		# Free objects
		del model
		del cost_matrix
		del train_datagen
		del train_generator
		del val_datagen
		del val_generator

	def evaluate(self):
		"""
		Run evaluation on test data.
		:return: None
		"""
		print('=== EVALUATING {} ==='.format(self.name))

		# Garbage collection
		gc.collect()

		# Check if best model file exists
		if not os.path.isfile(os.path.join(self.checkpoint_dir, self.best_model_file)):
			print('Best model file not found')
			return

		# Check if model was already evaluated
		if os.path.isfile(os.path.join(self.checkpoint_dir, self.evaluation_file)):
			print('Model already evaluated')
			return

		paths = self.get_db_path(self.db)

		all_metrics = {}

		# Augmentation for validation / test
		eval_augmentation = {k: v for k, v in self.augmentation.items() if
							 k == 'featurewise_center' or k == 'featurewise_std_normalization'}

		mean = 0
		std = 0

		for path, set in zip(paths, ['Train', 'Validation', 'Test']):
			print('\n=== {} dataset ===\n'.format(set))

			# Load test dataset
			ds_test = Dataset(path)
			# ds_test.standardize_data()

			# Get dataset details
			num_classes = ds_test.num_classes
			num_channels = ds_test.num_channels
			img_size = ds_test.img_size

			# Validation data generator
			test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=self.rescale_factor,
																		   **eval_augmentation)

			# Save mean and std of train set
			if set == 'Train':
				# Fit for zca_whitening, featurewise_center, featurewise_std_normalization
				if 'zca_whitening' in self.augmentation or 'featurewise_center' in self.augmentation or 'featurewise_std_normalization' in self.augmentation:
					test_datagen.fit(ds_test.x)
					mean = test_datagen.mean
					std = test_datagen.std
			else:
				if 'zca_whitening' in self.augmentation or 'featurewise_center' in self.augmentation or 'featurewise_std_normalization' in self.augmentation:
					test_datagen.mean = mean
					test_datagen.std = std

			# Test generator
			test_generator = test_datagen.flow(
				ds_test.x,
				ds_test.y,
				batch_size=self.batch_size,
				shuffle=False
			)

			# NNet object
			net_object = Net(img_size, self.activation, self.final_activation, self.f_a_params, self.use_tau, num_channels,
							 num_classes,
							 self.spp_alpha,
							 self.dropout)

			model = self.get_model(net_object, self.net_type)

			# Restore weights
			model.load_weights(os.path.join(self.checkpoint_dir, self.best_model_file))

			# Get predictions
			test_generator.reset()
			predictions = model.predict_generator(test_generator, verbose=1)

			metrics = self.compute_metrics(ds_test.y, predictions, num_classes)
			self.print_metrics(metrics)

			all_metrics[set] = metrics

			# Free objects
			del ds_test
			del test_datagen
			del test_generator
			del net_object
			del model
			del predictions
			del metrics
			gc.collect()

		with open(os.path.join(self.checkpoint_dir, self.evaluation_file), 'wb') as f:
			pickle.dump({'config': self.get_config(), 'metrics': all_metrics}, f)

	def compute_metrics(self, y_true, y_pred, num_classes):
		# Calculate metric
		sess = keras.backend.get_session()
		qwk = np_quadratic_weighted_kappa(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), 0,
										  num_classes - 1)
		ms = minimum_sensitivity(y_true, y_pred)
		mae = sess.run(K.mean(keras.losses.mean_absolute_error(y_true, y_pred)))
		mse = sess.run(K.mean(keras.losses.mean_squared_error(y_true, y_pred)))
		acc = sess.run(K.mean(keras.metrics.categorical_accuracy(y_true, y_pred)))
		top2 = sess.run(top_2_accuracy(y_true, y_pred))
		top3 = sess.run(top_3_accuracy(y_true, y_pred))
		off1 = accuracy_off1(y_true, y_pred)
		conf_mat = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))

		metrics = {
			'QWK': qwk,
			'MS': ms,
			'MAE': mae,
			'MSE': mse,
			'CCR': acc,
			'Top-2': top2,
			'Top-3': top3,
			'1-off': off1,
			'Confusion matrix': conf_mat
		}

		return metrics

	def print_metrics(self, metrics):
		print('Confusion matrix :\n{}'.format(metrics['Confusion matrix']))
		print('QWK: {:.4f}'.format(metrics['QWK']))
		print('CCR: {:.4f}'.format(metrics['CCR']))
		print('Top-2: {:.4f}'.format(metrics['Top-2']))
		print('Top-3: {:.4f}'.format(metrics['Top-3']))
		print('1-off: {:.4f}'.format(metrics['1-off']))
		print('MAE: {:.4f}'.format(metrics['MAE']))
		print('MSE: {:.4f}'.format(metrics['MSE']))
		print('MS: {:.4f}'.format(metrics['MS']))

	def get_model(self, net_object, name):
		if name == 'conv128':
			model = net_object.conv128()
		elif name == 'beckhamresnet':
			model = net_object.beckham_resnet()
		else:
			raise Exception('Invalid net type. You must select one of these: vgg19, conv128')

		return model

	def get_db_path(self, db):
		"""
		Get dataset path for train, validation and test for a given database name.
		:param db: database name.
		:return: train path, validation path, test path.
		"""
		if db.lower() == 'retinopathy':
			return "../../retinopathy/128/train", "../../retinopathy/128/val", "../../retinopathy/128/test"
		elif db.lower() == 'adience':
			return "../../adience/adience_train_256.h5", "../../adience/adience_val_256.h5", "../../adience/adience_test_256.h5"
		else:
			return "", "", ""

	def get_config(self):
		"""
		Get config dictionary from object config.
		:return: config dictionary.
		"""
		return {
			'name': self.name,
			'db': self.db,
			'net_type': self.net_type,
			'batch_size': self.batch_size,
			'epochs': self.epochs,
			'checkpoint_dir': self.checkpoint_dir,
			'loss': self.loss,
			'activation': self.activation,
			'use_tau' : self.use_tau,
			'final_activation': self.final_activation,
			'f_a_params': self.f_a_params,
			'spp_alpha': self.spp_alpha,
			'lr': self.lr,
			'momentum': self.momentum,
			'dropout': self.dropout,
			'task': self.task,
			'workers': self.workers,
			'queue_size': self.queue_size,
			'val_metrics': self.val_metrics,
			'rescale_factor': self.rescale_factor,
			'augmentation': self.augmentation
		}

	def set_config(self, config):
		"""
		Set object config from config dictionary
		:param config: config dictionary.
		:return: None
		"""
		self.db = 'db' in config and config['db'] or '10'
		self.net_type = 'net_type' in config and config['net_type'] or 'vgg19'
		self.batch_size = 'batch_size' in config and config['batch_size'] or 128
		self.epochs = 'epochs' in config and config['epochs'] or 100
		self.checkpoint_dir = 'checkpoint_dir' in config and config['checkpoint_dir'] or 'results'
		self.loss = 'loss' in config and config['loss'] or 'crossentropy'
		self.activation = 'activation' in config and config['activation'] or 'relu'
		self.final_activation = 'final_activation' in config and config['final_activation'] or 'softmax'
		self.f_a_params = config['f_a_params'] if 'f_a_params' in config else {}
		self.use_tau = config['use_tau'] if 'use_tau' in config and config['use_tau'] else False
		self.spp_alpha = 'spp_alpha' in config and config['spp_alpha'] or 0
		self.lr = 'lr' in config and config['lr'] or 0.1
		self.momentum = 'momentum' in config and config['momentum'] or 0
		self.dropout = 'dropout' in config and config['dropout'] or 0
		self.task = 'task' in config and config['task'] or 'both'
		self.workers = 'workers' in config and config['workers'] or 4
		self.queue_size = 'queue_size' in config and config['queue_size'] or 1024
		self.val_metrics = 'val_metrics' in config and config['val_metrics'] or ['acc', 'loss']
		self.rescale_factor = 'rescale_factor' in config and config['rescale_factor'] or 0
		self.augmentation = 'augmentation' in config and config['augmentation'] or {}

		if 'name' in config:
			self.name = config['name']
		else:
			self.set_auto_name()

	def save_to_file(self, path):
		"""
		Save experiment to pickle file.
		:param path: path where pickle file will be saved.
		:return: None
		"""
		pickle.dump(self.get_config(), path)

	def load_from_file(self, path):
		"""
		Load experiment from pickle file.
		:param path: path where pickle file is located.
		:return: None
		"""
		if os.path.isfile(path):
			self.set_config(pickle.load(path))
