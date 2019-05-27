import os
import json
import tensorflow as tf
from keras import backend as K
import gc
from experiment import Experiment

class ExperimentSet:
	"""
	Set of experiments that can be executed sequentially.
	"""
	def __init__(self, experiments=[]):
		self._experiments = experiments

	# PROPERTIES

	@property
	def experiments(self):
		return self._experiments

	@experiments.setter
	def experiments(self, experiments):
		self._experiments = experiments

	@experiments.deleter
	def experiments(self):
		del self._experiments

	# # # # # #

	def _validate_experiments(self):
		"""
		Validate experiments list
		:return: None
		"""
		if not type(self.experiments) is list:
			if type(self.experiments) is tuple:
				self.experiments = list(self.experiments)
			else:
				self.experiments = []

	def add_experiment(self, experiment):
		"""
		Add experiment to experiments list.
		:param experiment: new experiment that will be added.
		:return: None
		"""
		self._validate_experiments()
		self.experiments.append(experiment)

	def remove_experiment(self, name):
		"""
		Remove experiment from the experiments list by its name.
		:param name: name of the experiment that will be removed.
		:return: None
		"""
		self._validate_experiments()
		for experiment in self.experiments:
			if experiment.name == name:
				self.experiments.remove(experiment)

	def clear_experiments(self):
		"""
		Clear experiments list.
		:return:
		"""
		self.experiments = []

	def load_from_file(self, path):
		"""
		Load experiments from json file.
		:param path: path of the json file.
		:return: None
		"""

		# Load JSON file
		with open(path) as f:
			configs = json.load(f)

		# Add experiments
		for config in configs:
			executions = 'executions' in config and int(config['executions']) or 1
			for execution in range(1, executions + 1):
				exec_config = config.copy()
				if 'name' in exec_config:
					exec_config['name'] += "_{}".format(execution)
				exec_config['checkpoint_dir'] += "/{}".format(execution)
				experiment = Experiment()
				experiment.set_config(exec_config)
				self.add_experiment(experiment)

	def save_to_file(self, path):
		"""
		Save experiments set to json file
		:param path: path of the saved file.
		:return: None
		"""
		configs = []

		for experiment in self.experiments:
			configs.append(experiment.get_config())

		json.dump(configs, path)


	def run_all(self, gpu_number=0):
		"""
		Execute all the experiments
		:param gpu_number: GPU that will be used.
		:return: None
		"""
		for experiment in self.experiments:
			os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)
			with tf.device('/device:GPU:' + str(gpu_number)):
				if not experiment.finished and experiment.task != 'test': # 'train' or 'both'
					experiment.run()
				if experiment.task != 'train': # 'test' or 'both'
					experiment.evaluate()
			# Clear session
			K.clear_session()

			# Free memory
			del experiment
			gc.collect()
